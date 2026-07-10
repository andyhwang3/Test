"""
agents/audio_stt_agent.py

3단계 Agent: 오디오 스트림을 무손실 단일 채널로 추출한 뒤 고정밀 STT를 수행한다.
Demucs AI의 분리 능력을 극대화하여 원본(_raw_dump), 목소리(_vocals), 배경음(_bgm) 
총 3종의 독립된 wave 파일 패키지를 아웃풋 폴더에 완벽하게 분리 보존한다.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from schema import AudioSTTResult, TranscriptSegment, MXFBaseMeta


class AudioSTTAgent(BaseAgent):
    name = "audio_stt_agent"

    def __init__(
        self,
        model_size: str = "turbo",
        device: str = "cuda",
        gpu_id: int = 0,
        compute_type: str = "float16",
        ffmpeg_bin: str = "ffmpeg",
        dialogue_track_index: Optional[int] = None
    ):
        self.model_size = model_size
        self.device = device
        self.gpu_id = gpu_id
        self.device_str = f"{device}:{gpu_id}" if device == "cuda" else device
        self.compute_type = compute_type
        self.ffmpeg_bin = ffmpeg_bin
        self.dialogue_track_index = dialogue_track_index
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            model_to_load = self.model_size
            self._model = WhisperModel(
                model_to_load,
                device=self.device,
                device_index=self.gpu_id,
                compute_type=self.compute_type
            )
        return self._model

    def run(self, context: Dict[str, Any]) -> AudioSTTResult:
        file_path = context["file_path"]
        mxf_meta: MXFBaseMeta = context["mxf_meta"]
        output_dir = context.get("output_dir")

        base_dir = Path(output_dir) if output_dir else Path(file_path).resolve().parent
        stem_name = Path(file_path).stem
        
        raw_wav_path = base_dir / f"{stem_name}_raw_dump.wav"
        final_vocals_path = base_dir / f"{stem_name}_vocals.wav"  # 클린 목소리
        final_bgm_path = base_dir / f"{stem_name}_bgm.wav"        # 클린 BGM / MR
        final_txt_path = base_dir / f"{stem_name}_transcript.txt"

        # 1. MXF 오리지널 오디오 무손실 추출
        print("🔊 [STT 엔진] MXF 오디오 스트림 추출 중...")
        cmd_extract = [
            self.ffmpeg_bin, "-y",
            "-i", file_path
        ]
        if self.dialogue_track_index is not None:
            cmd_extract.extend(["-map", f"0:a:{self.dialogue_track_index}"])
        else:
            cmd_extract.append("-vn")
            
        cmd_extract.extend([
            "-ac", "1",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            str(raw_wav_path)
        ])
        subprocess.run(cmd_extract, capture_output=True)

        # 2. 내장형 토치오디오 몽키 패치 기반 Demucs 안전 분리 연산
        print("🧠 [STT 엔진] Demucs AI 오디오 소스 정밀 분리 가동 (Vocals & BGM 병렬 추출)...")
        
        try:
            import torchaudio
            import soundfile as sf
            from demucs.__main__ import main as demucs_main

            def windows_safe_save(path, src, sample_rate, bits_per_sample=16, **kwargs):
                data = src.cpu().detach().numpy().T
                subtype = 'PCM_16' if bits_per_sample == 16 else 'PCM_24'
                sf.write(path, data, sample_rate, subtype=subtype)

            original_save = torchaudio.save
            original_argv = sys.argv
            torchaudio.save = windows_safe_save

            sys.argv = [
                "demucs",
                "--two-stems", "vocals",
                "-d", self.device_str,
                "-o", str(base_dir),
                str(raw_wav_path)
            ]
            
            demucs_main()

            torchaudio.save = original_save
            sys.argv = original_argv

        except Exception as e:
            print(f"⚠️ [STT 엔진] Demucs 가동 중 내부 오류 발생: {e}")

        demucs_vocal_src = base_dir / "htdemucs" / f"{stem_name}_raw_dump" / "vocals.wav"
        demucs_bgm_src = base_dir / "htdemucs" / f"{stem_name}_raw_dump" / "no_vocals.wav"

        if demucs_vocal_src.exists() and demucs_bgm_src.exists():
            if final_vocals_path.exists(): os.remove(final_vocals_path)
            if final_bgm_path.exists(): os.remove(final_bgm_path)
            
            os.rename(str(demucs_vocal_src), str(final_vocals_path))
            os.rename(str(demucs_bgm_src), str(final_bgm_path))
            print("✨ [STT 엔진] 오디오 트리플 분리 대성공!")
            print(f"       🎙️ 목소리 전용 소스 저장 완료 -> '{final_vocals_path.name}'")
            print(f"       🎵 배경음악 전용 소스 저장 완료 -> '{final_bgm_path.name}'")
        else:
            print("⚠️ [STT 엔진] AI 분리 트랙 배출 실패 -> 오리지널 덤프를 기반 파일로 우회 설정합니다.")
            if final_vocals_path.exists(): os.remove(final_vocals_path)
            os.rename(str(raw_wav_path), str(final_vocals_path))

        try:
            import shutil
            if (base_dir / "htdemucs").exists():
                shutil.rmtree(base_dir / "htdemucs")
        except Exception:
            pass

        # 3. Whisper 추론 시작
        model = self._load_model()
        print(f"🎙️ [STT ENGINE] '{final_vocals_path.name}' 소스 기반 고정밀 음성 인식 진행 중...")

        segments, info = model.transcribe(
            str(final_vocals_path),
            language="ko",
            beam_size=5,
            vad_filter=False,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt="정확한 한국어 방송 대사 받아쓰기 자막 녹취록.",
            suppress_blank=False,
            no_speech_threshold=0.5,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4
        )

        fps = mxf_meta.fps or 25.0
        start_tc = mxf_meta.start_timecode or "00:00:00:00"

        transcript_segments = []
        txt_lines = []

        for seg in segments:
            if not seg.text.strip():
                continue

            s_tc = self._seconds_to_timecode(seg.start, fps, start_tc)
            e_tc = self._seconds_to_timecode(seg.end, fps, start_tc)

            # 🎯 [수정] 시작 타임코드 ~ 종료 타임코드 표기
            txt_lines.append(f"[{s_tc} ~ {e_tc}] {seg.text.strip()}\n")

            transcript_segments.append(
                TranscriptSegment(
                    start_sec=seg.start,
                    end_sec=seg.end,
                    text=seg.text,
                    track_index=self.dialogue_track_index if self.dialogue_track_index is not None else 0,
                    confidence=seg.avg_logprob,
                    start_timecode=s_tc,
                    end_timecode=e_tc,
                    speaker="SPEAKER_00"
                )
            )

        if txt_lines:
            with open(final_txt_path, "w", encoding="utf-8") as f:
                f.writelines(txt_lines)
            print(f"📝 [STT 엔진] 전체 대사 메모장 파일 배출 완료 -> '{final_txt_path.name}'")

        print(f"✨ [성공] 총 {len(transcript_segments)}개의 타임라인 문장 인식 완주.\n")
        return AudioSTTResult(language=info.language, segments=transcript_segments)

    @staticmethod
    def _seconds_to_timecode(seconds: float, fps: float, start_timecode: str) -> str:
        fps = fps or 25.0
        h, m, s, f = (int(x) for x in start_timecode.replace(";", ":").split(":"))
        start_frames = int(((h * 3600 + m * 60 + s) * fps) + f)
        elapsed_frames = int(round(seconds * fps))
        total_frames = start_frames + elapsed_frames
        total_seconds, ff = divmod(total_frames, int(round(fps)))
        hh, rem = divmod(total_seconds, 3600)
        mm, ss = divmod(rem, 60)
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"