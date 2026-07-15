"""
agents/audio_stt_agent.py

3단계 Agent: 오디오 스트림을 무손실 단일 채널로 추출한 뒤 고정밀 STT를 수행한다.
(Hugging Face 전용: batiai/batisay-ko-base 호환 및 모델 ID 매핑 적용)
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

    # 기존 단축 모델명을 Hugging Face 전체 경로로 매핑하는 사전
    MODEL_ALIAS_MAP = {
        "large-v3": "openai/whisper-large-v3",
        "large-v2": "openai/whisper-large-v2",
        "large": "openai/whisper-large",
        "medium": "openai/whisper-medium",
        "small": "openai/whisper-small",
        "base": "openai/whisper-base",
        "tiny": "openai/whisper-tiny",
    }

    def __init__(
        self,
        model_size: str = "batiai/batisay-ko-base",  # 💡 기본값 설정
        device: str = "cuda",
        gpu_id: int = 0,
        compute_type: str = "float16",
        ffmpeg_bin: str = "ffmpeg",
        dialogue_track_index: Optional[int] = None,
        use_raw_audio_for_stt: bool = True
    ):
        # 단축명(large-v3 등)이 들어오면 Hugging Face 정식 ID로 변환
        self.model_size = self.MODEL_ALIAS_MAP.get(model_size, model_size)
        self.device = device
        self.gpu_id = gpu_id
        self.device_str = f"{device}:{gpu_id}" if device == "cuda" else device
        self.compute_type = compute_type
        self.ffmpeg_bin = ffmpeg_bin
        self.dialogue_track_index = dialogue_track_index
        self.use_raw_audio_for_stt = use_raw_audio_for_stt
        self._model = None

    def _load_model(self):
        if self._model is None:
            import torch
            from transformers import pipeline

            print(f"📦 [STT 엔진] Hugging Face 파이프라인 로딩 중: {self.model_size}")
            
            dtype = torch.float16 if (self.compute_type == "float16" and self.device == "cuda") else torch.float32
            device_idx = self.gpu_id if self.device == "cuda" else -1

            self._model = pipeline(
                "automatic-speech-recognition",
                model=self.model_size,
                torch_dtype=dtype,
                device=device_idx,
            )
        return self._model

    def run(self, context: Dict[str, Any]) -> AudioSTTResult:
        file_path = context["file_path"]
        mxf_meta: MXFBaseMeta = context["mxf_meta"]
        output_dir = context.get("output_dir")

        base_dir = Path(output_dir) if output_dir else Path(file_path).resolve().parent
        stem_name = Path(file_path).stem
        
        raw_wav_path = base_dir / f"{stem_name}_raw_dump.wav"
        final_vocals_path = base_dir / f"{stem_name}_vocals.wav"
        final_bgm_path = base_dir / f"{stem_name}_bgm.wav"
        final_txt_path = base_dir / f"{stem_name}_transcript.txt"

        # -------------------------------------------------------------
        # 1. MXF 오디오 스트림 무손실 추출 (16kHz Mono)
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # 2. Demucs AI 오디오 소스 분리
        # -------------------------------------------------------------
        print("🧠 [STT 엔진] Demucs AI 오디오 소스 분리 가동 중...")
        demucs_model_name = "htdemucs_ft"
        
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
                "-n", demucs_model_name,
                "--two-stems", "vocals",
                "--shifts", "1",
                "-d", self.device_str,
                "-o", str(base_dir),
                str(raw_wav_path)
            ]
            demucs_main()

            torchaudio.save = original_save
            sys.argv = original_argv

            d_vocal = base_dir / demucs_model_name / f"{stem_name}_raw_dump" / "vocals.wav"
            d_bgm = base_dir / demucs_model_name / f"{stem_name}_raw_dump" / "no_vocals.wav"

            if d_vocal.exists() and d_bgm.exists():
                if final_vocals_path.exists(): os.remove(final_vocals_path)
                if final_bgm_path.exists(): os.remove(final_bgm_path)

                subprocess.run([
                    self.ffmpeg_bin, "-y", "-i", str(d_vocal),
                    "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(final_vocals_path)
                ], capture_output=True)

                subprocess.run([
                    self.ffmpeg_bin, "-y", "-i", str(d_bgm),
                    "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(final_bgm_path)
                ], capture_output=True)

        except Exception as e:
            print(f"⚠️ [STT 엔진] Demucs 가동 중 내부 오류 발생: {e}")
            if not final_vocals_path.exists():
                os.rename(str(raw_wav_path), str(final_vocals_path))

        try:
            import shutil
            if (base_dir / demucs_model_name).exists():
                shutil.rmtree(base_dir / demucs_model_name)
        except Exception:
            pass

        # -------------------------------------------------------------
        # 3. Whisper / Batisay 음성 인식 가동
        # -------------------------------------------------------------
        target_audio = str(raw_wav_path if raw_wav_path.exists() else final_vocals_path)

        pipe = self._load_model()
        print(f"🎙️ [STT ENGINE] Hugging Face ({self.model_size}) 음성 인식 진행 중...")

        generate_kwargs = {
            "language": "korean",
            "task": "transcribe",
            "num_beams": 5,
        }

        result = pipe(
            target_audio,
            return_timestamps=True,
            chunk_length_s=30,
            stride_length_s=5,
            generate_kwargs=generate_kwargs
        )

        chunks = result.get("chunks", [])

        fps = mxf_meta.fps or 25.0
        start_tc = mxf_meta.start_timecode or "00:00:00:00"

        transcript_segments = []
        txt_lines = []

        for chunk in chunks:
            text = chunk["text"].strip()
            if not text:
                continue

            start_sec, end_sec = chunk["timestamp"]
            start_sec = start_sec if start_sec is not None else 0.0
            end_sec = end_sec if end_sec is not None else start_sec + 1.0

            s_tc = self._seconds_to_timecode(start_sec, fps, start_tc)
            e_tc = self._seconds_to_timecode(end_sec, fps, start_tc)

            txt_lines.append(f"[{s_tc} ~ {e_tc}] {text}\n")

            transcript_segments.append(
                TranscriptSegment(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    text=text,
                    track_index=self.dialogue_track_index if self.dialogue_track_index is not None else 0,
                    confidence=1.0,
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
        return AudioSTTResult(language="ko", segments=transcript_segments)

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