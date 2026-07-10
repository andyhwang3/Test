"""
agents/speaker_diarization_agent.py

4단계 Agent: 사람의 목소리만 정제된 음원을 바탕으로 
Pyannote AI 엔진을 가동하여 화자 분리(Diarization)를 수행하고 Whisper 대사에 레이블을 융합한다.
[기능 추가] 화자 구분이 완벽히 매핑된 최종 텍스트 파일(*_diarized_transcript.txt)을 outdir에 함께 저장한다.
"""

import os
import itertools
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from agents.base import BaseAgent
from schema import AudioSTTResult


class SpeakerDiarizationAgent(BaseAgent):
    name = "speaker_diarization_agent"

    def __init__(
        self,
        hf_token: Optional[str] = None,
        gpu_id: int = 0,
        min_speakers: Optional[int] = 2,
        max_speakers: Optional[int] = None,
    ):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.gpu_id = gpu_id
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._pipeline = None

    def run(self, context: Dict[str, Any]) -> AudioSTTResult:
        audio_stt: AudioSTTResult = context["audio_stt"]
        file_path = context["file_path"]
        output_dir = context.get("output_dir")
        
        base_dir = Path(output_dir) if output_dir else Path(file_path).resolve().parent
        stem_name = Path(file_path).stem
        vocals_wav = base_dir / f"{stem_name}_vocals.wav"
        
        # 📁 화자 분리 전용 텍스트 파일 경로 정의
        final_txt_path = base_dir / f"{stem_name}_diarized_transcript.txt"
        
        target_wav = str(vocals_wav) if vocals_wav.exists() else file_path
        
        print("👤 [화자 분리] Pyannote AI 가동 (목소리 고유 주파수 지문 채취 및 화자 타임라인 동기화)...")

        if not self.hf_token:
            raise RuntimeError("HF_TOKEN 환경변수가 없어 Pyannote 화자 분리를 실행할 수 없습니다.")

        self._patch_runtime_compatibility()
        pipeline = self._load_pipeline()
        diarization = self._run_diarization(pipeline, target_wav)
        turns = self._collect_turns(diarization)

        if not turns:
            raise RuntimeError("Pyannote가 화자 구간을 하나도 반환하지 않았습니다.")

        speaker_count = len({speaker for _, _, speaker in turns})
        if self.min_speakers is not None and speaker_count < self.min_speakers:
            print(f"⚠️ [화자 분리] 감지 화자 수가 {speaker_count}명이라 {self.min_speakers}명으로 강제 재분석합니다.")
            diarization = self._run_diarization(pipeline, target_wav, num_speakers=self.min_speakers)
            turns = self._collect_turns(diarization)
            speaker_count = len({speaker for _, _, speaker in turns})

        print(f"👥 [화자 분리] 감지된 화자 수: {speaker_count}명")

        self._assign_speakers_by_overlap(audio_stt, turns)
        print(f"✨ [화자 분리] 성공: Whisper 세그먼트에 실제 인물 화자 정보 바인딩을 완료했습니다.")
            
        # 📝 화자 구분이 반영된 최종 대사 텍스트 파일 기록
        txt_lines = []
        for seg in audio_stt.segments:
            txt_lines.append(f"[{seg.start_timecode} ~ {seg.end_timecode}] {seg.speaker}: {seg.text.strip()}\n")
            
        if txt_lines:
            with open(final_txt_path, "w", encoding="utf-8") as f:
                f.writelines(txt_lines)
            print(f"📝 [화자 분리] 화자 분리 대사 메모장 파일 배출 완료 -> '{final_txt_path.name}'")
            
        return audio_stt

    @staticmethod
    def _patch_runtime_compatibility() -> None:
        # 🎯 1. [NumPy 2.0+ 호환성 패치] 8가지 대소문자 조합(2^3=8) 완벽 방어
        import numpy as np
        for combo in itertools.product(*[(c.upper(), c.lower()) for c in "nan"]):
            attr = "".join(combo)
            if not hasattr(np, attr):
                setattr(np, attr, np.nan)

        # 🎯 2. [torchaudio 호환성 패치] set_audio_backend 제거 대응
        torchaudio = importlib.import_module("torchaudio")
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda x: None

    def _load_pipeline(self):
        torch = importlib.import_module("torch")
        pyannote_audio = importlib.import_module("pyannote.audio")
        Pipeline = pyannote_audio.Pipeline

        if self._pipeline is None:
            # 🎯 3. [pyannote 버전 호환성 패치] token vs use_auth_token 동적 폴백
            try:
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=self.hf_token
                )
            except TypeError:
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token
                )

            if self._pipeline is None:
                raise RuntimeError("Pyannote 파이프라인 로딩 실패: 모델 접근 권한 또는 HF_TOKEN을 확인하세요.")

            if torch.cuda.is_available():
                self._pipeline.to(torch.device(f"cuda:{self.gpu_id}"))

        return self._pipeline

    def _run_diarization(self, pipeline, target_wav: str, num_speakers: Optional[int] = None):
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        elif self.min_speakers is not None:
            kwargs["min_speakers"] = self.min_speakers
        if num_speakers is None and self.max_speakers is not None:
            kwargs["max_speakers"] = self.max_speakers

        try:
            return pipeline(target_wav, **kwargs)
        except TypeError:
            return pipeline(target_wav)

    @staticmethod
    def _collect_turns(diarization) -> List[Tuple[float, float, str]]:
        turns = [
            (float(turn.start), float(turn.end), str(speaker))
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        return sorted(turns, key=lambda item: (item[0], item[1]))

    @staticmethod
    def _assign_speakers_by_overlap(audio_stt: AudioSTTResult, turns: List[Tuple[float, float, str]]) -> None:
        for seg in audio_stt.segments:
            best_speaker = None
            best_overlap = 0.0

            for turn_start, turn_end, speaker in turns:
                overlap = max(0.0, min(seg.end_sec, turn_end) - max(seg.start_sec, turn_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker

            if best_speaker is None:
                seg_mid = (seg.start_sec + seg.end_sec) / 2.0
                best_speaker = min(
                    turns,
                    key=lambda turn: min(abs(seg_mid - turn[0]), abs(seg_mid - turn[1]))
                )[2]

            seg.speaker = best_speaker