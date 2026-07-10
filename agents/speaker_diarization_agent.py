"""
agents/speaker_diarization_agent.py

4단계 Agent: 사람의 목소리만 정제된 음원을 바탕으로 
Pyannote AI 엔진을 가동하여 화자 분리(Diarization)를 수행하고 Whisper 대사에 레이블을 융합한다.
[기능 추가] 화자 구분이 완벽히 매핑된 최종 텍스트 파일(*_diarized_transcript.txt)을 outdir에 함께 저장한다.
"""

import os
import torch
import itertools
from pathlib import Path
from typing import Dict, Any, Optional
from agents.base import BaseAgent
from schema import AudioSTTResult


class SpeakerDiarizationAgent(BaseAgent):
    name = "speaker_diarization_agent"

    def __init__(self, hf_token: Optional[str] = None, gpu_id: int = 0):
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "HF_MOCK_TOKEN")
        self.gpu_id = gpu_id
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
        
        try:
            # 🎯 1. [NumPy 2.0+ 호환성 패치] 8가지 대소문자 조합(2^3=8) 완벽 방어
            import numpy as np
            for combo in itertools.product(*[(c.upper(), c.lower()) for c in "nan"]):
                attr = "".join(combo)
                if not hasattr(np, attr):
                    setattr(np, attr, np.nan)

            # 🎯 2. [torchaudio 호환성 패치] set_audio_backend 제거 대응
            import torchaudio
            if not hasattr(torchaudio, "set_audio_backend"):
                torchaudio.set_audio_backend = lambda x: None

            from pyannote.audio import Pipeline
            
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

                if torch.cuda.is_available():
                    self._pipeline.to(torch.device(f"cuda:{self.gpu_id}"))
            
            diarization = self._pipeline(target_wav)
            
            for seg in audio_stt.segments:
                seg_mid = (seg.start_sec + seg.end_sec) / 2.0
                assigned_speaker = "SPEAKER_00"
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.start <= seg_mid <= turn.end:
                        assigned_speaker = speaker
                        break
                seg.speaker = assigned_speaker
                
            print(f"✨ [화자 분리] 성공: Whisper 세그먼트에 실제 인물 화자 정보 바인딩을 완료했습니다.")
            
        except Exception as e:
            print(f"⚠️ [화자 분리] Pyannote 모듈 부팅 스킵: {e}")
            print("ℹ️ 파이프라인 안전 가드 발동 -> 기본 화자명(SPEAKER_00) 레이아웃을 보존합니다.")
            
        # 📝 화자 구분이 반영된 최종 대사 텍스트 파일 기록
        txt_lines = []
        for seg in audio_stt.segments:
            txt_lines.append(f"[{seg.start_timecode} ~ {seg.end_timecode}] {seg.speaker}: {seg.text.strip()}\n")
            
        if txt_lines:
            with open(final_txt_path, "w", encoding="utf-8") as f:
                f.writelines(txt_lines)
            print(f"📝 [화자 분리] 화자 분리 대사 메모장 파일 배출 완료 -> '{final_txt_path.name}'")
            
        return audio_stt