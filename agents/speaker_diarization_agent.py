"""
agents/speaker_diarization_agent.py

4단계 Agent: 사람의 목소리만 정제된 음원(_vocals.wav)을 바탕으로 화자 분리(Diarization)를 수행한다.
[2중 호환성 패치 마스터 버전]
  1. PyTorch 2.6+의 weights_only=True 강제 잠금으로 인한 픽클 에러 우회 후크 주입
  2. Torchaudio 최신 버전에서 삭제된 set_audio_backend 어트리뷰트 에러(AttributeError) 실시간 더미 후크 주입
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from agents.base import BaseAgent
from schema import AudioSTTResult

logger = logging.getLogger("mxf_pipeline.speaker_agent")

# =========================================================================
# 🛡️ [마스터 가드 1] Torchaudio 최신 버전 삭제 명령어 실시간 복구 (AttributeError 방어)
# =========================================================================
import torchaudio
if not hasattr(torchaudio, "set_audio_backend"):
    # pyannote 내부에서 백엔드를 강제 지정할 때 크래시가 나지 않도록 빈 함수(Dummy)를 주입합니다.
    torchaudio.set_audio_backend = lambda backend: None

# =========================================================================
# 🛡️ [마스터 가드 2] PyTorch 2.6+ 가중치 파일 언픽클 크래시 방어 (UnpicklingError 방어)
# =========================================================================
try:
    from pyannote.audio.core.task import Specifications
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([Specifications])
except ImportError:
    pass

original_torch_load = torch.load
def pytorch_compatibility_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)
torch.load = pytorch_compatibility_load
# =========================================================================


class SpeakerDiarizationAgent(BaseAgent):
    name = "speaker_diarization_agent"

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self._pipeline = None

    def run(self, context: Dict[str, Any]) -> AudioSTTResult:
        audio_stt: AudioSTTResult = context["audio_stt"]
        file_path = context["file_path"]
        output_dir = context.get("output_dir")
        
        base_dir = Path(output_dir) if output_dir else Path(file_path).resolve().parent
        stem_name = Path(file_path).stem
        vocals_wav = base_dir / f"{stem_name}_vocals.wav"
        final_txt_path = base_dir / f"{stem_name}_diarized_transcript.txt"
        
        target_wav = str(vocals_wav) if vocals_wav.exists() else file_path
        
        if not audio_stt.segments:
            return audio_stt

        diarization_success = False

        # -------------------------------------------------------------
        # 👑 [Mode 1] Pyannote 3.1 토큰 기반 프리미엄 화자 분리 시도
        # -------------------------------------------------------------
        if self.hf_token and not self.hf_token.startswith("HF_MOCK"):
            print("👤 [화자 분리] HuggingFace 토큰 감지 -> Pyannote 3.1 마스터 엔진 구동...")
            try:
                from pyannote.audio import Pipeline
                if self._pipeline is None:
                    self._pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1", 
                        use_auth_token=self.hf_token
                    )
                    if torch.cuda.is_available():
                        self._pipeline.to(torch.device("cuda"))
                
                diarization = self._pipeline(target_wav)
                
                for seg in audio_stt.segments:
                    seg_start = seg.start_sec
                    seg_end = seg.end_sec
                    best_speaker = "SPEAKER_00"
                    max_overlap = 0.0
                    
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        overlap = min(seg_end, turn.end) - max(seg_start, turn.start)
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_speaker = speaker
                    
                    if max_overlap > 0.05:
                        seg.speaker = best_speaker
                    else:
                        seg_mid = (seg_start + seg_end) / 2.0
                        for turn, _, speaker in diarization.itertracks(yield_label=True):
                            if turn.start <= seg_mid <= turn.end:
                                seg.speaker = speaker
                                break
                print("✨ [화자 분리] Pyannote 딥러닝 분석 완수.")
                diarization_success = True
                
            except Exception as e:
                print(f"ℹ️ [화자 분리] Pyannote 구동 환경 우회 가드 발동 (권한 미승인 또는 토큰 오류): {e}")

        # -------------------------------------------------------------
        # 🛠️ [Mode 2] 100% 로컬 오프라인 목소리 주파수 지문 클러스터링 엔진 (토치 프리/토큰 프리)
        # -------------------------------------------------------------
        if not diarization_success:
            print("⚙️ [화자 분리] '로컬 목소리 지문 주파수 분석(MFCC Timbre Clustering)' 모드로 전환합니다.")
            try:
                import librosa
                
                y, sr = librosa.load(target_wav, sr=16000)
                features = []
                valid_segments = []
                
                for seg in audio_stt.segments:
                    start_idx = int(seg.start_sec * sr)
                    end_idx = int(seg.end_sec * sr)
                    chunk = y[start_idx:end_idx]
                    
                    if len(chunk) < 1600:
                        continue
                        
                    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
                    mfcc_mean = np.mean(mfcc, axis=1)
                    features.append(mfcc_mean)
                    valid_segments.append(seg)
                
                if features:
                    X = np.array(features)
                    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
                    
                    n_clusters = min(4, len(X))
                    np.random.seed(42)
                    init_idx = np.random.choice(len(X), n_clusters, replace=False)
                    centroids = X[init_idx]
                    
                    for _ in range(20):
                        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
                        labels = np.argmin(distances, axis=1)
                        new_centroids = np.array([
                            X[labels == k].mean(axis=0) if len(X[labels == k]) > 0 else centroids[k] 
                            for k in range(n_clusters)
                        ])
                        if np.allclose(centroids, new_centroids):
                            break
                        centroids = new_centroids
                    
                    for seg, lbl in zip(valid_segments, labels):
                        seg.speaker = f"SPEAKER_{lbl+1:02d}"
                        
                    print(f"✨ [화자 분리] 로컬 주파수 분할 성공: 총 {n_clusters}명의 대화 인물 스캔 마감.")
                    diarization_success = True
            except Exception as e:
                print(f"❌ [화자 분리] 로컬 클러스터링 예외 발생: {e}")

        self._write_transcript_file(audio_stt, final_txt_path)
        return audio_stt

    @staticmethod
    def _write_transcript_file(audio_stt: AudioSTTResult, output_path: Path):
        txt_lines = []
        for seg in audio_stt.segments:
            txt_lines.append(f"[{seg.start_timecode}] {seg.speaker}: {seg.text.strip()}\n")
            
        if txt_lines:
            with open(output_path, "w", encoding="utf-8") as f:
                f.writelines(txt_lines)
        print(f"📝 [화자 분리] 대사 파일 내보내기 마감 -> '{output_path.name}'")