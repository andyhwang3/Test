"""
agents/speaker_diarization_agent.py

4단계 Agent: 사람의 목소리만 정제된 음원(_vocals.wav)을 바탕으로 화자 분리(Diarization)를 수행한다.
[최신 생태계 반영 버전 (pyannote.audio 4.x + torchcodec 네이티브 적용)]
  1. PyTorch 2.6+의 weights_only=True 강제 잠금으로 인한 픽클 에러 우회 후크 주입
  2. NumPy 2.0+에서 삭제된 np.NaN 어트리뷰트 에러(AttributeError) 실시간 복구 후크 주입
  3. 불필요한 Torchaudio 우회 패치 제거 (최신 torchcodec 라이브러리 네이티브 위임)
  4. main.py 할당 gpu_id 멀티 GPU 타겟팅 연동 패치
  5. 텍스트 출력 시 [시작시간 ~ 종료시간] 듀얼 타임코드 확장 반영
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# 💡 [최신 코덱 점검] torchcodec 모듈이 정상적으로 설치되었는지 확인
try:
    import torchcodec
except ImportError:
    print("❌ [치명적 오류] torchcodec 패키지가 누락되었습니다. 터미널에서 'pip install torchcodec'을 실행하세요.")

from agents.base import BaseAgent
from schema import AudioSTTResult

logger = logging.getLogger("mxf_pipeline.speaker_agent")

# =========================================================================
# 🛡️ [마스터 가드 1] NumPy 2.0+ 버전 하위 호환성 복구 (np.NaN AttributeError 방어)
# =========================================================================
if not hasattr(np, "NaN"):
    np.NaN = np.nan
# =========================================================================

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

    def __init__(self, hf_token: Optional[str] = None, gpu_id: Optional[Any] = None, **kwargs):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.gpu_id = gpu_id
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
        # 👑 [Mode 1] Pyannote 3.1 (Engine 4.x 기반) 토큰 기반 프리미엄 화자 분리 시도
        # -------------------------------------------------------------
        if self.hf_token and not self.hf_token.startswith("HF_MOCK"):
            print("👤 [화자 분리] HuggingFace 토큰 감지 -> Pyannote 마스터 엔진 구동...")
            try:
                from pyannote.audio import Pipeline
                if self._pipeline is None:
                    # 최신 API 권장사항에 맞춰 use_auth_token을 토큰 변수로 적용
                    try:
                        # 최신 버전 파라미터 (token)
                        self._pipeline = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1", 
                            token=self.hf_token
                        )
                    except TypeError:
                        # 구버전 파라미터 (use_auth_token) 백업
                        self._pipeline = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1", 
                            use_auth_token=self.hf_token
                        )
                    
                    if torch.cuda.is_available():
                        device_str = f"cuda:{self.gpu_id}" if self.gpu_id is not None else "cuda"
                        print(f"🚀 [화자 분리] Pyannote 가속 디바이스 지정: {device_str}")
                        self._pipeline.to(torch.device(device_str))
                
                # 최신 버전에서는 내부적으로 torchcodec과 FFmpeg를 직접 사용하여 디코딩합니다.
                diarization = self._pipeline(target_wav)
                
                # pyannote 최신 버전(DiarizeOutput) 및 구버전(Annotation) 자동 호환 추출
                annotation = getattr(diarization, "speaker_diarization", diarization)
                
                for seg in audio_stt.segments:
                    seg_start = seg.start_sec
                    seg_end = seg.end_sec
                    best_speaker = "SPEAKER_00"
                    max_overlap = 0.0
                    
                    for turn, _, speaker in annotation.itertracks(yield_label=True):
                        overlap = min(seg_end, turn.end) - max(seg_start, turn.start)
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_speaker = speaker
                    
                    if max_overlap > 0.05:
                        seg.speaker = best_speaker
                    else:
                        seg_mid = (seg_start + seg_end) / 2.0
                        for turn, _, speaker in annotation.itertracks(yield_label=True):
                            if turn.start <= seg_mid <= turn.end:
                                seg.speaker = speaker
                                break
                print("✨ [화자 분리] Pyannote 딥러닝 분석 완수.")
                diarization_success = True
                
            except Exception as e:
                print(f"❌ [화자 분리] Pyannote 최종 엔진 구동 실패: {e}")

        # -------------------------------------------------------------
        # 🛠️ [Mode 2] 100% 로컬 오프라인 목소리 주파수 지문 클러스터링 엔진 (대체 모드)
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
            start_time = seg.start_timecode
            
            end_time = getattr(seg, 'end_timecode', None)
            if not end_time and hasattr(seg, 'end_sec'):
                tot_sec = int(seg.end_sec)
                h = tot_sec // 3600
                m = (tot_sec % 3600) // 60
                s = tot_sec % 60
                ms = int((seg.end_sec - tot_sec) * 100)
                end_time = f"{h:02d}:{m:02d}:{s:02d}.{ms:02d}"
            elif not end_time:
                end_time = "??:??:??"

            txt_lines.append(f"[{start_time} ~ {end_time}] {seg.speaker}: {seg.text.strip()}\n")
            
        if txt_lines:
            with open(output_path, "w", encoding="utf-8") as f:
                f.writelines(txt_lines)
        print(f"📝 [화자 분리] 대사 파일 내보내기 마감 -> '{output_path.name}'")