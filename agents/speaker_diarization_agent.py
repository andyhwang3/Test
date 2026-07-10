"""
agents/speaker_diarization_agent.py

4단계 Agent: 사람의 목소리만 정제된 음원(_vocals.wav)을 바탕으로 화자 분리(Diarization)를 수행한다.
[3중 호환성 패치 + GPU ID 매핑 + 타임코드 확장 마스터 버전 (시스템 내부 구조 커널 방어)]
  1. PyTorch 2.6+의 weights_only=True 강제 잠금으로 인한 픽클 에러 우회 후크 주입
  2. Torchaudio 최신 버전에서 완전 삭제된 backend 모듈 및 서브모듈 실시간 가상 후크 주입 방어 (inspect 교란 방어 추가)
  3. NumPy 2.0+에서 삭제된 np.NaN 어트리뷰트 에러(AttributeError) 실시간 복구 후크 주입
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

from agents.base import BaseAgent
from schema import AudioSTTResult

logger = logging.getLogger("mxf_pipeline.speaker_agent")

# =========================================================================
# 🛡️ [마스터 가드 1] Torchaudio 최신 버전 삭제 명령어 및 모듈 실시간 가상화 (시스템 내부 커널 교란 방어)
# =========================================================================
import torchaudio
from types import ModuleType

if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None

if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"

# 💡 [정밀 보완] 파이썬 inspect 기능이 내부 파일 경로(__file__) 등을 조회할 때의 오작동을 원천 차단합니다.
class DynamicDummyModule(ModuleType):
    def __getattr__(self, name):
        # 파이썬 특수 시스템 속성(__으로 시작하고 끝나는 속성) 요청 시 가짜 객체 대신 표준 디폴트 값 반환
        if name.startswith("__") and name.endswith("__"):
            if name in ("__file__", "__cached__"):
                return ""
            if name in ("__path__", "__all__"):
                return []
            return None
            
        class UniversalStub:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, *args, **kwargs): return self
            def __getattr__(self, sub_name):
                if sub_name.startswith("__") and sub_name.endswith("__"):
                    if sub_name in ("__file__", "__cached__"):
                        return ""
                    if sub_name in ("__path__", "__all__"):
                        return []
                    return None
                return self
        return UniversalStub()

# 시스템 모듈 커널에 강제 후킹 등록
if "torchaudio.backend" not in sys.modules:
    backend_dummy = DynamicDummyModule("torchaudio.backend")
    sys.modules["torchaudio.backend"] = backend_dummy
    torchaudio.backend = backend_dummy

if "torchaudio.backend.common" not in sys.modules:
    common_dummy = DynamicDummyModule("torchaudio.backend.common")
    sys.modules["torchaudio.backend.common"] = common_dummy
    setattr(torchaudio.backend, "common", common_dummy)
# =========================================================================

# =========================================================================
# 🛡️ [마스터 가드 3] NumPy 2.0+ 버전 하위 호환성 복구 (np.NaN AttributeError 방어)
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
                        device_str = f"cuda:{self.gpu_id}" if self.gpu_id is not None else "cuda"
                        print(f"🚀 [화자 분리] Pyannote 가속 디바이스 지정: {device_str}")
                        self._pipeline.to(torch.device(device_str))
                
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
                print(f"❌ [화자 분리] Pyannote 최종 엔진 구동 실패: {e}")

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
                    chunk = y