"""
agents/audio_stt_agent.py

3단계 Agent: MXF 내 오디오 트랙을 텍스트로 변환한다.

- MXF는 보통 stereo pair가 여러 개(대사/음악/효과) 들어있으므로, dialogue_track_index를
  명시적으로 지정하거나, 지정이 없으면 RMS 에너지 변동성(발화 패턴)이 가장 큰 트랙을 대사로 추정한다.
- STT 엔진은 faster-whisper(large-v3)를 기본으로 사용. GPU(RTX A4000 등)에서 device="cuda" 권장.

Windows 참고: `pip install nvidia-cublas-cu12` 등으로 설치되는 DLL은 site-packages 밑
(nvidia\\cublas\\bin, nvidia\\cudnn\\bin)에 위치하는데, Windows는 이 경로를 자동으로
PATH/DLL 검색 경로에 넣어주지 않는다. 그래서 import 시점에 os.add_dll_directory로
직접 등록해준다 (Linux/macOS는 필요 없음).
"""

import subprocess
import sys
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from schema import MXFBaseMeta, AudioSTTResult, TranscriptSegment


def _register_windows_cuda_dll_dirs() -> None:
    """pip으로 설치된 nvidia-cublas-cu12 / nvidia-cudnn-cu12의 DLL 폴더를
    Windows DLL 검색 경로에 추가한다. 실패해도 조용히 넘어간다 (CPU 모드로 폴백 가능하도록)."""
    if sys.platform != "win32":
        return
    try:
        import nvidia.cublas as _cublas
        import nvidia.cudnn as _cudnn
    except ImportError:
        return

    for pkg in (_cublas, _cudnn):
        bin_dir = Path(pkg.__file__).parent / "bin"
        if bin_dir.is_dir():
            try:
                os.add_dll_directory(str(bin_dir))
            except (OSError, AttributeError):
                pass


_register_windows_cuda_dll_dirs()


class AudioSTTAgent(BaseAgent):
    name = "audio_stt_agent"

    def __init__(
        self,
        ffmpeg_bin: str = "ffmpeg",
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        dialogue_track_index: Optional[int] = None,
    ):
        self.ffmpeg_bin = ffmpeg_bin
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.dialogue_track_index = dialogue_track_index
        self._model = None  # 지연 로딩 - 여러 파일 처리 시 재사용

    def run(self, context: Dict[str, Any]) -> AudioSTTResult:
        file_path = context["file_path"]
        mxf_meta: MXFBaseMeta = context["mxf_meta"]

        track_index = self.dialogue_track_index
        if track_index is None:
            track_index = self._guess_dialogue_track(file_path, mxf_meta)

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = os.path.join(tmp_dir, "dialogue.wav")
            self._extract_audio_track(file_path, track_index, wav_path)
            segments, language = self._transcribe(wav_path, track_index)

        return AudioSTTResult(language=language, segments=segments)

    # ---- 대사 트랙 추정 ----

    def _guess_dialogue_track(self, file_path: str, mxf_meta: MXFBaseMeta) -> int:
        """
        각 오디오 트랙의 RMS 표준편차(발화의 강약 변화)를 비교해 가장 변동성이 큰 트랙을 대사로 추정.
        음악/앰비언스는 상대적으로 레벨이 평탄한 경우가 많다는 경험칙 기반의 단순 휴리스틱.
        정확도가 중요하면 dialogue_track_index를 직접 지정할 것.
        """
        if len(mxf_meta.audio_tracks) <= 1:
            return 0

        best_idx, best_score = 0, -1.0
        for track in mxf_meta.audio_tracks:
            score = self._track_rms_variance(file_path, track.index)
            if score > best_score:
                best_idx, best_score = track.index, score
        return best_idx

    def _track_rms_variance(self, file_path: str, track_index: int) -> float:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy가 필요합니다: pip install numpy") from exc

        cmd = [
            self.ffmpeg_bin, "-i", file_path,
            "-map", f"0:a:{track_index}",
            "-af", "asetnsamples=n=4096,astats=metadata=1:reset=1",
            "-f", "null", "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        import re
        rms_values = [float(v) for v in re.findall(r"RMS_level=(-?[\d.]+)", proc.stderr)]
        return float(np.std(rms_values)) if rms_values else 0.0

    # ---- 오디오 추출 + 전사 ----

    def _extract_audio_track(self, file_path: str, track_index: int, out_wav: str) -> None:
        cmd = [
            self.ffmpeg_bin, "-y", "-i", file_path,
            "-map", f"0:a:{track_index}",
            "-ac", "1", "-ar", "16000",
            out_wav,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    def _load_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper가 설치되어 있지 않습니다. `pip install faster-whisper`"
                ) from exc
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def _transcribe(self, wav_path: str, track_index: int):
        model = self._load_model()
        raw_segments, info = model.transcribe(wav_path, word_timestamps=False)

        segments: List[TranscriptSegment] = [
            TranscriptSegment(
                start_sec=seg.start,
                end_sec=seg.end,
                text=seg.text.strip(),
                track_index=track_index,
                confidence=getattr(seg, "avg_logprob", None),
            )
            for seg in raw_segments
        ]
        return segments, info.language
