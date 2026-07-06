"""
agents/audio_qc_agent.py

오디오 이상 탐지 Agent:
- EBU R128 Loudness (Integrated) / True Peak : ffmpeg ebur128 필터
- 클리핑(디지털 오버) 구간 : ffmpeg astats 필터의 peak level 기반
- 무음/드롭아웃 구간 : ffmpeg silencedetect 필터
- 클릭/팝(급격한 스펙트럴 변화) : librosa spectral flux 기반 (선택적 - librosa 설치 시에만 동작)
"""

import re
import subprocess
from typing import Any, Dict, List

from agents.base import BaseAgent
from schema import MXFBaseMeta, AudioQCResult, AudioAnomaly


class AudioQCAgent(BaseAgent):
    name = "audio_qc_agent"

    def __init__(
        self,
        ffmpeg_bin: str = "ffmpeg",
        target_lufs: float = -23.0,   # 방송 표준(EBU R128) 기준값. 매체별로 조정.
        clipping_threshold_db: float = -0.1,
        silence_threshold_db: str = "-50dB",
        silence_min_duration: float = 2.0,
    ):
        self.ffmpeg_bin = ffmpeg_bin
        self.target_lufs = target_lufs
        self.clipping_threshold_db = clipping_threshold_db
        self.silence_threshold_db = silence_threshold_db
        self.silence_min_duration = silence_min_duration

    def run(self, context: Dict[str, Any]) -> AudioQCResult:
        file_path = context["file_path"]
        mxf_meta: MXFBaseMeta = context["mxf_meta"]

        integrated_lufs, true_peak = self._measure_loudness(file_path)

        anomalies: List[AudioAnomaly] = []
        if integrated_lufs is not None and abs(integrated_lufs - self.target_lufs) > 2.0:
            anomalies.append(
                AudioAnomaly(
                    kind="loudness_violation",
                    start_time_sec=0.0,
                    end_time_sec=mxf_meta.duration_sec,
                    detail=f"integrated={integrated_lufs:.1f} LUFS (target={self.target_lufs} LUFS)",
                )
            )

        for track in mxf_meta.audio_tracks:
            anomalies.extend(self._detect_clipping(file_path, track.index))
            anomalies.extend(self._detect_silence(file_path, track.index))
            anomalies.extend(self._detect_click_pop(file_path, track.index))

        return AudioQCResult(
            integrated_loudness_lufs=integrated_lufs,
            true_peak_dbtp=true_peak,
            anomalies=anomalies,
        )

    # ---- EBU R128 Loudness / True Peak ----

    def _measure_loudness(self, file_path: str):
        cmd = [
            self.ffmpeg_bin, "-i", file_path,
            "-af", "ebur128=peak=true",
            "-f", "null", "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        stderr = proc.stderr

        integrated = self._extract_float(r"I:\s*(-?[\d.]+) LUFS", stderr)
        true_peak = self._extract_float(r"Peak:\s*(-?[\d.]+) dBFS", stderr)
        return integrated, true_peak

    @staticmethod
    def _extract_float(pattern: str, text: str):
        matches = re.findall(pattern, text)
        return float(matches[-1]) if matches else None

    # ---- 클리핑 ----

    def _detect_clipping(self, file_path: str, track_index: int) -> List[AudioAnomaly]:
        cmd = [
            self.ffmpeg_bin, "-i", file_path,
            "-map", f"0:a:{track_index}",
            "-af", "astats=metadata=1:reset=1",
            "-f", "null", "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        peaks = [float(v) for v in re.findall(r"Peak_level=(-?[\d.]+)", proc.stderr)]

        anomalies = []
        window_sec = 0.1  # astats reset=1 기준 대략적 윈도우 (실제 값은 asetnsamples와 함께 조정 필요)
        for i, peak in enumerate(peaks):
            if peak >= self.clipping_threshold_db:
                t = i * window_sec
                anomalies.append(
                    AudioAnomaly(
                        kind="clipping",
                        start_time_sec=t,
                        end_time_sec=t + window_sec,
                        track_index=track_index,
                        detail=f"peak={peak:.2f} dBFS",
                    )
                )
        return anomalies

    # ---- 무음 / 드롭아웃 ----

    def _detect_silence(self, file_path: str, track_index: int) -> List[AudioAnomaly]:
        cmd = [
            self.ffmpeg_bin, "-i", file_path,
            "-map", f"0:a:{track_index}",
            "-af", f"silencedetect=noise={self.silence_threshold_db}:d={self.silence_min_duration}",
            "-f", "null", "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        starts = [float(v) for v in re.findall(r"silence_start:\s*([\d.]+)", proc.stderr)]
        ends = [float(v) for v in re.findall(r"silence_end:\s*([\d.]+)", proc.stderr)]

        anomalies = []
        for start, end in zip(starts, ends):
            anomalies.append(
                AudioAnomaly(
                    kind="silence",
                    start_time_sec=start,
                    end_time_sec=end,
                    track_index=track_index,
                )
            )
        return anomalies

    # ---- 클릭/팝 (스펙트럴 급변) ----

    def _detect_click_pop(self, file_path: str, track_index: int) -> List[AudioAnomaly]:
        try:
            import numpy as np
            import librosa
        except ImportError:
            # librosa 미설치 시 이 검사만 건너뜀 (다른 QC 항목은 정상 수행)
            return []

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = os.path.join(tmp_dir, "track.wav")
            subprocess.run(
                [self.ffmpeg_bin, "-y", "-i", file_path, "-map", f"0:a:{track_index}", wav_path],
                capture_output=True, check=True,
            )
            y, sr = librosa.load(wav_path, sr=None, mono=True)

        flux = librosa.onset.onset_strength(y=y, sr=sr)
        threshold = float(np.mean(flux) + 4 * np.std(flux))
        onset_frames = np.where(flux > threshold)[0]
        times = librosa.frames_to_time(onset_frames, sr=sr)

        return [
            AudioAnomaly(
                kind="click_pop",
                start_time_sec=float(t),
                end_time_sec=float(t) + 0.05,
                track_index=track_index,
                detail="스펙트럴 flux 급변 감지",
            )
            for t in times
        ]
