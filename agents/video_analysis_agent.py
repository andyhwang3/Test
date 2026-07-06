"""
agents/video_analysis_agent.py

2단계 Agent: 장면전환(Scene Cut) 위치와 기본 화질 이상(블랙 프레임, 프리즈 프레임)을 탐지한다.

- Scene cut: PySceneDetect(ContentDetector) 사용. GPU가 있다면 TransNetV2 등으로 교체 가능.
- 화질 이상: ffmpeg의 blackdetect / freezedetect 필터를 사용 (추가 의존성 없이 ffmpeg만으로 동작).
- 프레임 번호 -> MXF 타임코드 변환은 mxf_meta_agent가 넘겨준 start_timecode + fps를 기준으로 계산한다.
- extract_frames=True(기본값)일 경우 각 장면전환 지점의 프레임을 PNG로 추출해
  <scene_output_dir>/scene_0001_00-00-12-05.png 형태로 저장하고 SceneCut.image_path에 기록한다.
  scene_output_dir은 orchestrator가 입력 파일명 기준으로 자동 생성해 context에 넣어준다.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from schema import MXFBaseMeta, SceneCut, VideoAnalysisResult, VideoAnomaly


class VideoAnalysisAgent(BaseAgent):
    name = "video_analysis_agent"

    def __init__(
        self,
        ffmpeg_bin: str = "ffmpeg",
        scene_threshold: float = 27.0,
        extract_frames: bool = True,
    ):
        self.ffmpeg_bin = ffmpeg_bin
        self.scene_threshold = scene_threshold
        self.extract_frames = extract_frames

    def run(self, context: Dict[str, Any]) -> VideoAnalysisResult:
        file_path = context["file_path"]
        mxf_meta: MXFBaseMeta = context["mxf_meta"]
        scene_output_dir: Optional[str] = context.get("scene_output_dir")

        scene_cuts = self._detect_scene_cuts(file_path, mxf_meta)

        if self.extract_frames and scene_output_dir:
            self._extract_scene_frames(file_path, scene_cuts, scene_output_dir)

        anomalies = self._detect_anomalies(file_path)

        return VideoAnalysisResult(scene_cuts=scene_cuts, anomalies=anomalies)

    # ---- Scene cut (PySceneDetect) ----

    def _detect_scene_cuts(self, file_path: str, mxf_meta: MXFBaseMeta) -> List[SceneCut]:
        # scenedetect는 선택적 의존성 - 설치 안 된 환경에서도 파이프라인이 죽지 않도록 지연 임포트
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import ContentDetector
        except ImportError as exc:
            raise RuntimeError(
                "PySceneDetect가 설치되어 있지 않습니다. `pip install scenedetect[opencv]`"
            ) from exc

        video = open_video(file_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.scene_threshold))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        fps = mxf_meta.fps or video.frame_rate
        cuts: List[SceneCut] = []
        for start_time, _end_time in scene_list:
            frame_num = start_time.get_frames()
            time_sec = start_time.get_seconds()
            cuts.append(
                SceneCut(
                    frame_number=frame_num,
                    timecode=self._frame_to_timecode(frame_num, fps, mxf_meta.start_timecode),
                    time_sec=time_sec,
                )
            )
        return cuts

    @staticmethod
    def _frame_to_timecode(frame_num: int, fps: float, start_timecode: str) -> str:
        """프레임 번호를 MXF의 start_timecode 기준 절대 타임코드로 변환 (drop-frame 미고려 단순 변환)."""
        fps = fps or 25.0
        h, m, s, f = (int(x) for x in start_timecode.replace(";", ":").split(":"))
        start_frames = int(((h * 3600 + m * 60 + s) * fps) + f)
        total_frames = start_frames + frame_num
        total_seconds, ff = divmod(total_frames, int(round(fps)))
        hh, rem = divmod(total_seconds, 3600)
        mm, ss = divmod(rem, 60)
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"

    # ---- 장면전환 프레임 PNG 추출 ----

    def _extract_scene_frames(self, file_path: str, scene_cuts: List[SceneCut], out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        for i, cut in enumerate(scene_cuts, start=1):
            safe_tc = cut.timecode.replace(":", "-").replace(";", "-")
            filename = f"scene_{i:04d}_{safe_tc}.png"
            out_path = os.path.join(out_dir, filename)

            cmd = [
                self.ffmpeg_bin, "-y",
                "-ss", f"{cut.time_sec:.3f}",
                "-i", file_path,
                "-frames:v", "1",
                "-q:v", "2",
                out_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if proc.returncode == 0 and os.path.exists(out_path):
                cut.image_path = out_path
            # 실패해도 해당 컷의 image_path만 비워두고 나머지 컷/단계는 계속 진행

    # ---- 화질 이상 (ffmpeg blackdetect / freezedetect) ----

    def _detect_anomalies(self, file_path: str) -> List[VideoAnomaly]:
        anomalies: List[VideoAnomaly] = []
        anomalies.extend(self._run_filter(file_path, "blackdetect=d=0.5:pic_th=0.98", "black_frame",
                                           r"black_start:(?P<start>[\d.]+) black_end:(?P<end>[\d.]+)"))
        anomalies.extend(self._run_filter(file_path, "freezedetect=n=-60dB:d=1", "freeze_frame",
                                           r"freeze_start: (?P<start>[\d.]+).*?freeze_end: (?P<end>[\d.]+)"))
        return anomalies

    def _run_filter(self, file_path: str, vf: str, kind: str, pattern: str) -> List[VideoAnomaly]:
        cmd = [self.ffmpeg_bin, "-i", file_path, "-vf", vf, "-an", "-f", "null", "-"]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        matches = re.finditer(pattern, proc.stderr, re.DOTALL)
        results = []
        for m in matches:
            results.append(
                VideoAnomaly(
                    kind=kind,
                    start_time_sec=float(m.group("start")),
                    end_time_sec=float(m.group("end")),
                )
            )
        return results
