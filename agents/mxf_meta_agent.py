"""
agents/mxf_meta_agent.py

1단계 Agent: MXF 파일의 Header Metadata를 읽어 표준 스키마로 정규화한다.

기본은 ffprobe(JSON) 로 코덱/해상도/fps/오디오 트랙 등을 뽑고,
시스템에 bmxlib의 `mxf2raw` CLI가 설치되어 있으면 OP 패턴, 타임코드 트랙처럼
ffprobe가 못 잡는 MXF 고유 정보까지 보강한다. (BBC bmxlib: https://github.com/bbc/bmx)
"""

import json
import shutil
import subprocess
from typing import Any, Dict, List

from agents.base import BaseAgent
from schema import MXFBaseMeta, AudioTrackInfo


class MXFMetaAgent(BaseAgent):
    name = "mxf_meta_agent"

    def __init__(self, ffprobe_bin: str = "ffprobe", mxf2raw_bin: str = "mxf2raw"):
        self.ffprobe_bin = ffprobe_bin
        self.mxf2raw_bin = mxf2raw_bin

    def run(self, context: Dict[str, Any]) -> MXFBaseMeta:
        file_path = context["file_path"]
        probe = self._ffprobe(file_path)
        meta = self._parse_ffprobe(file_path, probe)

        if shutil.which(self.mxf2raw_bin):
            self._enrich_with_bmxlib(file_path, meta)

        return meta

    # ---- ffprobe ----

    def _ffprobe(self, file_path: str) -> Dict[str, Any]:
        cmd = [
            self.ffprobe_bin,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            file_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8", errors="replace")
        return json.loads(proc.stdout)

    def _parse_ffprobe(self, file_path: str, probe: Dict[str, Any]) -> MXFBaseMeta:
        fmt = probe.get("format", {})
        streams = probe.get("streams", [])

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

        fps = self._parse_fps(video_stream.get("r_frame_rate", "0/1"))
        start_tc = self._extract_timecode(video_stream, fmt)

        audio_tracks: List[AudioTrackInfo] = []
        for idx, a in enumerate(audio_streams):
            audio_tracks.append(
                AudioTrackInfo(
                    index=idx,
                    channels=int(a.get("channels", 0)),
                    sample_rate=int(a.get("sample_rate", 0)),
                    bit_depth=self._bit_depth_from_sample_fmt(a.get("sample_fmt")),
                )
            )

        return MXFBaseMeta(
            file_path=file_path,
            codec=video_stream.get("codec_name", "unknown"),
            resolution=f'{video_stream.get("width", 0)}x{video_stream.get("height", 0)}',
            fps=fps,
            duration_sec=float(fmt.get("duration", 0.0) or 0.0),
            start_timecode=start_tc,
            audio_tracks=audio_tracks,
            raw={"format": fmt, "video_stream": video_stream},
        )

    @staticmethod
    def _parse_fps(rate_str: str) -> float:
        try:
            num, den = rate_str.split("/")
            den = float(den) or 1.0
            return round(float(num) / den, 3)
        except (ValueError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def _bit_depth_from_sample_fmt(sample_fmt: str) -> int:
        mapping = {
            "s16": 16, "s16p": 16,
            "s24": 24, "s24p": 24,
            "s32": 32, "s32p": 32,
            "flt": 32, "fltp": 32,
        }
        return mapping.get(sample_fmt, 0)

    @staticmethod
    def _extract_timecode(video_stream: Dict[str, Any], fmt: Dict[str, Any]) -> str:
        tags = {**video_stream.get("tags", {}), **fmt.get("tags", {})}
        return tags.get("timecode", "00:00:00:00")

    # ---- bmxlib (선택적 보강) ----

    def _enrich_with_bmxlib(self, file_path: str, meta: MXFBaseMeta) -> None:
        """
        mxf2raw --info-format json <file> 로 OP 패턴 등 MXF 고유 필드를 보강.
        bmxlib 미설치 환경에서는 조용히 건너뛴다 (ffprobe 결과만으로도 파이프라인은 동작).
        """
        try:
            cmd = [self.mxf2raw_bin, "--info-format", "json", file_path]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8", errors="replace")
            info = json.loads(proc.stdout)
            meta.op_pattern = info.get("essence", {}).get("op_label")
            meta.raw["bmxlib"] = info
        except Exception:
            # bmxlib 파싱 실패는 치명적이지 않음 - ffprobe 메타만으로 다음 단계 진행
            pass
