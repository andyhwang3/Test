"""
schema.py
전체 파이프라인에서 주고받는 표준 데이터 구조 정의.
각 Agent는 이 스키마에 맞춰 결과를 채워 넣고, Orchestrator가 최종적으로 병합한다.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class AudioTrackInfo:
    index: int
    channels: int
    sample_rate: int
    bit_depth: Optional[int] = None
    role: Optional[str] = None  # dialogue / music / effects / unknown


@dataclass
class MXFBaseMeta:
    """1단계: MXF 기본 메타 (Header Metadata / ffprobe 기반)"""
    file_path: str = ""
    codec: str = ""
    resolution: str = ""
    fps: float = 0.0
    duration_sec: float = 0.0
    start_timecode: str = "00:00:00:00"
    audio_tracks: List[AudioTrackInfo] = field(default_factory=list)
    op_pattern: Optional[str] = None  # OP1a 등
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneCut:
    frame_number: int
    timecode: str
    time_sec: float
    confidence: Optional[float] = None
    image_path: Optional[str] = None


@dataclass
class VideoAnomaly:
    kind: str  # black_frame / freeze_frame / blocking 등
    start_time_sec: float
    end_time_sec: float
    detail: Optional[str] = None


@dataclass
class VideoAnalysisResult:
    scene_cuts: List[SceneCut] = field(default_factory=list)
    anomalies: List[VideoAnomaly] = field(default_factory=list)


@dataclass
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str
    track_index: int
    confidence: Optional[float] = None


@dataclass
class AudioSTTResult:
    language: Optional[str] = None
    segments: List[TranscriptSegment] = field(default_factory=list)


@dataclass
class AudioAnomaly:
    kind: str  # clipping / silence / loudness_violation / click_pop / sync_drift
    start_time_sec: float
    end_time_sec: float
    track_index: Optional[int] = None
    detail: Optional[str] = None


@dataclass
class AudioQCResult:
    integrated_loudness_lufs: Optional[float] = None
    true_peak_dbtp: Optional[float] = None
    anomalies: List[AudioAnomaly] = field(default_factory=list)


@dataclass
class PipelineReport:
    """최종 병합 결과 - 이 객체가 JSON으로 직렬화되어 저장/전달된다."""
    mxf_meta: MXFBaseMeta = field(default_factory=MXFBaseMeta)
    video_analysis: VideoAnalysisResult = field(default_factory=VideoAnalysisResult)
    audio_stt: AudioSTTResult = field(default_factory=AudioSTTResult)
    audio_qc: AudioQCResult = field(default_factory=AudioQCResult)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
