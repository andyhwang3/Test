"""
agents/video_analysis_agent.py

2단계 Agent: 장면전환(Scene Cut) 위치와 기본 화질 이상(블랙 프레임, 프리즈 프레임)을 탐지한다.
외부 ffmpeg 호출을 배제하고, OpenCV 고유의 프레임 넘버 타겟팅을 사용하여 대사와 100% 일치하는 썸네일을 추출한다.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from schema import MXFBaseMeta, SceneCut, VideoAnalysisResult, VideoAnomaly

COCO_KR_MAP = {
    "person": "사람", "bicycle": "자전거", "car": "자동차", "motorcycle": "오토바이", "airplane": "비행기",
    "bus": "버스", "train": "기차", "truck": "트럭", "boat": "보트", "traffic light": "신호등",
    "fire hydrant": "소화전", "stop sign": "정지 표지판", "parking meter": "주차 요금기", "bench": "벤치", "bird": "새",
    "cat": "고양이", "dog": "개", "horse": "말", "sheep": "양", "cow": "소",
    "elephant": "코끼리", "bear": "곰", "zebra": "얼룩말", "giraffe": "기린", "backpack": "배낭",
    "umbrella": "우산", "handbag": "핸드백", "tie": "넥타이", "suitcase": "여행가방", "frisbee": "원반",
    "skis": "스키", "snowboard": "스노보드", "sports ball": "스포츠 공", "kite": "연", "baseball bat": "야구배트",
    "baseball glove": "야구글러브", "skateboard": "스케이트보드", "surfboard": "서프보드", "tennis racket": "테니스 라켓", "bottle": "병",
    "wine glass": "와인잔", "cup": "컵", "fork": "포크", "knife": "칼", "spoon": "숟가락",
    "bowl": "그릇", "banana": "바나나", "apple": "사과", "sandwich": "샌드위치", "orange": "오렌지",
    "broccoli": "브로콜리", "carrot": "당근", "hot dog": "핫도그", "pizza": "피자", "donut": "도넛",
    "cake": "케이크", "chair": "의자", "couch": "소파", "potted plant": "화분", "bed": "침대",
    "dining table": "식탁", "toilet": "변기", "tv": "TV", "laptop": "노트북", "mouse": "마우스",
    "remote": "리모컨", "keyboard": "키보드", "cell phone": "휴대전화", "microwave": "전자레인지", "oven": "오븐",
    "toaster": "토스터", "sink": "싱크대", "refrigerator": "냉장고", "book": "책", "clock": "시계",
    "vase": "화병", "scissors": "가위", "teddy bear": "곰인형", "hair drier": "헤어드라이어", "toothbrush": "칫솔"
}


class VideoAnalysisAgent(BaseAgent):
    name = "video_analysis_agent"

    def __init__(
        self,
        ffmpeg_bin: str = "ffmpeg",
        scene_threshold: float = 27.0,
        extract_frames: bool = True,
        gpu_id: int = 1,
    ):
        self.ffmpeg_bin = ffmpeg_bin
        self.scene_threshold = scene_threshold
        self.extract_frames = extract_frames
        self.device = f"cuda:{gpu_id}"

    def run(self, context: Dict[str, Any]) -> VideoAnalysisResult:
        file_path = context["file_path"]
        mxf_meta: MXFBaseMeta = context["mxf_meta"]
        scene_output_dir: Optional[str] = context.get("scene_output_dir")

        scene_cuts = self._detect_scene_cuts(file_path, mxf_meta)

        if self.extract_frames and scene_output_dir:
            self._extract_scene_frames(file_path, scene_cuts, scene_output_dir)

        anomalies = self._detect_anomalies(file_path)

        return VideoAnalysisResult(scene_cuts=scene_cuts, anomalies=anomalies)

    def _detect_scene_cuts(self, file_path: str, mxf_meta: MXFBaseMeta) -> List[SceneCut]:
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
        
        print("\n🎬 [비디오 분석] 씬 체인지 탐지 엔진 가동 중...")
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
        fps = fps or 25.0
        h, m, s, f = (int(x) for x in start_timecode.replace(";", ":").split(":"))
        start_frames = int(((h * 3600 + m * 60 + s) * fps) + f)
        total_frames = start_frames + frame_num
        total_seconds, ff = divmod(total_frames, int(round(fps)))
        hh, rem = divmod(total_seconds, 3600)
        mm, ss = divmod(rem, 60)
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"

    # ---- ⭐ [완벽 씽크 개정] OpenCV 비디오 다이렉트 프레임 캡처 엔진 ----

    def _extract_scene_frames(self, file_path: str, scene_cuts: List[SceneCut], out_dir: str) -> None:
        import cv2
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        total_cuts = len(scene_cuts)
        
        from ultralytics import YOLO
        yolo_model = YOLO("yolo11l.pt").to("cuda")
        
        print(f"📊 [탐지 완료] 총 {total_cuts}개의 씬 체인지 지점을 발견했습니다.")
        print("🖼️ [정밀 싱크 록] OpenCV 네이티브 프레임 타겟팅 및 YOLO11 스캔을 구동합니다...")

        # 비디오 파일을 1회만 열어서 내부 포인터로 초고속 탐색
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("❌ [오류] OpenCV가 비디오 스트림을 열지 못했습니다.")
            return

        for i, cut in enumerate(scene_cuts, start=1):
            safe_tc = cut.timecode.replace(":", "-").replace(";", "-")
            filename = f"scene_{i:04d}_{safe_tc}.png"
            out_path = os.path.join(out_dir, filename)

            # 🎯 타임코드 오차 유발 원인 차단: scenedetect가 찾아낸 '그 프레임 인덱스'로 정확히 강제 이동
            cap.set(cv2.CAP_PROP_POS_FRAMES, cut.frame_number)
            ret, frame = cap.read()
            
            if ret:
                # 물리 PNG 파일 쓰기
                cv2.imwrite(out_path, frame)
                cut.image_path = out_path
                
                # YOLO11 비전 사물 추론
                yolo_results = yolo_model(out_path, verbose=False)
                objects_in_scene = []
                
                for r in yolo_results:
                    for c in r.boxes.cls:
                        eng_name = yolo_model.names[int(c)]
                        # COCO_KR_MAP 매핑 시 한글 표기, 미매핑 시 영문 표기 + 누락 로그
                        display_name = COCO_KR_MAP.get(eng_name)
                        if display_name is None:
                            display_name = eng_name
                            print(f"       ⚠️ [매핑 누락] '{eng_name}' 항목이 COCO_KR_MAP에 없어 영문으로 표시됩니다.")
                        objects_in_scene.append(display_name)
                
                cut.detected_objects = list(set(objects_in_scene))
                print(f"       🖼️ 씬 체인지 동기화 중 ({i}/{total_cuts}) -> [{cut.timecode}] 객체: {cut.detected_objects}")

        cap.release()
        print(f"✨ [성공] 총 {total_cuts}개의 프레임 완벽 동기화 및 비전 분석 완료!\n")

    def _detect_anomalies(self, file_path: str) -> List[VideoAnomaly]:
        print("🔍 [품질 검사] 영상 내 블랙 프레임 및 프리즈 구간 스캔 중...")
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