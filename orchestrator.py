"""
orchestrator.py

전체 흐름:
  1) MXFMetaAgent 실행 (다른 Agent들의 입력이 되므로 반드시 먼저 실행)
  2) VideoAnalysisAgent / AudioSTTAgent 를 병렬 실행
  3) SpeakerDiarizationAgent를 가동하여 임시 SPEAKER_00을 진짜 화자명으로 강제 치환
  4) scene_cuts(장면 전환점)들을 구간으로 역산한 후, 각 씬에 썸네일 경로 및 오브젝트 리스트 병합
  5) [싱크 업그레이드] 중앙값(Midpoint) 및 최대 교집합(Overlap) 연산으로 J-cut 대사 매핑 오류 원천 수정
  6) [미디어 플레이어 연동] 비디오 플레이어 화면 고정 및 씬/대사 클릭 시 타임라인 즉시 이동(Seek) 인터랙션 뷰어 구현
"""

import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Any

from agents.mxf_meta_agent import MXFMetaAgent
from agents.video_analysis_agent import VideoAnalysisAgent
from agents.audio_stt_agent import AudioSTTAgent
from agents.speaker_diarization_agent import SpeakerDiarizationAgent
from schema import PipelineReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mxf_pipeline.orchestrator")


class PipelineOrchestrator:
    def __init__(
        self,
        meta_agent: Optional[MXFMetaAgent] = None,
        video_agent: Optional[VideoAnalysisAgent] = None,
        stt_agent: Optional[AudioSTTAgent] = None,
        speaker_agent: Optional[SpeakerDiarizationAgent] = None,
        storyline_agent: Optional[Any] = None, 
        max_workers: int = 2,
    ):
        self.meta_agent = meta_agent or MXFMetaAgent()
        self.video_agent = video_agent or VideoAnalysisAgent()
        self.stt_agent = stt_agent or AudioSTTAgent()
        self.speaker_agent = speaker_agent or SpeakerDiarizationAgent()
        self.storyline_agent = storyline_agent  
        self.max_workers = max_workers

    def run(self, file_path: str, output_dir: Optional[str] = None) -> PipelineReport:
        report = PipelineReport()

        # 1단계: 메타 추출
        meta_result = self.meta_agent.safe_run({"file_path": file_path})
        if not meta_result.ok:
            report.errors.append(f"[{meta_result.agent_name}] {meta_result.error}")
            logger.error("메타 추출 실패 - 이후 단계를 진행할 수 없습니다: %s", meta_result.error)
            return report

        report.mxf_meta = meta_result.data

        base_dir = Path(output_dir) if output_dir else Path(file_path).resolve().parent
        scene_output_dir = base_dir / Path(file_path).stem / "scenes"

        context: Dict = {
            "file_path": file_path,
            "mxf_meta": report.mxf_meta,
            "scene_output_dir": str(scene_output_dir),
            "output_dir": output_dir,  
        }

        # 2단계: 코어 비디오/오디오 분석 병렬 실행
        stage_agents = {
            self.video_agent: "video_analysis",
            self.stt_agent: "audio_stt",
        }

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(agent.safe_run, context): field_name
                       for agent, field_name in stage_agents.items()}

            for future in as_completed(futures):
                field_name = futures[future]
                result = future.result()
                if result.ok:
                    setattr(report, field_name, result.data)
                else:
                    report.errors.append(f"[{result.agent_name}] {result.error}")
                    logger.warning("%s 실패: %s", result.agent_name, result.error)
        
        # 2.5단계: Demucs 보컬 음원 기반 고정밀 화자 분리(Diarization) 가동
        if getattr(report, "audio_stt", None) and report.audio_stt.segments:
            context["audio_stt"] = report.audio_stt
            speaker_result = self.speaker_agent.safe_run(context)
            if speaker_result.ok:
                report.audio_stt = speaker_result.data
            else:
                report.errors.append(f"[{speaker_result.agent_name}] {speaker_result.error}")
                logger.warning("%s 실패: %s", speaker_result.agent_name, speaker_result.error)

        # 3단계: 스토리라인 맥락 분석 진입
        if self.storyline_agent and getattr(report, "audio_stt", None) and report.audio_stt.segments:
            context["audio_stt"] = report.audio_stt
            logger.info("대사 분석 기반 스토리라인 씬 요약 및 썸네일 생성 시작...")
            story_result = self.storyline_agent.safe_run(context)
            if story_result.ok:
                report.storyline_analysis = story_result.data
            else:
                report.errors.append(f"[{story_result.agent_name}] {story_result.error}")

        # 4단계: 중앙값 및 오버랩 면적 기반 고정밀 타임라인 분배 분할 매핑
        logger.info("📊 [구조화] J-Cut/L-Cut 방어 알고리즘 가동, 구간별 대사 싱크 정밀 바인딩 중...")
        scene_based_logs = []
        
        video_data = getattr(report, "video_analysis", None)
        audio_data = getattr(report, "audio_stt", None)
        
        detected_cuts = getattr(video_data, "scene_cuts", []) if video_data else []
        stt_segments = getattr(audio_data, "segments", []) if audio_data else []

        if detected_cuts:
            detected_cuts = sorted(detected_cuts, key=lambda x: getattr(x, "time_sec", 0.0))
            total_cuts = len(detected_cuts)

            scenes_windows = []
            for i, cut in enumerate(detected_cuts):
                start_sec = getattr(cut, "time_sec", 0.0)
                if i + 1 < total_cuts:
                    end_sec = getattr(detected_cuts[i+1], "time_sec", float('inf'))
                else:
                    end_sec = float('inf')
                scenes_windows.append((start_sec, end_sec, cut, i + 1))

            dialogues_per_scene = {info[3]: [] for info in scenes_windows}

            for seg in stt_segments:
                seg_start = getattr(seg, "start_sec", 0.0)
                seg_end = getattr(seg, "end_sec", 0.0)
                seg_mid = (seg_start + seg_end) / 2.0

                best_scene_num = None
                max_overlap = -1.0

                for start_sec, end_sec, _, scene_num in scenes_windows:
                    if start_sec <= seg_mid < end_sec:
                        best_scene_num = scene_num
                        break
                    
                    overlap = min(seg_end, end_sec) - max(seg_start, start_sec)
                    if overlap > max_overlap and overlap > 0:
                        max_overlap = overlap
                        best_scene_num = scene_num

                if best_scene_num is None:
                    for start_sec, end_sec, _, scene_num in scenes_windows:
                        if start_sec <= seg_start < end_sec:
                            best_scene_num = scene_num
                            break

                if best_scene_num is not None:
                    s_tc = getattr(seg, "start_timecode", "")
                    e_tc = getattr(seg, "end_timecode", "")
                    
                    if not e_tc and hasattr(seg, 'end_sec') and seg.end_sec is not None:
                        tot_sec = int(seg.end_sec)
                        h = tot_sec // 3600
                        m = (tot_sec % 3600) // 60
                        s = tot_sec % 60
                        ms = int((seg.end_sec - tot_sec) * 100)
                        e_tc = f"{h:02d}:{m:02d}:{s:02d}.{ms:02d}"
                    elif not e_tc:
                        e_tc = "??:??:??"

                    tc_display = f"{s_tc} ~ {e_tc}"

                    dialogues_per_scene[best_scene_num].append({
                        "timecode": tc_display,
                        "start_timecode": s_tc,
                        "end_timecode": e_tc,
                        "start_sec": seg_start,  # 💡 비디오 Seek를 위한 초 단위 데이터 바인딩
                        "speaker": getattr(seg, "speaker", "SPEAKER_00"),
                        "text": getattr(seg, "text", "").strip()
                    })

            for start_sec, end_sec, cut, scene_num in scenes_windows:
                scene_based_logs.append({
                    "scene_number": scene_num,
                    "start_sec": start_sec,  # 💡 씬 박스 클릭 연동용 초 데이터 바인딩
                    "start_timecode": getattr(cut, "timecode", "00:00:00:00"),
                    "end_timecode": getattr(detected_cuts[scene_num], "timecode", "끝") if scene_num < total_cuts else "끝",
                    "thumbnail_path": getattr(cut, "image_path", ""),
                    "detected_objects": getattr(cut, "detected_objects", []),
                    "dialogues": dialogues_per_scene[scene_num]
                })
        else:
            all_dialogues = []
            for seg in stt_segments:
                s_tc = getattr(seg, "start_timecode", "")
                e_tc = getattr(seg, "end_timecode", "")
                seg_start = getattr(seg, "start_sec", 0.0)
                
                if not e_tc and hasattr(seg, 'end_sec') and seg.end_sec is not None:
                    tot_sec = int(seg.end_sec)
                    h = tot_sec // 3600
                    m = (tot_sec % 3600) // 60
                    s = tot_sec % 60
                    ms = int((seg.end_sec - tot_sec) * 100)
                    e_tc = f"{h:02d}:{m:02d}:{s:02d}.{ms:02d}"
                elif not e_tc:
                    e_tc = "??:??:??"

                tc_display = f"{s_tc} ~ {e_tc}"
                
                all_dialogues.append({
                    "timecode": tc_display,
                    "start_timecode": s_tc,
                    "end_timecode": e_tc,
                    "start_sec": seg_start,
                    "speaker": getattr(seg, "speaker", "SPEAKER_00"),
                    "text": getattr(seg, "text", "").strip()
                })

            start_tc = getattr(report.mxf_meta, "start_timecode", "00:00:00:00") if report.mxf_meta else "00:00:00:00"
            scene_based_logs.append({
                "scene_number": 1,
                "start_sec": 0.0,
                "start_timecode": start_tc,
                "end_timecode": "끝",
                "thumbnail_path": "",
                "detected_objects": [],
                "dialogues": all_dialogues
            })

        report.scene_based_logs = scene_based_logs
        return report

    @staticmethod
    def save_report(report: PipelineReport, out_path: str) -> None:
        meta_data = {}
        if getattr(report, "mxf_meta", None):
            try:
                meta_data = asdict(report.mxf_meta) if hasattr(report.mxf_meta, "__dataclass_fields__") else report.mxf_meta.__dict__
            except Exception:
                meta_data = str(report.mxf_meta)

        storyline_data = []
        if getattr(report, "storyline_analysis", None):
            shifts = getattr(report.storyline_analysis, "shifts", [])
            for shift in shifts:
                storyline_data.append({
                    "start_timecode": getattr(shift, "start_timecode", ""),
                    "summary": getattr(shift, "summary", ""),
                    "thumbnail_path": getattr(shift, "thumbnail_path", "")
                })

        scene_based_logs = getattr(report, "scene_based_logs", [])

        final_clean_json = {
            "meta": meta_data,
            "storyline_summary": storyline_data,
            "total_scenes": len(scene_based_logs),
            "scenes": scene_based_logs  
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_clean_json, f, ensure_ascii=False, indent=2, default=str)
        logger.info("리포트 JSON 저장 완료: %s", out_path)

        html_path = out_path.replace(".json", ".html")
        
        fn = os.path.basename(meta_data.get("file_path", "Unknown File"))
        fps = meta_data.get("fps", 25.0)
        stc = meta_data.get("start_timecode", "00:00:00:00")
        dur = meta_data.get("duration_sec", 0.0)
        h, rem = divmod(int(dur), 3600)
        m, s = divmod(rem, 60)
        duration_str = f"{h:02d}:{m:02d}:{s:02d}"

        story_cards_html = ""
        for s_item in storyline_data:
            s_thumb = s_item.get('thumbnail_path', '')
            s_thumb_url = Path(s_thumb).as_posix() if s_thumb else "https://placehold.co/300x169/1e293b/cbd5e1?text=No+Image"
            story_cards_html += f"""
            <div class="story-card">
                <img src="{s_thumb_url}" alt="Story Thumbnail" onerror="this.src='https://placehold.co/300x169/1e293b/cbd5e1?text=Thumbnail'">
                <div class="story-card-body">
                    <span class="tc-badge">{s_item.get('start_timecode')}</span>
                    <p class="story-text">{s_item.get('summary')}</p>
                </div>
            </div>
            """

        scenes_section_html = ""
        for sc in scene_based_logs:
            sc_thumb = sc.get('thumbnail_path', '')
            sc_thumb_url = Path(sc_thumb).as_posix() if sc_thumb else "https://placehold.co/300x169/1e293b/cbd5e1?text=No+Image"

            obj_badges = ""
            for obj in sc.get("detected_objects", []):
                obj_badges += f'<span class="obj-badge">{obj}</span>'
            if not obj_badges:
                obj_badges = '<span class="obj-badge-none">감지된 사물 없음</span>'

            dialogues_html = ""
            for dlg in sc.get("dialogues", []):
                speaker = dlg.get("speaker", "SPEAKER_00")
                spk_color_idx = int(speaker.split("_")[-1]) % 4 if "_" in speaker else 0
                spk_class = f"spk-clr-{spk_color_idx}"

                # 💡 대사 라인 클릭 시 자바스크립트 seekTo() 함수 연동 (상위 씬박스 클릭 이벤트 버블링 방지 포함)
                dialogues_html += f"""
                <div class="dlg-row" onclick="event.stopPropagation(); seekTo({dlg.get('start_sec', 0.0)})">
                    <span class="dlg-tc">▶ {dlg.get('timecode')}</span>
                    <span class="dlg-spk {spk_class}">{speaker}</span>
                    <span class="dlg-text">{dlg.get('text')}</span>
                </div>
                """
            if not dialogues_html:
                dialogues_html = '<div class="dlg-row-none">이 구간에는 대사가 존재하지 않습니다.</div>'

            # 💡 씬 박스 자체를 클릭 가능한 영역으로 지정하고 마우스 포인터 활성화 스타일 부여
            scenes_section_html += f"""
            <div class="scene-box" onclick="seekTo({sc.get('start_sec', 0.0)})">
                <div class="scene-header">
                    <div class="scene-title">🎬 SCENE #{sc.get('scene_number')} <span style="font-size:12px; color:#93c5fd; font-weight:normal; margin-left:8px;">(클릭 시 이동)</span></div>
                    <div class="scene-duration">⏳ {sc.get('start_timecode')} ~ {sc.get('end_timecode')}</div>
                </div>
                
                <div class="scene-split-layout">
                    <div class="scene-visual-block">
                        <div class="meta-label">🖼️ 씬 대표 썸네일 (YOLO11):</div>
                        <img src="{sc_thumb_url}" class="scene-img-frame" alt="Scene Frame" onerror="this.src='https://placehold.co/300x169/1e293b/cbd5e1?text=No+Frame+Captured'">
                    </div>
                    
                    <div class="scene-data-block">
                        <div class="meta-label">👁️ 화면 식별 오브젝트:</div>
                        <div class="badge-container">{obj_badges}</div>
                        
                        <div class="meta-label" style="margin-top: 14px;">💬 타임라인 대사 스포팅 (클릭 이동):</div>
                        <div class="dlg-container">
                            {dialogues_html}
                        </div>
                    </div>
                </div>
            </div>
            """

        html_template = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MXF AI Media Pipeline Report</title>
    <style>
        :root {{
            --bg-main: #0f172a;
            --bg-card: #1e293b;
            --bg-input: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #6366f1;
            --accent: #10b981;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1650px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #4f46e5, #06b6d4);
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 32px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        header h1 {{ margin: 0 0 12px 0; font-size: 28px; font-weight: 700; letter-spacing: -0.5px; }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            background: rgba(0, 0, 0, 0.2);
            padding: 16px;
            border-radius: 8px;
        }}
        .meta-grid div {{ font-size: 14px; }}
        .meta-grid div strong {{ color: #a5f3fc; display: block; margin-bottom: 2px; }}
        
        h2 {{ font-size: 20px; border-left: 5px solid var(--primary); padding-left: 10px; margin: 30px 0 20px 0; color: #cbd5e1; }}
        
        /* 💡 플레이어와 타임라인 배치를 위한 2단 스플릿 레이아웃 */
        .workspace {{
            display: flex;
            gap: 32px;
            align-items: flex-start;
        }}
        
        /* 💡 좌측 비디오 플레이어 사이드바 (스크롤 시 화면에 고정) */
        .player-sidebar {{
            position: sticky;
            top: 24px;
            width: 480px;
            flex-shrink: 0;
            background-color: var(--bg-card);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.4);
        }}
        
        .video-box {{
            width: 100%;
            background-color: #000;
            border-radius: 8px;
            overflow: hidden;
            aspect-ratio: 16/9;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.8);
            margin-bottom: 16px;
        }}
        
        .file-selector-btn {{
            display: block;
            width: 100%;
            background-color: #312e81;
            border: 1px dashed #6366f1;
            color: #c7d2fe;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.2s;
        }}
        .file-selector-btn:hover {{ background-color: #3730a3; border-color: #818cf8; }}
        #video-uploader {{ display: none; }}

        /* 💡 우측 타임라인 콘텐츠 영역 */
        .content-area {{
            flex-grow: 1;
            min-width: 0;
        }}
        
        .story-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 30px;
        }}
        .story-card {{
            background-color: var(--bg-card);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #334155;
        }}
        .story-card img {{ width: 100%; height: 150px; object-fit: cover; background: #0f172a; }}
        .story-card-body {{ padding: 12px; }}
        .tc-badge {{
            display: inline-block;
            background-color: var(--primary);
            color: white;
            font-size: 11px;
            font-weight: bold;
            padding: 1px 6px;
            border-radius: 4px;
            margin-bottom: 6px;
            font-family: monospace;
        }}
        .story-text {{ margin: 0; font-size: 13.5px; color: #e2e8f0; word-break: keep-all; }}

        .timeline-container {{ display: flex; flex-direction: column; gap: 24px; }}
        
        /* 💡 클릭 가능한 씬 박스 스타일 조정 */
        .scene-box {{
            background-color: var(--bg-card);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #334155;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            cursor: pointer;
            transition: all 0.2s;
        }}
        .scene-box:hover {{ border-color: var(--primary); transform: translateY(-2px); background-color: #24324d; }}
        
        .scene-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #334155;
            padding-bottom: 12px;
            margin-bottom: 16px;
        }}
        .scene-title {{ font-size: 18px; font-weight: bold; color: #60a5fa; }}
        .scene-duration {{ font-family: monospace; font-size: 15px; color: var(--text-muted); font-weight: bold; }}
        
        .scene-split-layout {{ display: flex; gap: 20px; align-items: flex-start; }}
        .scene-visual-block {{ width: 280px; flex-shrink: 0; }}
        .scene-img-frame {{
            width: 100%;
            height: 157px;
            object-fit: cover;
            border-radius: 8px;
            border: 1px solid #475569;
            background-color: #0f172a;
        }}
        .scene-data-block {{ flex-grow: 1; min-width: 0; }}

        .meta-label {{ font-size: 12px; font-weight: bold; color: var(--text-muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .badge-container {{ background: rgba(0,0,0,0.2); padding: 8px; border-radius: 6px; border: 1px solid #1e293b; }}
        .obj-badge {{
            display: inline-block;
            background: #1e3a8a;
            color: #93c5fd;
            border: 1px solid #2563eb;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 12px;
            margin-right: 4px;
            margin-bottom: 2px;
        }}
        .obj-badge-none {{ font-size: 12px; color: var(--text-muted); font-style: italic; }}
        
        .dlg-container {{
            background-color: #0f172a;
            border-radius: 8px;
            padding: 4px;
            border: 1px solid #1e293b;
            max-height: 200px;
            overflow-y: auto;
        }}
        .dlg-container::-webkit-scrollbar {{ width: 6px; }}
        .dlg-container::-webkit-scrollbar-thumb {{ background: #475569; border-radius: 4px; }}
        
        /* 💡 대사 행 클릭 포인터 디자인 */
        .dlg-row {{
            display: flex;
            padding: 6px 10px;
            border-bottom: 1px solid #1e293b;
            font-size: 13.5px;
            align-items: flex-start;
            cursor: pointer;
            transition: background 0.1s;
        }}
        .dlg-row:last-child {{ border-bottom: none; }}
        .dlg-row:hover {{ background-color: #1e293b; color: #fff; }}
        
        .dlg-tc {{ font-family: monospace; color: var(--accent); min-width: 210px; flex-shrink: 0; font-weight: bold; margin-right: 12px; }}
        .dlg-spk {{ width: 110px; flex-shrink: 0; font-weight: bold; font-size: 12.5px; font-family: monospace; }}
        .dlg-text {{ color: #e2e8f0; word-break: break-all; }}
        .dlg-row-none {{ padding: 15px; font-size: 12px; color: var(--text-muted); text-align: center; font-style: italic; }}

        .spk-clr-0 {{ color: #f43f5e; }}
        .spk-clr-1 {{ color: #3b82f6; }}
        .spk-clr-2 {{ color: #eab308; }}
        .spk-clr-3 {{ color: #a855f7; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎥 AI Media Pipeline Analysis Report</h1>
            <div class="meta-grid">
                <div><strong>파일명 (File Name)</strong> {fn}</div>
                <div><strong>타임코드 오프셋 (Start TC)</strong> {stc}</div>
                <div><strong>프레임 레이트 (FPS)</strong> {fps} fps</div>
                <div><strong>총 미디어 길이 (Duration)</strong> {duration_str}</div>
                <div><strong>총 감지된 장면 수</strong> {len(scene_based_logs)} 개의 씬</div>
            </div>
        </header>

        <div class="workspace">
            
            <div class="player-sidebar">
                <div class="meta-label" style="color: #61dafb; font-size: 13px; margin-bottom: 10px;">🎯 미디어 타임라인 플레이어</div>
                <video id="video-player" class="video-box" controls></video>
                
                <label for="video-uploader" class="file-selector-btn">
                    📂 분석 영상 파일 로드 (.mp4 권장)
                </label>
                <input type="file" id="video-uploader" accept="video/*">
                
                <div style="margin-top: 15px; font-size: 12px; color: var(--text-muted); line-height: 1.4; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 6px;">
                    💡 <strong>사용 방법:</strong> 프로젝트 소스 영상(또는 MP4 프록시)을 로드한 뒤, 우측의 <strong>씬(SCENE) 박스</strong>나 <strong>대사 타임라인</strong>을 클릭하면 해당 시간으로 자동 점프합니다.
                </div>
            </div>

            <div class="content-area">
                <h2>🧠 로컬 Qwen3.6 스토리라인 요약 및 대표 맥락 분기점</h2>
                <div class="story-grid">
                    {story_cards_html}
                </div>

                <h2>📊 씬 단위 계층 타임라인 (물리 썸네일 & 사물 인지 & 화자 대본 결합)</h2>
                <div class="timeline-container">
                    {scenes_section_html}
                </div>
            </div>
            
        </div>
    </div>

    <script>
        const player = document.getElementById('video-player');
        const uploader = document.getElementById('video-uploader');

        // 로컬 비디오 파일 브라우저 보안 우회 로더 Hook
        uploader.addEventListener('change', function(e) {{
            const file = e.target.files[0];
            if (file) {{
                const fileURL = URL.createObjectURL(file);
                player.src = fileURL;
                player.play().catch(err => console.log("자동 재생 정책으로 일시 정지됨."));
            }}
        }});

        // 클릭 시 지정된 시간 초(seconds)로 즉시 이동하는 제어 함수
        function seekTo(seconds) {{
            if (!player.src) {{
                alert("⚠️ 비디오가 로드되지 않았습니다! 왼쪽 패널에서 영상 파일을 먼저 선택해 주세요.");
                return;
            }}
            player.currentTime = parseFloat(seconds);
            player.play().catch(() => {{}});
            
            // 시각적 피드백 효과: 플레이어 일시 반짝임
            player.style.outline = "3px solid #10b981";
            setTimeout(() => player.style.outline = "none", 400);
        }}
    </script>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_template)
        logger.info("리포트 웹뷰 대시보드 HTML 저장 완료: %s", html_path)