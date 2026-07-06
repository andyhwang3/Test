"""
orchestrator.py

전체 흐름:
  1) MXFMetaAgent 실행 (다른 Agent들의 입력이 되므로 반드시 먼저 실행)
  2) VideoAnalysisAgent / AudioSTTAgent / AudioQCAgent 를 병렬 실행
     (전부 subprocess/GPU 바운드 작업이라 ThreadPoolExecutor로 충분히 병렬화 가능)
  3) 결과를 PipelineReport 하나로 병합 후 JSON 저장

CLI에서 --input/--output만 넘기면 순수 로컬 실행으로 끝난다 (외부 오케스트레이션 도구 불필요).
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

from agents.mxf_meta_agent import MXFMetaAgent
from agents.video_analysis_agent import VideoAnalysisAgent
from agents.audio_stt_agent import AudioSTTAgent
from agents.audio_qc_agent import AudioQCAgent
from schema import PipelineReport, VideoAnalysisResult, AudioSTTResult, AudioQCResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mxf_pipeline.orchestrator")


class PipelineOrchestrator:
    def __init__(
        self,
        meta_agent: Optional[MXFMetaAgent] = None,
        video_agent: Optional[VideoAnalysisAgent] = None,
        stt_agent: Optional[AudioSTTAgent] = None,
        qc_agent: Optional[AudioQCAgent] = None,
        max_workers: int = 3,
    ):
        self.meta_agent = meta_agent or MXFMetaAgent()
        self.video_agent = video_agent or VideoAnalysisAgent()
        self.stt_agent = stt_agent or AudioSTTAgent()
        self.qc_agent = qc_agent or AudioQCAgent()
        self.max_workers = max_workers

    def run(self, file_path: str, output_dir: Optional[str] = None) -> PipelineReport:
        report = PipelineReport()

        # 1단계: 메타 추출 (순차 - 이후 단계의 입력값)
        meta_result = self.meta_agent.safe_run({"file_path": file_path})
        if not meta_result.ok:
            report.errors.append(f"[{meta_result.agent_name}] {meta_result.error}")
            logger.error("메타 추출 실패 - 이후 단계를 진행할 수 없습니다: %s", meta_result.error)
            return report

        report.mxf_meta = meta_result.data

        # 파일명 기준으로 "<output_dir>/<파일명(확장자 제외)>/scenes/" 폴더를 준비
        # 예: sample.mxf -> ./sample/scenes/scene_0001_00-00-12-05.png
        base_dir = Path(output_dir) if output_dir else Path(file_path).resolve().parent
        scene_output_dir = base_dir / Path(file_path).stem / "scenes"

        context: Dict = {
            "file_path": file_path,
            "mxf_meta": report.mxf_meta,
            "scene_output_dir": str(scene_output_dir),
        }

        # 2단계: 병렬 분석
        stage_agents = {
            self.video_agent: "video_analysis",
            self.stt_agent: "audio_stt",
            self.qc_agent: "audio_qc",
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
                    logger.warning("%s 실패: %s (다른 단계는 계속 진행됨)", result.agent_name, result.error)

        return report

    @staticmethod
    def save_report(report: PipelineReport, out_path: str) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        logger.info("리포트 저장 완료: %s", out_path)
