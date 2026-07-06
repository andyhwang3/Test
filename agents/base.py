"""
agents/base.py
모든 Agent가 상속하는 공통 인터페이스.
Orchestrator는 이 인터페이스만 알면 되고, 각 Agent 내부 구현(ffmpeg 호출, 모델 추론 등)은 몰라도 된다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict
import logging

logger = logging.getLogger("mxf_pipeline")


@dataclass
class AgentResult:
    agent_name: str
    ok: bool
    data: Any = None
    error: str = ""
    duration_sec: float = 0.0


class BaseAgent(ABC):
    """
    각 Agent는 run(context)만 구현하면 된다.
    context: 이전 단계 결과가 담긴 dict (예: {"mxf_meta": MXFBaseMeta, "file_path": str, ...})
    반환값: 해당 단계의 결과 dataclass 인스턴스 (schema.py 참조)
    """

    name: str = "base_agent"

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Any:
        ...

    def safe_run(self, context: Dict[str, Any]) -> AgentResult:
        start = perf_counter()
        try:
            data = self.run(context)
            return AgentResult(
                agent_name=self.name,
                ok=True,
                data=data,
                duration_sec=perf_counter() - start,
            )
        except Exception as exc:  # noqa: BLE001 - 파이프라인 단계 실패를 전체 중단시키지 않기 위함
            logger.exception("%s 실행 중 오류", self.name)
            return AgentResult(
                agent_name=self.name,
                ok=False,
                error=str(exc),
                duration_sec=perf_counter() - start,
            )
