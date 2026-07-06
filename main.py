"""
main.py - CLI 진입점

사용 예:
  python main.py --input sample.mxf --output report.json
  python main.py --input sample.mxf --output report.json --whisper-device cpu --dialogue-track 0
  python main.py --input sample.mxf --output report.json --output-dir ./results

파일별 장면전환 PNG는 <output-dir>/<입력파일명(확장자 제외)>/scenes/ 폴더에 자동 생성됩니다.
--output-dir을 지정하지 않으면 입력 파일과 같은 폴더 기준으로 생성됩니다.
"""

import argparse

from agents.mxf_meta_agent import MXFMetaAgent
from agents.video_analysis_agent import VideoAnalysisAgent
from agents.audio_stt_agent import AudioSTTAgent
from agents.audio_qc_agent import AudioQCAgent
from orchestrator import PipelineOrchestrator


def parse_args():
    p = argparse.ArgumentParser(description="MXF Ingest QC 파이프라인")
    p.add_argument("--input", required=True, help="입력 MXF 파일 경로")
    p.add_argument("--output", default="report.json", help="결과 JSON 저장 경로")
    p.add_argument("--output-dir", default=None,
                   help="파일별 장면전환 PNG 폴더를 생성할 기준 경로 (미지정 시 입력 파일과 같은 폴더)")
    p.add_argument("--no-scene-images", action="store_true",
                   help="장면전환 PNG 추출을 건너뛰고 타임코드 목록만 생성")
    p.add_argument("--whisper-model", default="large-v3", help="faster-whisper 모델 크기")
    p.add_argument("--whisper-device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dialogue-track", type=int, default=None,
                   help="대사 오디오 트랙 인덱스를 직접 지정 (미지정 시 자동 추정)")
    return p.parse_args()


def main():
    args = parse_args()

    orchestrator = PipelineOrchestrator(
        meta_agent=MXFMetaAgent(),
        video_agent=VideoAnalysisAgent(extract_frames=not args.no_scene_images),
        stt_agent=AudioSTTAgent(
            model_size=args.whisper_model,
            device=args.whisper_device,
            dialogue_track_index=args.dialogue_track,
        ),
        qc_agent=AudioQCAgent(),
    )

    report = orchestrator.run(args.input, output_dir=args.output_dir)
    orchestrator.save_report(report, args.output)

    if report.errors:
        print(f"완료 (일부 단계 오류 있음): {args.output}")
        for err in report.errors:
            print(f"  - {err}")
    else:
        print(f"완료: {args.output}")


if __name__ == "__main__":
    main()
