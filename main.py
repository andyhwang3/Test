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
from agents.speaker_diarization_agent import SpeakerDiarizationAgent  # ⬅ 이 줄 추가
from agents.storyline_agent import StorylineAgent  # ⭐ [추가]
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
    p.add_argument("--video-gpu", type=int, default=1, help="영상 분석(YOLO)에 쓸 GPU 인덱스")   # ⬅ 추가
    p.add_argument("--stt-gpu", type=int, default=0, help="STT/Demucs/화자분리에 쓸 GPU 인덱스")   # ⬅ 추가
    return p.parse_args()


def main():
    args = parse_args()

    orchestrator = PipelineOrchestrator(
        meta_agent=MXFMetaAgent(),
        video_agent=VideoAnalysisAgent(extract_frames=not args.no_scene_images, gpu_id=args.video_gpu),
        stt_agent=AudioSTTAgent(
            model_size=args.whisper_model,
            device=args.whisper_device,
            gpu_id=args.stt_gpu,
            dialogue_track_index=args.dialogue_track,
        ),
        # qc_agent=AudioQCAgent(),
        speaker_agent=SpeakerDiarizationAgent(gpu_id=args.stt_gpu),  # STT와 같은 GPU에 두는 게 VRAM 재사용 측면에서 유리
        storyline_agent=StorylineAgent(ollama_model="qwen3.6"),  # ⭐ [추가] 에이전트 주입
    )

    report = orchestrator.run(args.input, output_dir=args.output_dir)
    
    # ⭐ [수정 및 추가] --output-dir 경로가 있고, --output이 상대 경로(파일명)일 때 경로 결합
    import os
    output_report_path = args.output
    if args.output_dir and not os.path.isabs(output_report_path):
        output_report_path = os.path.join(args.output_dir, output_report_path)
        # 만약 지정한 폴더(D:\results 등)가 생성되어 있지 않다면 안전하게 자동 생성
        os.makedirs(args.output_dir, exist_ok=True)

    # ⭐ 기존 args.output 대신 새로 계산된 output_report_path 로 저장합니다.
    orchestrator.save_report(report, output_report_path)

    if report.errors:
        print(f"완료 (일부 단계 오류 있음): {args.output}")
        for err in report.errors:
            print(f"  - {err}")
    else:
        print(f"완료: {args.output}")


if __name__ == "__main__":
    main()
