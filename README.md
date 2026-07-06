# MXF Ingest QC 파이프라인

MXF 파일 하나를 넣으면 다음을 자동으로 수행합니다.

1. **메타 추출** (`agents/mxf_meta_agent.py`) - ffprobe(+선택적 bmxlib)로 코덱/해상도/fps/타임코드/오디오 트랙 구성을 파싱
2. **영상 분석** (`agents/video_analysis_agent.py`) - PySceneDetect로 장면전환 프레임/타임코드 추출 + 해당 프레임을 PNG로 저장, ffmpeg blackdetect/freezedetect로 화질 이상 탐지
3. **Audio STT** (`agents/audio_stt_agent.py`) - 대사 트랙을 자동 추정(또는 지정)하여 faster-whisper로 전사
4. **Audio QC** (`agents/audio_qc_agent.py`) - EBU R128 Loudness/True Peak, 클리핑, 무음, 클릭/팝 탐지

2~4단계는 서로 독립적이라 `orchestrator.py`에서 스레드풀로 병렬 실행됩니다.

## 장면전환 PNG 출력

파일마다 자동으로 폴더가 생성되고, 그 안에 장면전환 지점의 프레임이 PNG로 저장됩니다.

```
<output-dir>/<입력파일명(확장자 제외)>/scenes/
  ├── scene_0001_00-00-00-00.png
  ├── scene_0002_00-00-12-05.png
  └── scene_0003_00-00-27-14.png
```

- 파일명은 `scene_<순번>_<타임코드>.png` 형식이며, 타임코드의 `:`는 파일명에 쓸 수 없어 `-`로 치환됩니다.
- `--output-dir`을 지정하지 않으면 입력 MXF 파일과 같은 폴더 기준으로 생성됩니다.
- PNG 추출을 건너뛰고 타임코드 목록만 필요하면 `--no-scene-images` 옵션을 사용하세요.
- 각 `SceneCut` 객체의 `image_path` 필드에도 해당 PNG 경로가 함께 기록되어 `report.json`에서 바로 확인할 수 있습니다.

## 설치

```bash
# 시스템 의존성 (Ubuntu 예시)
sudo apt-get install ffmpeg

# Python 의존성
pip install -r requirements.txt
```

## 실행

```bash
python main.py --input sample.mxf --output report.json --output-dir ./results
```

GPU(RTX A4000 등)에서 Whisper를 돌리려면 `--whisper-device cuda` (기본값),
CPU만 있다면 `--whisper-device cpu --whisper-model medium` 권장.

완전히 로컬에서만 동작하며 외부 오케스트레이션 도구(n8n 등)에 대한 의존성은 없습니다.
여러 파일을 순차/배치로 돌리고 싶다면 `main.py`를 셸 스크립트나 `for` 루프로 감싸서 반복 호출하면 됩니다.

```bash
for f in /path/to/mxf/*.mxf; do
  python main.py --input "$f" --output "$(basename "$f" .mxf).json"
done
```

## 확장 포인트

- `agents/base.py`의 `BaseAgent`만 상속하면 새 Agent(예: A/V 싱크 드리프트 전용 Agent)를 손쉽게 추가 가능
- `orchestrator.py`의 `stage_agents` 딕셔너리에 등록만 하면 자동으로 병렬 실행 대상에 포함됨
- 대용량 배치 처리 시 `ThreadPoolExecutor` -> 여러 파일을 동시에 큐잉하는 상위 레벨 스케줄러(Celery, n8n Queue 등)로 교체 권장
