"""
agents/storyline_agent.py

5단계 Agent: 화자 분리가 완료된 대사 타임라인 데이터를 기반으로 
로컬 Ollama(Qwen 3.6)를 호출하여 스토리라인 분기점을 추론하고 한글 씬 요약을 생성한다.
OpenCV 프레임 매퍼를 도입하여 타임라인 싱크 어긋남 문제를 완전히 치료한다.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from agents.base import BaseAgent
from schema import AudioSTTResult, StorylineAnalysisResult, StorylineShift


class StorylineAgent(BaseAgent):
    name = "storyline_agent"

    def __init__(self, ollama_model: str = "qwen3.6", ffmpeg_bin: str = "ffmpeg"):
        self.ollama_model = ollama_model
        self.ffmpeg_bin = ffmpeg_bin

    def run(self, context: Dict[str, Any]) -> StorylineAnalysisResult:
        file_path = context["file_path"]
        mxf_meta = context["mxf_meta"]
        output_dir = context.get("output_dir")
        audio_stt: AudioSTTResult = context["audio_stt"]

        if not audio_stt.segments:
            return StorylineAnalysisResult()

        text_timeline = ""
        for seg in audio_stt.segments:
            speaker_label = getattr(seg, "speaker", "SPEAKER_00")
            # 🎯 [수정] 프롬프트 입력 시 대사 시작초와 종료초 함께 전달
            text_timeline += f"[{seg.start_sec:.1f}s ~ {seg.end_sec:.1f}s][{speaker_label}] {seg.text}\n"

        try:
            from ollama import Client
        except ImportError as exc:
            raise RuntimeError("ollama 라이브러리가 필요합니다. 'pip install ollama'") from exc

        prompt = f"""
        아래의 [화자별 대사 타임라인 데이터]를 분석하여 이야기의 대주제, 사건, 장소 또는 맥락(스토리라인)이 크게 바뀌는 주요 전환점(Scene Shift Points)들을 구별하세요.
        반드시 'shifts'라는 키를 가진 JSON 객체(Object) 형태로만 출력해야 합니다. 다른 문장이나 부연 설명은 절대 금지합니다.

        [출력 구조 예시]
        {{
          "shifts": [
            {{"start_sec": 0.0, "summary": "오프닝 및 주요 인물 등장 예고"}},
            {{"start_sec": 145.2, "summary": "피고인의 진술 번복 및 법정 갈등 고조"}}
          ]
        }}

        [화자별 대사 타임라인 데이터]
        {text_timeline}
        """

        print(f"\n[Step 1/3] 📝 화자 분리 대사 정제 완료 (총 {len(audio_stt.segments)}개 문장 / {len(text_timeline)}자)")
        print(f"[Step 2/3] 🧠 로컬 {self.ollama_model} 엔진에 컨텍스트 주입 및 맥락 분석 시작...")
        
        stop_event = threading.Event()

        def visual_ticker():
            start_time = time.time()
            spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            idx = 0
            while not stop_event.is_set():
                elapsed = time.time() - start_time
                sys.stdout.write(f"\r⏳ AI가 문맥을 흡수하여 JSON 구조를 연산 중입니다... [{elapsed:.1f}초 경과] {spinner[idx % len(spinner)]}")
                sys.stdout.flush()
                idx += 1
                time.sleep(0.1)
            sys.stdout.write("\r" + " " * 90 + "\r")
            sys.stdout.flush()

        ticker_thread = threading.Thread(target=visual_ticker, daemon=True)
        ticker_thread.start()
        
        try:
            client = Client(timeout=600.0)
            response_stream = client.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                stream=True 
            )

            full_content = ""
            first_text_received = False
            
            for chunk in response_stream:
                content = chunk.get('message', {}).get('content', '')
                full_content += content
                
                if not first_text_received and any(c.isalnum() for c in content):
                    stop_event.set()
                    ticker_thread.join()
                    print("\n" + "🤖" + "="*23 + f" {self.ollama_model.upper()} REAL-TIME STREAMING " + "="*23)
                    print(full_content, end="", flush=True)
                    first_text_received = True
                elif first_text_received:
                    print(content, end="", flush=True)
                    
            print("\n" + "="*75 + "\n")
            
            if not first_text_received:
                stop_event.set()
                try:
                    ticker_thread.join()
                except RuntimeError:
                    pass
                print(full_content)
                
        except Exception as e:
            stop_event.set()
            print(f"\n❌ Ollama 추론 중 네트워크 오류 발생: {e}\n")
            return StorylineAnalysisResult()

        print("[Step 3/3] 🎬 분석된 분기점 기반 썸네일 이미지 추출 및 타임코드 변환을 수행합니다.")

        raw_shifts = []
        try:
            raw_data = json.loads(full_content)
            if isinstance(raw_data, dict):
                if "shifts" in raw_data:
                    raw_shifts = raw_data["shifts"]
                else:
                    for val in raw_data.values():
                        if isinstance(val, list):
                            raw_shifts = val
                            break
            elif isinstance(raw_data, list):
                raw_shifts = raw_data
        except Exception as e:
            print(f"❌ JSON 파싱 에러 발생: {e}")
            raw_shifts = []

        base_dir = Path(output_dir) if output_dir else Path(file_path).resolve().parent
        story_output_dir = base_dir / Path(file_path).stem / "storylines"
        story_output_dir.mkdir(parents=True, exist_ok=True)

        import cv2
        cap = cv2.VideoCapture(file_path)
        fps = mxf_meta.fps or 25.0
        
        shifts: List[StorylineShift] = []
        total_shifts = len(raw_shifts)
        
        for i, item in enumerate(raw_shifts, start=1):
            try:
                start_sec = float(item.get("start_sec", 0.0))
            except (ValueError, TypeError):
                continue
                
            summary = item.get("summary", "요약 정보 없음")
            timecode = self._seconds_to_timecode(start_sec, fps, mxf_meta.start_timecode)
            
            print(f"       📸 맥락 썸네일 동기화 중 ({i}/{total_shifts}) -> [{timecode}] {summary}")
            
            safe_tc = timecode.replace(":", "-").replace(";", "-")
            thumb_filename = f"story_{i:02d}_{safe_tc}.png"
            thumb_path = os.path.join(story_output_dir, thumb_filename)
            
            target_frame = int(round(start_sec * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            
            if ret:
                cv2.imwrite(thumb_path, frame)
                
            actual_thumb = thumb_path if os.path.exists(thumb_path) else None
            
            shifts.append(
                StorylineShift(
                    start_sec=start_sec,
                    start_timecode=timecode,
                    summary=summary,
                    thumbnail_path=actual_thumb
                )
            )

        cap.release()
        print(f"✨ [성공] 총 {total_shifts}개의 주요 스토리라인 씬 요약 및 동기화 저장을 완료했습니다!\n")
        return StorylineAnalysisResult(shifts=shifts)

    @staticmethod
    def _seconds_to_timecode(seconds: float, fps: float, start_timecode: str) -> str:
        fps = fps or 25.0
        h, m, s, f = (int(x) for x in start_timecode.replace(";", ":").split(":"))
        start_frames = int(((h * 3600 + m * 60 + s) * fps) + f)
        elapsed_frames = int(round(seconds * fps))
        total_frames = start_frames + elapsed_frames
        total_seconds, ff = divmod(total_frames, int(round(fps)))
        hh, rem = divmod(total_seconds, 3600)
        mm, ss = divmod(rem, 60)
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"