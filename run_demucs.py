# D:\mxf_pipeline\run_demucs.py
import sys
import torchaudio
import soundfile as sf
from demucs.__main__ import main as demucs_main

# 🎯 [핵심 패치] 에러를 일으키는 토치오디오의 저장 기능을 무력화하고
# 가장 안전한 soundfile 백엔드로 강제 교체합니다.
def windows_safe_save(path, src, sample_rate, bits_per_sample=16, **kwargs):
    # 오디오 텐서 (Channels, Frames) -> soundfile 규격 (Frames, Channels)로 회전
    data = src.cpu().detach().numpy().T
    subtype = 'PCM_16' if bits_per_sample == 16 else 'PCM_24'
    sf.write(path, data, sample_rate, subtype=subtype)

# 원래 기능을 우리가 만든 안전한 가드로 덮어쓰기
torchaudio.save = windows_safe_save

if __name__ == "__main__":
    # 기존에 터미널에 입력하던 타겟 인자들을 배열로 주입
    sys.argv = [
        "demucs", 
        "-d", "cuda:0", 
        "--two-stems", "vocals",
        "D:\\results\\지옥에서온판사 12회 본방1_audio.wav", 
        "-o", "D:\\results"
    ]
    print("🚀 몽키 패치 가드 가동 -> Demucs 고속 보컬 분리를 시작합니다...")
    demucs_main()
    print("✨ [성공] 에러 없이 vocals.wav 파일이 정상 저장되었습니다!")