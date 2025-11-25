import json
import os
import pandas as pd
import subprocess
import imageio_ffmpeg

# ================= [설정값] =================
VIDEO_PATH = "test2.mp4"
JSON_PATH = "shots2.json"
OUTPUT_DIR = "segments"
CSV_PATH = "dataset2.csv"

# ★ 여기를 수정하세요! (시작하고 싶은 번호)
START_NUMBER = 129
# ===========================================

def step2_cut():
    if not os.path.exists(JSON_PATH):
        print("❌ json 파일이 없습니다. 1단계를 먼저 실행하세요!")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        shots = json.load(f)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    data_for_csv = []

    print(f"🎬 [2단계] 영상 자르기 & 데이터셋 생성 (총 {len(shots)}개)")
    print(f"👉 파일 번호는 {START_NUMBER}번 부터 시작합니다.")
    
    # [수정된 부분] enumerate(shots, start=START_NUMBER) 
    # 이렇게 하면 i가 0이 아니라 130부터 시작합니다.
    for i, shot in enumerate(shots, start=START_NUMBER):
        start = shot['start']
        duration = shot['end'] - start
        label = shot['label']

        # i가 130부터 들어오므로 segment_0130.mp4 가 됩니다.
        filename = f"segment_{i:04d}.mp4"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # 파일 있으면 삭제 (덮어쓰기)
        if os.path.exists(output_path): os.remove(output_path)

        cmd = [
            ffmpeg_exe, "-y",
            "-i", VIDEO_PATH,      
            "-ss", str(start),     
            "-t", str(duration),   
            "-c:v", "libx264",     
            "-c:a", "aac",         
            "-preset", "fast",     
            "-crf", "23",          
            "-loglevel", "error",  
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            data_for_csv.append({
                "video_path": f"segments/{filename}",
                "label": label
            })
            
            if label == 'replay':
                print(f"   🎥 {filename} 저장 (★REPLAY 감지됨!)")
            elif i % 10 == 0:
                print(f"   🎥 {filename} 저장 완료")

        except:
            print(f"   ❌ {filename} 실패")

    # CSV 파일 저장
    df = pd.DataFrame(data_for_csv)
    # [꿀팁] mode='a' (append)를 쓰면 기존 CSV 밑에 이어붙일 수 있지만,
    # 지금은 헷갈릴 수 있으니 그냥 새로 만들고 나중에 엑셀에서 합치시는 걸 추천합니다.
    df.to_csv(CSV_PATH, index=False)
    
    print("\n🎉 모든 작업 완료!")
    print(f"📂 영상 폴더: {OUTPUT_DIR}")
    print(f"📝 라벨 파일: {CSV_PATH}")

if __name__ == "__main__":
    step2_cut()