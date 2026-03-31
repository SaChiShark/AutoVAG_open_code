import os
import csv
from faster_whisper import WhisperModel,available_models
from tqdm import tqdm

BASE_DIR = '../../data/courses'
MODEL_SIZE = "SoybeanMilk/faster-whisper-Breeze-ASR-25"
DEVICE = "cuda"
COMPUTE_TYPE = "float16" 
NUM_WORKERS = 4        

def format_timestamp(seconds):
    millisec = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millisec:03}"

def save_to_srt(segments, srt_filename):
    os.makedirs(os.path.dirname(srt_filename), exist_ok=True)
    with open(srt_filename, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            text = segment.text.strip()
            srt_file.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

# --- 主程式 ---
if __name__ == "__main__":
    # 1. 載入 Faster-Whisper 模型
    # cpu_threads 指定用於音檔解碼的 CPU 核心數
    print(f"正在載入 Faster-Whisper 模型: {MODEL_SIZE}...")
    model = WhisperModel(
        MODEL_SIZE, 
        device=DEVICE, 
        compute_type=COMPUTE_TYPE,
        num_workers=NUM_WORKERS
    )

    # 2. 讀取任務清單
    tasks = []
    with open("missing_srt.csv", mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # 跳過表頭
        for row in reader:
            tasks.append(row)

    # 3. 執行轉錄 (Faster-Whisper 的 transcribe 本身就非常快)
    progress_bar = tqdm(total=len(tasks), desc="轉錄進度")
    
    for row in tasks:
        try:
            course_id, video_id, audio_name = row[0], row[1], row[2]
            audio_path = os.path.join(BASE_DIR, course_id, 'audios', audio_name.split('.')[0] + '.mp3')
            output_path = os.path.join(BASE_DIR, course_id, 'srts_', f'{video_id}_normal_whisper.srt')

            segments, info = model.transcribe(
                audio_path, 
                beam_size=5,
                initial_prompt="這是一個課程影片。請用繁體回應",
                vad_filter=False
            )

            # 執行轉錄並保存
            save_to_srt(segments, output_path)
            
        except Exception as e:
            print(f"\n處理 {audio_name} 時出錯: {e}")
        finally:
            progress_bar.update(1)

    progress_bar.close()
    print("所有任務完成！")