import whisper
import threading
import queue
import csv
from tqdm import tqdm
import os
def format_timestamp(seconds):
    millisec = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millisec:03}"

def whisper_to_srt(whisper_output, srt_filename="output.srt"):
    with open(srt_filename, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(whisper_output, start=1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()

            srt_file.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

    print(f"SRT 字幕已保存至 {srt_filename}")

def worker(bar):
    while not task_queue.empty():
        row = task_queue.get()
        model = whisper.load_model("large-v3-turbo")
        row[2] = row[2].split('.')[0] + '.mp3'
        result = model.transcribe(f'{base_dir}/{row[0]}/audios/{row[2]}',initial_prompt = "這是一個課程影片。請用繁體回應")
        os.makedirs(f'{base_dir}/{row[0]}/srts',exist_ok=True)
        whisper_to_srt(result['segments'],f'{base_dir}/{row[0]}/srts/{row[1]}_normal_whisper.srt')
        with lock:
            bar.update(1)
        task_queue.task_done()

base_dir = '../../data/courses'
task_queue = queue.Queue()
lock = threading.Lock()

with open("missing_srt.csv", mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)
    flag = False
    count = 0
    for row in reader:
        if not flag:
            flag = True
            continue
        task_queue.put(row)
        count += 1
        
progress_bar = tqdm(total=count, position=0, leave=True)
num_threads = 4
threads = []


for i in range(num_threads):
    thread = threading.Thread(target=worker,args=(progress_bar,))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()
