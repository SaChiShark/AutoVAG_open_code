import os
import re
from datetime import timedelta
import subprocess
import tqdm
import psutil
import sys
import multiprocessing
def parse_srt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    pattern = r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})"
    matches = re.findall(pattern, content)

    time_ranges = []
    for start, end in matches:
        start_time = parse_time(start)
        end_time = parse_time(end)
        middle_time = start_time + (end_time - start_time) / 2
        time_ranges.append(middle_time)

    return time_ranges

def parse_time(time_str):
    h, m, s, ms = map(int, re.split('[:,]', time_str))
    return timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)

def take_screenshot(video_path, output_dir, time_points):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if len(os.listdir(output_dir)) == len(time_points):
        return
    for i, time_point in enumerate(tqdm.tqdm(time_points, desc="Taking screenshots", leave=False)):
        output_file = os.path.join(output_dir, f"screenshot_{i + 1}.jpg")
        if os.path.exists(output_file):
            continue
        timestamp = str(time_point).split('.')[0]
        command = [
            "ffmpeg",
            "-y",
            "-ss", timestamp,
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            output_file
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def process_videos_and_srt(video_path, srt_path, output_path):
    time_points = parse_srt(srt_path)
    take_screenshot(video_path, output_path, time_points)

def check_disk_space(disk="/"):
    usage = psutil.disk_usage(disk)
    free_space_gb = usage.free / (1024 ** 2)
    return free_space_gb

def process_video_task(task):
    course = task['course']
    video = task['video']
    video_path = task['video_path']
    srt_path = task['srt_path']
    output_path = task['output_path']
    
    print(f"[{course}] Processing video: {video}")
    process_videos_and_srt(video_path, srt_path, output_path)
    return (course, video)

if __name__ == '__main__':
    BASE_DIR = '../../data/courses'
    tasks = []
    for course in sorted(os.listdir(BASE_DIR)):
        course_path = os.path.join(BASE_DIR, course)
        srts_dir = os.path.join(course_path, 'srts')
        videos_dir = os.path.join(course_path, 'videoes')
        screenshot_dir = os.path.join(course_path, 'screenshots')
        
        if not (os.path.isdir(videos_dir) and os.path.isdir(srts_dir)):
            continue
        
        os.makedirs(screenshot_dir, exist_ok=True)
        srts = os.listdir(srts_dir)
        for video in sorted(os.listdir(videos_dir)):
            video_id = video.split('.')[0]
            
            video_path = os.path.join(videos_dir, video)
            if f'{video_id}.srt' in srts:
                srt_path = os.path.join(srts_dir, f'{video_id}.srt')
            else:
                continue
            
            output_path = os.path.join(screenshot_dir, video_id)
            os.makedirs(output_path, exist_ok=True)
            
            tasks.append({
                'course': course,
                'video': video,
                'video_path': video_path,
                'srt_path': srt_path,
                'output_path': output_path
            })
    
    pool = multiprocessing.Pool(processes=14)
    
    try:
        for result in tqdm.tqdm(pool.imap_unordered(process_video_task, tasks), total=len(tasks), desc="Overall progress"):
            pass

        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
