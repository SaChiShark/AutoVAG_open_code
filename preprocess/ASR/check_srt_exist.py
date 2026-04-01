import os
import csv

base_dir = '../../data/courses'
count = 0
missing_count = 0
missing = [
    ['course','video_id','video_name']
]
for course in os.listdir(base_dir):
    if not os.path.exists(f'{base_dir}/{course}/srts'):
        os.mkdir(f'{base_dir}/{course}/srts')
    srts = os.listdir(f'{base_dir}/{course}/srts')
    for video in os.listdir(f'{base_dir}/{course}/videoes'):
        count += 1
        video_id = video.split('.')[0]
        if  not f'{video_id}.srt' in srts:
            missing_count += 1
            missing.append([course,video_id,video])
            
with open("missing_srt.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(missing)