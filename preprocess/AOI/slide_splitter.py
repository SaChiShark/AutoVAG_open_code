import os
import json
import tqdm
import imagehash
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from multiprocessing import Pool
import argparse
def extract_number(file_name):
    # 從檔案名稱中提取數字部分，假設格式為 screenshot_{number}.jpg
    base_name = os.path.basename(file_name)
    number_str = base_name.split('_')[1].split('.')[0]
    return int(number_str)

def my_ssim(i1, i2, win_size, data_range, channel_axis):
    return ssim(i1, i2, win_size=win_size, data_range=data_range, channel_axis=channel_axis)

def compare_images(img1, img2):
    """
    比較兩張圖片的感知哈希，若差異大於等於 threshold，則返回 True
    """
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)
    difference = abs(hash1 - hash2)
    return difference

def process_video(course_path, video):
    """
    處理單個 video 下的所有截圖，判斷頁面切換並輸出比對結果
    """
    # 定義結果儲存路徑
    split_results_dir = os.path.join(course_path, "split_results")
    compare_results_dir = os.path.join(course_path, "compare_results")
    split_results_file = os.path.join(split_results_dir, f"{video}.json")
    compare_results_file = os.path.join(compare_results_dir, f"{video}.json")
    # 組成影片截圖的資料夾路徑
    screen_shot_path = os.path.join(course_path, "screenshots", video)
    if not os.path.exists(screen_shot_path):
        print(f"Screenshot directory does not exist: {screen_shot_path}")
        return

    # 依照檔名中數字順序排序
    screenshot_files = sorted(os.listdir(screen_shot_path), key=extract_number)
    if not screenshot_files:
        print(f"No screenshots found in {screen_shot_path}")
        return

    screenshot_last = None
    page_counter = 1
    split_result = {}
    compare_result = {}

    # 使用 tqdm 顯示內部 screenshot 處理進度
    for idx, screenshot in enumerate(tqdm.tqdm(screenshot_files, desc=f"Processing {video}", leave=True, dynamic_ncols=True)):
        screenshot_path = os.path.join(screen_shot_path, screenshot)
        if idx == 0:
            screenshot_last = Image.open(screenshot_path)
        else:
            screenshot_id = extract_number(screenshot)
            screenshot_now = Image.open(screenshot_path)
            if np.array(screenshot_now).shape != np.array(screenshot_last).shape:
                compare_result[screenshot_id] = {
                    'pash': -1,
                    'ssim': -1
                }
                screenshot_last = screenshot_now
                continue
            # 比較感知哈希
            phash_result = compare_images(screenshot_now, screenshot_last)
            # 比較 SSIM
            ssim_result = my_ssim(np.array(screenshot_now), np.array(screenshot_last), win_size=51, data_range=255, channel_axis=2)
            if ssim_result == -1:
                continue
            compare_result[screenshot_id] = {
                'pash': int(phash_result),
                'ssim': ssim_result
            }
            # 若感知哈希判定有變化且 SSIM 小於 0.5，認定為換頁
            if (phash_result < 20  and ssim_result < 0.7) or phash_result >= 20:
                split_result[page_counter] = extract_number(screenshot_files[idx-1])
                page_counter += 1
            screenshot_last = screenshot_now

    # 將最後一個頁面的結尾設定為最後一張截圖的 id
    last_screenshot_id = extract_number(screenshot_files[-1])
    split_result[page_counter] = last_screenshot_id

    # 寫入結果到 JSON 檔
    os.makedirs(split_results_dir, exist_ok=True)
    with open(split_results_file, 'w') as f:
        json.dump(split_result, f, indent=4)
    os.makedirs(compare_results_dir, exist_ok=True)
    
    with open(compare_results_file, 'w') as f:
        json.dump(compare_result, f, indent=4)


if __name__ == '__main__':
    base_path = '../../data/courses'
    tasks = []
    # 遍歷所有課程
    for course in sorted(os.listdir(base_path)):
        course_path = os.path.join(base_path, course)
        screenshot_dir = os.path.join(course_path, "screenshots")
        if not os.path.exists(screenshot_dir):
            continue
        # 遍歷該課程下所有 video（資料夾名稱）
        for video in os.listdir(screenshot_dir):
            #process_video(course_path, video)
            tasks.append((course_path, video,))
    # 使用 multiprocessing Pool 平行處理各個 video
    with Pool() as pool:
        # 外層進度條顯示所有 video 的處理進度
        for _ in pool.starmap(process_video, tasks):
            pass
