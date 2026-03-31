import pysrt
import os
import json
import tqdm
import multiprocessing

#做出基本的資料集，需要補上每個題目的ans

# 安全的時間計算函數
def safe_shift(time, seconds):
    new_ordinal = time.ordinal + (seconds * 1000)
    if new_ordinal < 0:
        return pysrt.SubRipTime(0, 0, 0)  # 返回 "00:00:00,000"
    return pysrt.SubRipTime.from_ordinal(new_ordinal)

# 計算中間時間
def get_midpoint(start, end):
    mid_ordinal = (start.ordinal + end.ordinal) // 2
    return pysrt.SubRipTime.from_ordinal(mid_ordinal)

def generate_legal_page(path):
    ans = []
    for page in os.listdir(path):
        ans.append(int(page[:-4]))
    return sorted(ans)

def find(subs,screenshot_path):
    result = {
        'texts':[]
    }
    context_list = []
    # 遍歷每段字幕
    for i, sub in enumerate(subs):
        # 當前字幕
        result['texts'].append(sub.text)
        if not os.path.exists(f'{screenshot_path}/screenshot_{i+1}.jpg'):
            continue

        # 計算當前字幕的中間點
        current_midpoint = get_midpoint(sub.start, sub.end)
        
        # 計算前後 15 秒的時間範圍
        start_range = safe_shift(current_midpoint, -15)
        end_range = safe_shift(current_midpoint, 15)

        # 蒐集前後文
        previous_context = []
        next_context = []

        # 找到前文字幕
        for j in range(i - 1, -1, -1):  # 倒序找前面的字幕
            mid_point = get_midpoint(subs[j].start, subs[j].end)  # 計算前文字幕的中間點
            if start_range <= mid_point <= end_range:  # 判斷是否在範圍內
                previous_context.insert(0, {
                    'text_id':j,
                    'time': str(current_midpoint - mid_point)
                    })
            else:
                break  # 如果超出範圍，停止搜尋

        # 找到後文字幕
        for j in range(i + 1, len(subs)):  # 正序找後面的字幕
            mid_point = get_midpoint(subs[j].start, subs[j].end)  # 計算後文字幕的中間點
            if start_range <= mid_point <= end_range:  # 判斷是否在範圍內
                next_context.append({
                    'text_id':j,
                    'time': str(mid_point - current_midpoint)
                    })
            else:
                break  # 如果超出範圍，停止搜尋

        # 整理結果
        context_list.append({
            'screenshot_id':i+1,
            'current': i,
            'previous': previous_context,
            'next': next_context,
            'current_midpoint':str(current_midpoint)
        })
    result['context_list'] = context_list
    return result

# 單獨處理一個 srt 檔案的函數（每個進程運行這個函數）
def process_srt(args):
    course, srt, base_path = args
    course_path = f'{base_path}/{course}'
    video = srt.split('.')[0]

    screenshot_path = f'{course_path}/screenshots/{video}'
    srt_path = f'{course_path}/srts/{srt}'

    if not os.path.exists(screenshot_path):
        return course, video, None  # 沒有截圖，跳過

    subs = pysrt.open(srt_path)
    result = find(subs, screenshot_path)
    
    return course, video, result

# 主程式
if __name__ == "__main__":
    base_path = '../../data/courses'
    dataset = {}
    
    tasks = []
    for course in sorted(os.listdir(base_path)):
        course_path = f'{base_path}/{course}/srts'
        if not os.path.exists(course_path):
            continue
        
        for srt in sorted(os.listdir(course_path)):
            if  '.srt' in srt:
                tasks.append((course, srt, base_path))  # 傳遞參數
            else:
                print(srt)


    # 使用 multiprocessing Pool 並行處理
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm.tqdm(pool.imap(process_srt, tasks), total=len(tasks), desc="Processing SRTs"))

    # 整理結果
    for course, video, result in tqdm.tqdm(results):
        if result is not None:
            if course not in dataset:
                dataset[course] = {}
            dataset[course][video] = result

    # 寫入 JSON 文件
    os.makedirs('../../datasets', exist_ok=True)
    os.makedirs('../../datasets/base_dataset', exist_ok=True)
    for course in tqdm.tqdm(dataset.keys()):
        for video in tqdm.tqdm(dataset[course].keys()):
            os.makedirs(f'../../datasets//base_dataset/{course}/',exist_ok=True)
            #if not os.path.exists(f'/home/mvnl/code/cool/dataset/base_dataset/{course}/{video}.json'):
            with open(f'../../datasets//base_dataset/{course}/{video}.json', 'w', encoding='utf-8') as f:
                json.dump(dataset[course][video],f,indent=4,ensure_ascii=False)