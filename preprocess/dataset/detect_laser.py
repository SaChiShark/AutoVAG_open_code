from ultralytics import YOLO
import tqdm
import cv2
import json
import os
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

# 加载 YOLO 模型
model = YOLO('laser_detect_model.pt', verbose=True).to('cuda')

# 设定路径
base_path = '../../'


batch_size = 30
batch_image = []
batch_arg = []

# 定义多线程读取图片的函数
def load_image(screenshot_path):
    if os.path.exists(screenshot_path):
        return cv2.imread(screenshot_path)
    return None
# 遍历数据集
for course in tqdm.tqdm(os.listdir(f'{base_path}/datasets/opencode/base_dataset'), desc='Course'):
    dataset_path = f'{base_path}/datasets/opencode/base_dataset/{course}'
    for video in tqdm.tqdm(os.listdir(dataset_path), desc='Video'):
        with open(f'{dataset_path}/{video}', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        video = video.split('.')[0]
        now_page = 1
        with open(f'{base_path}/data/courses/{course}/split_results/{video}.json', 'r', encoding='utf-8') as f:
            split_result = json.load(f)
        
        # 预先过滤 context_list 中无效的项
        valid_contexts = []
        for context in dataset['context_list']:
            screenshot_id = context['screenshot_id']
            if screenshot_id > split_result[str(now_page)]:
                now_page += 1
            screenshot_path = f'{base_path}/data/courses/{course}/screenshots/{video}/screenshot_{screenshot_id}.jpg'
            if os.path.exists(screenshot_path):
                context['page'] = now_page
                valid_contexts.append(context)
            # 不存在图片的 context 将被跳过
        # 更新数据集中的 context_list
        dataset['context_list'] = valid_contexts

        # 创建线程池并并行加载图片
        image_futures = {}
        change = False
        with ProcessPoolExecutor(max_workers=20) as executor:
            for i, context in enumerate(dataset['context_list']):
                if  'yolo_detect_result' in context.keys() and  'conf' in context['yolo_detect_result'].keys():
                    continue
                change = True
                screenshot_id = context['screenshot_id']
                screenshot_path = f'{base_path}/data/courses/{course}/screenshots/{video}/screenshot_{screenshot_id}.jpg'
                future = executor.submit(load_image, screenshot_path)
                image_futures[future] = (course, video, i)
            
            # 收集结果
            for future in as_completed(image_futures):
                course, video, i = image_futures[future]
                img = future.result()
                if img is not None:
                    batch_image.append(img)
                    batch_arg.append([course, video, i])
                
                # 达到 batch_size 时进行处理
                if len(batch_image) >= batch_size:
                    results = model(batch_image)
                    for arg, result in zip(batch_arg, results):
                        course, video, i = arg
                        dataset['context_list'][i]['yolo_detect_result'] = {
                            'cls': result.boxes.cls.cpu().numpy().tolist(),
                            'xywh': result.boxes.xywh.cpu().numpy().tolist(),
                            'conf': result.boxes.conf.cpu().numpy().tolist()
                        }
                    # 清空批次数据并释放 GPU 内存
                    batch_image.clear()
                    batch_arg.clear()

        # 处理剩余未完成的 batch
        if batch_image:
            results = model(batch_image)
            for arg, result in zip(batch_arg, results):
                course, video, i = arg
                dataset['context_list'][i]['yolo_detect_result'] = {
                    'cls': result.boxes.cls.cpu().numpy().tolist(),
                    'xywh': result.boxes.xywh.cpu().numpy().tolist(),
                    'conf': result.boxes.conf.cpu().numpy().tolist()
                }
            batch_image.clear()
            batch_arg.clear()
            del results
        
        if change:
            with open(f'{dataset_path}/{video}.json', 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)

