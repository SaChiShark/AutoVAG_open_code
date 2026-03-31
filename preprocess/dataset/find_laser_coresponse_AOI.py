import json 
import os
import math
import tqdm
def compute_distance(laser_point,boxes_center):
    return math.sqrt((laser_point[0] - boxes_center[0])**2 + (laser_point[1] - boxes_center[1])**2)
def is_inside(laser_point, box):
    # 检查boxA是否在boxB内
    x1, y1, x2, y2 = box
    x = (x2 - x1) * 0.2
    y = (y2 - y1) * 0.2
    x1 -= x
    x2 += x
    y1 -= y
    y2 += y
    return (laser_point[0] >= x1) and (laser_point[1] >= y1) and (laser_point[0] <= x2) and (laser_point[1] <= y2)
def find_nearest_AOI(laser_point,AOIs):
    indexes = []
    distances = []
    for i,aoi in enumerate(AOIs):
        box = aoi['boxes']
        aoi_center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        if is_inside(laser_point,box):
            distances.append(compute_distance(laser_point, aoi_center))
            indexes.append(i)
    if len(indexes) == 0:
        return -1
    else:
        min_index, min_distance = min(enumerate(distances), key=lambda x: x[1])
        return indexes[min_index]

def read_AOIs(page,course,video):
    with open(f'{base_path}/data/courses/{course}/AOIs/{video}/page_{page}/aoi.json') as f:
        AOIs = json.load(f)['AOI']
    return AOIs
base_path = '../..'
dataset_path = f'{base_path}/datasets/base_dataset'

for course in tqdm.tqdm(os.listdir(dataset_path),desc='Course'):
    course_path = f'{dataset_path}/{course}'
    for video in tqdm.tqdm(os.listdir(course_path),desc='Video'):
        with open(f'{course_path}/{video}',encoding='utf-8') as f:
            dataset = json.load(f)
        now_page = 1
        AOIs = read_AOIs(now_page,course,video.replace('.json',''))
        for i in range(len(dataset['context_list'])):
            page = dataset['context_list'][i]['page']
            if 'ans' in dataset['context_list'][i].keys():
                continue
            #找到所有AOI
            if page != now_page:
                AOIs = read_AOIs(page,course,video.replace('.json',''))
                now_page = page
            yolo_detect_result = dataset['context_list'][i]['yolo_detect_result']
            if len(yolo_detect_result['cls']) == 0:
                dataset['context_list'][i]['ans'] = -1
                continue
            target_laser_loc = None
            conf_now = 0
            #找到雷射點位置
            for cls,xywh,conf in zip(yolo_detect_result['cls'],yolo_detect_result['xywh'],yolo_detect_result['conf']):
               if conf > conf_now:
                    target_laser_loc = xywh
            if target_laser_loc is None:
                dataset['context_list'][i]['ans'] = -1
                continue
            dataset['context_list'][i]['ans'] = find_nearest_AOI((target_laser_loc[0] + target_laser_loc[2] / 2,target_laser_loc[1] + target_laser_loc[3]),AOIs)
            
        with open(f'{course_path}/{video}','w',encoding='utf-8') as f:
            json.dump(dataset,f,ensure_ascii=False,indent=4)