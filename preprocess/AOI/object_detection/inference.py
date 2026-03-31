import argparse

import cv2
import shutil
from ditod import add_vit_config
from tqdm import tqdm
import torch
import os
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer,_create_text_labels
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
import json
import numpy as np
import multiprocessing
def is_inside(boxA, boxB):
    # 检查boxA是否在boxB内
    Ax1, Ay1, Ax2, Ay2 = boxA
    Bx1, By1, Bx2, By2 = boxB
    return (Ax1 >= Bx1) and (Ay1 >= By1) and (Ax2 <= Bx2) and (Ay2 <= By2)
def calculate_intersection_area(boxA, boxB):
    # 计算交集区域的坐标
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    # 如果没有重叠区域，则返回0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area
def remove_contained_boxes_by_area(boxes, threshold=0.7):
    # 初始化结果列表
    filtered_boxes = []
    area = []

    for i in range(len(boxes)):
        boxA = boxes[i]
        area.append((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    # 遍历每个框框
    for i in range(len(boxes)):
        boxA = boxes[i]
        areaA = area[i]
        is_contained = False

        # 检查当前框框是否被其他框框包含
        for j in range(len(boxes)):
            if i != j :  # 不要与自身比较 and classes[i] == classes[j]
                boxB = boxes[j]
                intersection_area = calculate_intersection_area(boxA, boxB)
                
                # 计算被包含的比例
                if (intersection_area / areaA) >= threshold and areaA < area[j]:
                    is_contained = True
                    break

        # 如果没有被包含，添加到结果列表中
        if not is_contained:
            filtered_boxes.append(i)

    return np.array(filtered_boxes)
def draw_aoi(img,predictions,thing_classes,save_path,score_threshold = None):
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    labels = _create_text_labels(classes, scores,thing_classes)
    if score_threshold != None:
        top_id = np.where(scores.numpy() > score_threshold)[0].tolist()
        scores = torch.tensor(scores.numpy()[top_id])
        boxes.tensor = torch.tensor(boxes.tensor.numpy()[top_id])
        classes = [classes[ii] for ii in top_id]
        labels = [labels[ii] for ii in top_id]
    no_overlay = remove_contained_boxes_by_area(boxes.tensor.numpy())
    if len(no_overlay) != len(labels):          
        scores = torch.tensor(scores.numpy()[no_overlay])
        boxes.tensor = torch.tensor(boxes.tensor.numpy()[no_overlay])
        classes = [classes[ii] for ii in no_overlay]
        labels = [labels[ii] for ii in no_overlay]
    for i, boxe in enumerate(boxes.tensor):
        x1,y1,x2,y2 = round(boxe[0].item()),round(boxe[1].item()),round(boxe[2].item()),round(boxe[3].item())
        cv2.imwrite(save_path + '/img/{}.jpg'.format(i),img[y1:y2,x1:x2,:])
    predictions.pred_boxes = boxes
    predictions.scores = scores
    predictions.pred_classes = np.array(classes)
    save2json(predictions,img.shape[:2],save_path +'/aoi.json')
def worker(task_queue, event):
    """Worker 等待 event，然後開始從 queue 取出任務，當 queue 為空時結束"""
    while True:
        try:
            img,output,page_path = task_queue.get(timeout=2)  # 設置 timeout 避免無限等待
            process_output(img,output,page_path)
        except:
            if event.is_set() and task_queue.empty():  # 檢查 event 且 queue 是否空
                break

def main():
    global md,args
    parser = argparse.ArgumentParser(description="Detectron2 inference script")

    parser.add_argument(
        "--config-file",
        default="./cascade/cascade_dit_base.yaml",
        metavar="FILE",
        help="path to config file",
    )    

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default = ["MODEL.WEIGHTS",'./finetuned_DIT.pth'],
        nargs=argparse.REMAINDER,
    )

    
    
    args = parser.parse_args()
    args.save2json = False

    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    # Step 2: add model weights URL to config
    cfg.merge_from_list(args.opts)
    
    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
    
    # Step 4: define model
    #predictors = []
    #for i in range(4):
    #    predictors.append(DefaultPredictor(cfg))
    model = DefaultPredictor(cfg)
    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    if cfg.DATASETS.TEST[0]=='icdar2019_test':
        md.set(thing_classes=["table"])
    else:
        md.set(thing_classes=["image","math","plot","table","text","title"])
        
    
    base_path = '../../../data/courses'
    #process_list = []
    queue = multiprocessing.Queue()
    event = multiprocessing.Event()
    processes = []
    for i in range(16):
        p = multiprocessing.Process(target=worker, args=(queue, event))
        processes.append(p)
        p.start()
    for course in tqdm(os.listdir(base_path),desc='Courses: '):
        course_path = f'{base_path}/{course}'
        shutil.rmtree(f'{course_path}/AOIs', ignore_errors=True)
        os.makedirs(f'{course_path}/AOIs',exist_ok=True)
        for split_result in tqdm(os.listdir(f'{course_path}/split_results'),desc='Videos: '):
            video = split_result.split('.')[0]
            split_result = json.load(open(f'{course_path}/split_results/{split_result}'))
            os.makedirs(f'{course_path}/AOIs/{video}',exist_ok=True)
            need_postprocess = AOI_detect(course_path,video,split_result,model)
            for task in need_postprocess:
                #process_output(task[0],task[1],task[2])
                queue.put(task)
    print('Wait for postprocess')
    event.set()
    for p in processes:
        p.join()

        

def process_output(img,output,page_path):       
    global  md, args
    os.makedirs(page_path,exist_ok=True)
    os.makedirs(f'{page_path}/img',exist_ok=True)
    v = Visualizer(img[:, :, ::-1],
                    md,
                    scale=1.0,
                    instance_mode=ColorMode.SEGMENTATION)
    draw_aoi(
        img=img,
        predictions=output,
        thing_classes=["image","math","plot","table","text","title"],
        save_path=page_path,
        score_threshold=0.5)
    result = v.draw_instance_predictions(output,score_threshold = 0.5, remote_overlay= True)
    result_image = result.get_image()[:, :, ::-1]
    # step 6: save
    cv2.imwrite(f'{page_path}/AOI_no_overlay.jpg', result_image)
    result = v.draw_instance_predictions(output,score_threshold = 0.5, remote_overlay= False)
    result_image = result.get_image()[:, :, ::-1]
    cv2.imwrite(f'{page_path}/AOI.jpg', result_image)
    if args.save2json:
        save2json(output, img.shape[:2],f'{page_path}/AOI.json')
def AOI_detect(course_path,video,split_result,model):
    global  md, args
    screenshot_path = f'{course_path}/screenshots/{video}'
    AOI_path = f'{course_path}/AOIs/{video}'
    need_postprocess = []
    page_paths = []
    for key in tqdm(split_result.keys(),desc='Pages: '):
        if os.path.exists(f'{AOI_path}/page_{key}'):
            continue
        page_path = f'{AOI_path}/page_{key}'
        img = cv2.imread(f'{screenshot_path}/screenshot_{split_result[key]}.jpg')
        page_paths.append(page_path)
        # Step 5: run inference
        with torch.no_grad():
            output = model(img)
        need_postprocess.append([img,output['instances'].to('cpu'),page_path])
    return need_postprocess

def save2json(predictions,img_shape,output_file_name):
    import json
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    
    boxes = boxes.tensor
    output = {
        'AOI':[],
        'shape':img_shape
    }
    for i in range(len(classes)):
        output["AOI"].append({
            'boxes':boxes[i].numpy().tolist(),
            'score':scores[i].item(),
            'class':classes[i]
        })
    with open(output_file_name,'w+') as f:
        json.dump(output,f, indent=4)
if __name__ == '__main__':
    main()