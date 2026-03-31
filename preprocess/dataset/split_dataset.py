import json
import tqdm
import os
with open('../../datasets/base_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)
    
for course in tqdm.tqdm(dataset.keys()):
    for video in tqdm.tqdm(dataset[course].keys()):
        os.makedirs(f'../../datasets//base_dataset/{course}/',exist_ok=True)
        #if not os.path.exists(f'/home/mvnl/code/cool/dataset/base_dataset/{course}/{video}.json'):
        with open(f'../../datasets//base_dataset/{course}/{video}.json', 'w', encoding='utf-8') as f:
            json.dump(dataset[course][video],f,indent=4,ensure_ascii=False)