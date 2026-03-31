import json
import os
import random
import tqdm
def make_subtitle(previous,current,next_s,texts):
    subtitle = ''
    for i,p in enumerate(previous):
        subtitle_id = len(previous)-i
        temp_s = texts[p['text_id']]
        subtitle += f'[-{subtitle_id}]{temp_s}'
    subtitle += f'[0]{texts[current]}'
    for i,n in enumerate(next_s):
        subtitle_id = i + 1
        temp_s = texts[n['text_id']]
        subtitle += f'[{subtitle_id}]{temp_s}'
    return subtitle

def make_human(subtitle,aoi_describes):
    #base = f"請問以下字幕(每段字幕前面都會有[number]，代表講者的說話順序其中[0]為講者當前說的話):\"{subtitle}\"最可能對應的是哪段敘述:"
    base = f'### 字幕格式：\n\
            - 每段字幕的 `[number]` 表示說話順序：\n\
            - `[0]` 代表當前講話內容。\n\
            - `[1], [2], ...` 代表未來的內容，數值越大代表越晚說。\n\
            - `[-1], [-2], ...` 代表過去的內容，數值越小代表時間早說\n\
            ### 指令\n \
            請問以下字幕:\"{subtitle}\"最可能對應的是哪段敘述:?\n\
            ###敘述\n'
    for i,aoi_describe in enumerate(aoi_describes):
        base += f"\n{i}:\"{aoi_describe}\""
    return base

def make_line(human, ans,course,video,aoi_count):
    conversation = [
            {"from": "human", "value": human},
            {"from": "gpt", "value": f"最可能對應的是:\"{ans}\""}
        ]
    return {
            "conversations": conversation,
            "system": "You are a highly accurate natural language processing model specializing in semantic similarity tasks. Your task is to analyze a given subtitle and compare it with a list of narrative descriptions. Based on the semantic meaning of the subtitle and the narratives, identify which narrative is the closest match to the subtitle.Return the index of the closest narrative. Focus on the meaning and context of the text, rather than exact word matches.",  # 系统提示词，可选填
            "course":course,
            "video":video,
            'aoi_count':aoi_count
        }


base_path = '../..'
token_len = []
train = []
valid = []
setting = {
    'train':['ML2021','Gai','ADL'],
    'valid':['ML2022']
}
count = 0
token_counts = 0
for course_count,course in enumerate(os.listdir(f'{base_path}/datasets/base_dataset')):
    for video in tqdm.tqdm(os.listdir(f'{base_path}/datasets/base_dataset/{course}')):
        video = video.replace('.json','')
        with open(f'{base_path}/datasets/base_dataset/{course}/{video}.json',encoding='utf-8') as f:
            dataset = json.load(f)
            
        for context in dataset['context_list']:
            if context['ans'] == -1:
                continue
            subtitle = make_subtitle(context['previous'],context['current'],context['next'],dataset['texts'])
            page = context['page']
            if os.path.exists(f'{base_path}/data/courses/{course}/AOIs/{video}/page_{page}/describes_gpt'):
                key = 'text_1'
                descibe_path = f'{base_path}/data/courses/{course}/AOIs/{video}/page_{page}/describes_gpt'
            if os.path.exists(f'{base_path}/data/courses/{course}/AOIs/{video}/page_{page}/describes_phi4'):
                key = 'Chinese'
                descibe_path = f'{base_path}/data/courses/{course}/AOIs/{video}/page_{page}/describes_phi4'
            else:
                descibe_path = f'{base_path}/data/courses/{course}/AOIs/{video}/page_{page}/describes_pixtral_12B_8bit'
                key = 'Chinese_llama_8b'
            aoi_describes = [0] * len(os.listdir(descibe_path))
            
            for aoi_describe_path in os.listdir(descibe_path):
                with open(f'{descibe_path}/{aoi_describe_path}',encoding='utf-8') as f:
                    aoi_describe = json.load(f)[key]
                    aoi_describe = aoi_describe.replace('Describe :','')
                    aoi_describe_page = int(aoi_describe_path.split('.')[0])
                    if not '0.json' in os.listdir(descibe_path):
                        aoi_describes[aoi_describe_page - 1] = aoi_describe
                    else:
                        aoi_describes[aoi_describe_page] = aoi_describe
            ans_text = aoi_describes[context['ans']]
            if not (course in setting['valid']):
                random.shuffle(aoi_describes)
            ans = aoi_describes.index(ans_text)
            
            human = make_human(subtitle,aoi_describes)
            count += 1
            line = make_line(human,ans,course,video,len(aoi_describes))
            if course in setting['valid']:
                valid.append(line)
            else:
                train.append(line)
with open(f'{base_path}/datasets/train.json','w',encoding='utf-8') as f:
    json.dump(train,f,ensure_ascii=False,indent=2)
with open(f'{base_path}/datasets/valid.json','w',encoding='utf-8') as f:
    json.dump(valid,f,ensure_ascii=False,indent=2)