import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import threading
import queue
import tqdm
import json
model_id = 'microsoft/Phi-4-multimodal-instruct'
model_path = 'finetuned_model'
kwargs = {}
kwargs['torch_dtype'] = torch.bfloat16

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
print(processor.tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype='auto'
).cuda()
print("model.config._attn_implementation:", model.config._attn_implementation)

generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'
# 假設這是你的 inference 函數
def inference(AOI_path,AOI_number):
    chat = [
        {
        'role': 'user',
        'content': f"<|image_1|>This is a course slide imgae that contains  Areas of Interest (AOI) and I will give you later. Please follow the instructions below when responding:\n\n. When you receive an image containing an AOI, carefully observe and describe the content and characteristics of that area.\n Your description should be detailed and clear, highlighting the key elements in the image.",
        },
        {
            'role': 'assistant',
            'content': "Okay, I received the lecture slides. Please provide image of area of interest (AOI) and I will describe it as you request.",
        },
        {'role': 'user', 'content': f'<|image_2|>Please describe this AOI in detail.'},
    ]
    #screenshot_path = AOI_path.replace('AOIs_all_screenshot','screenshot')
    #original = Image.open(f'{screenshot_path}.jpg')
    original = Image.open(f'{AOI_path}/original.jpg')
    image = Image.open(f'{AOI_path}/img/{AOI_number}.jpg')
    prompt = processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    # need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
    if prompt.endswith('<|endoftext|>'):
        prompt = prompt.rstrip('<|endoftext|>')
    #print(prompt)
    inputs = processor(prompt, [original,image], return_tensors='pt').to('cuda:0')
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        generation_config=generation_config,
        num_logits_to_keep=0
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response

# 寫入 worker，持續從 queue 中取得任務並寫入檔案
def writer_worker(response_queue):
    while True:
        # 從 queue 中取得項目 (blocking)
        item = response_queue.get()
        # 當收到 sentinel (None) 時，結束此 thread
        if item is None:
            response_queue.task_done()
            break
        # item 為 (dir_path, aoi_name, response) 的 tuple
        dir_path, aoi_name, response = item
        output_file = f'{dir_path}/describes_phi4/{aoi_name}.json'
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'Chinese':response
                },f,indent=4,ensure_ascii=False)
        except Exception as e:
            print(f"Error writing {output_file}: {e}")
        response_queue.task_done()

def main():
    BATH = '../../../data/courses/'
    # 建立共享的 Queue
    response_queue = queue.Queue()
    num_workers = 4  # 可依需求調整 writer thread 數量
    workers = []

    # 啟動 writer threads
    for _ in range(num_workers):
        t = threading.Thread(target=writer_worker, args=(response_queue,))
        t.start()
        workers.append(t)
    
    #for course_i, course in enumerate(['31291', '33749', '35915', '38179', '41797']):
    for course_i,course in enumerate(os.listdir(BATH)):
        
        base_path = f'{BATH}/{course}/AOIs'
        for video in tqdm.tqdm(os.listdir(base_path),desc=f'{course_i} / 5 Video: '):
            for page in tqdm.tqdm(sorted(os.listdir(f'{base_path}/{video}')),'Page: '):
                os.makedirs(f'{base_path}/{video}/{page}/describes_phi4',exist_ok=True)
                for AOI in tqdm.tqdm(os.listdir(f'{base_path}/{video}/{page}/img'),'AOIs: '):
                    aoi_name = AOI.replace('.jpg', '')
                    if not AOI.endswith('.jpg') or os.path.exists(f'{base_path}/{video}/{page}/describes_phi4/{aoi_name}.json'):
                        continue
                    dir_path = f'{base_path}/{video}/{page}'
                    response = inference(dir_path, aoi_name)
                    response = response.replace('|end|>','')
                    response_queue.put((dir_path, aoi_name, response))

    # 等待 queue 中所有任務處理完成
    response_queue.join()

    # 發送 sentinel 給所有 worker threads 以結束
    for _ in range(num_workers):
        response_queue.put(None)
    for t in workers:
        t.join()

if __name__ == '__main__':
    main()