import json,os
from vllm import LLM, SamplingParams
import tqdm
from transformers import AutoTokenizer
model_id = "meta-llama/Llama-3.1-8B-Instruct"
model = LLM(model=model_id,max_model_len=8192,enable_prefix_caching=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
def translation(original_texts):
    prompts = []
    def format_message(original_text):
        return [
            {
                "role": "system",
                "content": "You are a professional translator specializing in English-to-Traditional Chinese (繁體中文) translation. \
        Ensure that your translations are accurate, fluent, and natural, preserving the original tone and intent of the text. \
        Do not include any explanations, labels, or formatting—only output the translated text."
            },
            {
                "role": "user",
                "content": f"Translate the following English text into Traditional Chinese (繁體中文):\n\n{original_text}"
            }
        ]
    for original_text in original_texts:
        message = format_message(original_text)
        prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=3000)
    outputs = model.generate(prompts, sampling_params)
    generated_texts = []
    for output in outputs:
        generated_texts.append(output.outputs[0].text)
    return generated_texts

base_path = '/home/mvnl/code/open_code/data/courses'
count = 0
error_count = 0
error_list = []
batch_size = 5000
bar = tqdm.tqdm(total=193,desc='AOI: ')
batch = []
for course in os.listdir(base_path):
    course_path = f'{base_path}/{course}/AOIs'
    for video in os.listdir(course_path):
        video_path = f'{course_path}/{video}'
        for page in sorted(os.listdir(video_path)):
            describe_path = f'{video_path}/{page}/describes_pixtral_12B_8bit'
            if not os.path.exists(describe_path):
                continue
            for aoi in sorted(os.listdir(describe_path)):
                batch.append(f'{describe_path}/{aoi}')
                if len(batch) == batch_size:
                    describes = []
                    originals = []
                    for path in batch:
                        with open(path) as f:
                            describe = json.load(f)
                            describes.append(describe)
                            originals.append(describe['Original'])
                    translateds = translation(originals)
                    for i in range(batch_size):
                        describes[i]['Chinese_llama_8b'] = translateds[i]
                        with open(batch[i],'w',encoding='utf-8') as f:
                            json.dump(describes[i],f,indent=4,ensure_ascii=False)
                    bar.update(len(batch))
                    batch = []
describes = []
originals = []
for path in batch:
    with open(path) as f:
        describe = json.load(f)
        describes.append(describe)
        originals.append(describe['Original'])
translateds = translation(originals)
for i in range(len(batch)):
    describes[i]['Chinese_llama_8b'] = translateds[i]
    with open(batch[i],'w',encoding='utf-8') as f:
        json.dump(describes[i],f,indent=4,ensure_ascii=False)
bar.update(len(batch))