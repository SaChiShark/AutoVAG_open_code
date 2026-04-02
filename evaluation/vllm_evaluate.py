from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import json
import csv
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--valid_dataset", type=str, required=True, help="Valid dataset path")
parser.add_argument("--lora_path", type=str, default=None, help="LoRA path")
args = parser.parse_args()
lora_path = args.lora_path
if lora_path is None:
    lora_name = "no_lora"
else:
    lora_name = lora_path.split('/')[-1]
valid_data_path = args.valid_dataset
with open(valid_data_path,'r',encoding='utf-8') as f:
    valid_dataset = json.load(f)
model = LLM(model="meta-llama/Llama-3.2-3B-Instruct", enable_lora=True,max_model_len=3600, max_lora_rank=32,
    gpu_memory_utilization=0.9)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")


def translation(questions):
    prompts = []
    anses = []
    aoi_counts = []
    courses = []
    def format_message(question):
        return [
            {
                "role": "system",
                "content": "You are a highly accurate natural language processing model specializing in semantic similarity tasks. Your task is to analyze a given subtitle and compare it with a list of narrative descriptions. Based on the semantic meaning of the subtitle and the narratives, identify which narrative is the closest match to the subtitle.Return the index of the closest narrative. Focus on the meaning and context of the text, rather than exact word matches."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    for question in questions:
        if question['aoi_count'] > 1:
            message = format_message(question['conversations'][0]['value'])
            prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
            prompts.append(prompt)
            anses.append(question['conversations'][1]['value'])
            courses.append(question['course'])
            aoi_counts.append(question['aoi_count'])
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=4000)
    if lora_path is None:
        outputs = model.generate(prompts, sampling_params)
    else:
        outputs = model.generate(prompts, sampling_params,lora_request=LoRARequest("AOI selector", 1, lora_path))
    generated_texts = []
    question_count = {}
    correct_count = {}
    for ans,output,course,aoi_count in zip(anses,outputs,courses,aoi_counts):
        if not course in question_count.keys():
            question_count[course] = [0,0,0,0]
            correct_count[course] = [0,0,0,0]
        for i in range(4):
            if aoi_count >= i + 2:
                question_count[course][i] +=1
                if ans == output.outputs[0].text:
                    correct_count[course][i] +=1
        try:
            generated_texts.append(int(output.outputs[0].text.replace('最可能對應的是:','').replace('\"','')))
        except:
            generated_texts.append(-1)
    return generated_texts,question_count,correct_count

generated_texts,question_counts,correct_counts = translation(valid_dataset)
valid_dataset_name = valid_data_path.split('/')[-1].replace('.json','')
with open(f'eval_result/{lora_name}_{valid_dataset_name}_prdict_result.json', 'w') as f:
    json.dump(generated_texts,f)
with open(f'eval_result/{lora_name}_{valid_dataset_name}.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    # 寫入一列資料
    writer.writerow(['Course','Question count >= 2','Question count >= 3','Question count >= 4','Question count >= 5','Acc >= 2','Acc >= 3','Acc >= 4','Acc >= 5'])
    for key in question_counts.keys():
        writer.writerow([key] + [question_count for question_count in question_counts[key]] + [correct_count for correct_count in correct_counts[key]])
      
with open('eval_result/evaluation_summary.csv', 'a', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    # 寫入一列資料
    correct_count = 0
    question_count = 0
    for key in question_counts.keys():
        correct_count  += correct_counts[key][0]
        question_count += question_counts[key][0]
    
    writer.writerow([lora_name,valid_dataset_name, correct_count / question_count])
    
    
