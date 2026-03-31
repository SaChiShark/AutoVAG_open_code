import os
import requests
import torch
from PIL import Image
import soundfile
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

model_path = 'microsoft/Phi-4-multimodal-instruct'

kwargs = {}
kwargs['torch_dtype'] = torch.bfloat16

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print(processor.tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype='auto',
    _attn_implementation='flash_attention_2',
).cuda()
print("model.config._attn_implementation:", model.config._attn_implementation)

generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'


AOI_path = '/home/mvnl/code/DSA/course/A1_stack_and_queue/AOIs/6/page_1/img'
AOI_count = len(os.listdir(AOI_path))
chat = [
    {'role': 'user', 'content': f"<|image_1|>This is a course slide imgae that contains {AOI_count} Areas of Interest (AOI) and I will give you later. Please follow the instructions below when responding:\n\n1. When you receive an image containing an AOI, carefully observe and describe the content and characteristics of that area.\n2. Begin each response with 'Image x:', where x represents the number of that AOI (e.g., Image0:, Image1:, etc.).\n3. Your description should be detailed and clear, highlighting the key elements in the image.\n4. At the end of your description, please include a clear termination symbol [END] to indicate that the description is complete.\n\nFor example:\n\nUser message: 'Please describe this area, Image 0:' along with the corresponding image. Please describe this area, Image 1:' along with the corresponding image.\nYour answer: 'Image 0: [detailed description] [END] \n Image 1: [detailed description] [END]'\n\nPlease provide your description according to the instructions above and an empty description and only [END] is not allowed."},
    {
        'role': 'assistant',
        'content': "Alright, please provide the subsequent images and the areas of interest (AOI) you are interested in, and I will describe them according to your requirements.",
    },
    {'role': 'user', 'content': '<|image_2|>Please describe this image.'},
]
original = Image.open('/home/mvnl/code/DSA/course/A1_stack_and_queue/AOIs/6/page_1/original.jpg')
images = [Image.open(AOI_path + '/' + path) for path in os.listdir(AOI_path)]
prompt = processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
if prompt.endswith('<|endoftext|>'):
    prompt = prompt.rstrip('<|endoftext|>')

print(f'>>> Prompt\n{prompt}')

inputs = processor(prompt, [original,images[0]], return_tensors='pt').to('cuda:0')
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')