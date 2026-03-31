import json
import os
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
import tqdm
# Load the multimodal Pixtral-12B model
model_id = "mistral-community/pixtral-12b"
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

#quantization_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_compute_dtype=torch.float16,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_use_double_quant=True,
#)

from transformers import LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda",
    quantization_config=quantization_config,
    attn_implementation={"text_config": "flash_attention_2", "vision_config": "eager"}
)

processor = AutoProcessor.from_pretrained(model_id)
def prepare_dtype(data, dtype=torch.float16):
    if isinstance(data, torch.Tensor):
        # 只針對浮點數轉型，避免把整數的 input_ids 也轉掉
        return data.to(dtype) if data.is_floating_point() else data
    elif isinstance(data, list):
        return [prepare_dtype(i, dtype) for i in data]
    elif isinstance(data, dict):
        return {k: prepare_dtype(v, dtype) for k, v in data.items()}
    return data
def process_images_and_generate(original_img, AOIs):
    images = [Image.open(original_img)] + [Image.open(aoi) for aoi in AOIs]
    content = []
    if images[0].size[1] > 480:
        scale = 480 / images[0].size[1] 
        for i in range(len(images)):
            images[i] = images[i].resize((int(images[i].size[0] * scale),int(images[i].size[1] * scale)))
    for i in range(len(AOIs)):
        content.append({"type": "text", "content": f"Please describe this area, Image {i}:"})
        content.append({"type": "image"})
    chat = [
        {
            "role": "user",
            "content": [
                {"type": "text", "content":"This is a course slide imgae that contains several Areas of Interest (AOI) and I will give you later. Please follow the instructions below when responding:\n\n1. When you receive an image containing an AOI, carefully observe and describe the content and characteristics of that area.\n2. Begin each response with 'Image x:', where x represents the number of that AOI (e.g., Image1:, Image2:, etc.).\n3. Your description should be detailed and clear, highlighting the key elements in the image.\n4. At the end of your description, please include a clear termination symbol [END] to indicate that the description is complete.\n\nFor example:\n\nUser message: 'Please describe this area, Image 1:' along with the corresponding image. Please describe this area, Image 2:' along with the corresponding image.\nYour answer: 'Image 1: [detailed description] [END] \n Image 2: [detailed description] [END]'\n\nPlease provide your description according to the instructions above."},
                {"type": "image"},
            ]
        },
        {
        "role": "assistant",
        "content": "Alright, please provide the subsequent images and the areas of interest (AOI) you are interested in, and I will describe them according to your requirements.",
        },
        {
            "role": "user",
            "content": content
        }
    ]


    # Apply chat template
    prompt = processor.apply_chat_template(chat)

    # Process inputs for the model
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)
    inputs = prepare_dtype(inputs)
    # Generate response
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=2048, pad_token_id=processor.tokenizer.eos_token_id)

    # Decode output
    input_length = inputs["input_ids"].shape[1] 
    new_tokens = generate_ids[:, input_length:]  

    # Decode only new tokens
    output = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output

# Base directory
base_path = '../../data/courses'

def process_all_course():
    # Process each course's images
    for j,course in enumerate(tqdm.tqdm(os.listdir(base_path),desc="Course: ")):
        AOIs_path = f'{base_path}/{course}/AOIs'

        for video in tqdm.tqdm(os.listdir(AOIs_path),desc='Video:'):

            video_path = f'{AOIs_path}/{video}'
            for page in tqdm.tqdm(os.listdir(video_path),desc=f"Page {course} {video}: "):
                if os.path.exists(f'{video_path}/{page}/describes_pixtral_12B_8bit.txt'):
                    continue
                original_img = f'{video_path}/{page}/original.jpg'
                AOIs = [f'{video_path}/{page}/img/{i}.jpg' for i in range(len(os.listdir(f'{video_path}/{page}/img')))]
                ## Generate descriptions
                response = process_images_and_generate(original_img, AOIs)
                with open(f'{video_path}/{page}/describes_pixtral_12B_8bit.txt', "w", encoding="utf-8") as file:
                    file.write(response)
process_all_course()