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

from transformers import LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda",
    quantization_config=quantization_config,
    attn_implementation={"text_config": "flash_attention_2", "vision_config": "eager"}  # 啟用 Flash Attention v2
)

processor = AutoProcessor.from_pretrained(model_id)

# Function to process images and construct the chat prompt
def process_images_and_generate(original_img, AOIs):
    images = [Image.open(original_img)] + [Image.open(aoi) for aoi in AOIs]
    content = []
    if images[0].size[1] > 720:
        scale = 720 / images[0].size[1] 
        for i in range(len(images)):
            images[i] = images[i].resize((int(images[i].size[0] * scale),int(images[i].size[1] * scale)))
    for i in range(len(AOIs)):
        content.append({"type": "text", "content": f"Please describe this area, Image {i}:"})#請描述這個區域 圖 {i}:"}),
        content.append({"type": "image"})
    AOI_count = len(AOIs)
    # Construct the multimodal prompt
    chat = [
        {
            "role": "user",
            "content": [
                #{"type": "text", "content":  "這是一份完整的簡報，裡面包含了數個您感興趣的區域 (Area of Interest, AOI)。請依照以下指示回答：\n\n1. 當收到包含 AOI 的圖片時，請仔細觀察並描述該區域的內容與特徵。\n2. 請以『圖x:』作為每個回答的開頭，其中 x 為該 AOI 的編號（例如：圖1:, 圖2: 等）。\n3. 您的描述應具體且清晰，重點說明圖片中的關鍵細節。\n4. 在描述結束時，請加入一個明確的結束符號（例如：[結束] 或 ---END---），以告知描述已完成。\n\n例如：\n\n使用者訊息：『請描述這個區域 圖1:』\n您的回答：『圖1: 此處顯示的是…… [詳細描述] [結束]』\n\n請根據上述要求進行描述。"},
                {"type": "text", "content":f"This is a course slide imgae that contains {AOI_count} Areas of Interest (AOI) and I will give you later. Please follow the instructions below when responding:\n\n1. When you receive an image containing an AOI, carefully observe and describe the content and characteristics of that area.\n2. Begin each response with 'Image x:', where x represents the number of that AOI (e.g., Image1:, Image2:, etc.).\n3. Your description should be detailed and clear, highlighting the key elements in the image.\n4. At the end of your description, please include a clear termination symbol [END] to indicate that the description is complete.\n\nFor example:\n\nUser message: 'Please describe this area, Image 0:' along with the corresponding image. Please describe this area, Image 1:' along with the corresponding image.\nYour answer: 'Image 0: [detailed description] [END] \n Image 1: [detailed description] [END]'\n\nPlease provide your description according to the instructions above and an empty description and only [END] is not allowed."},
                {"type": "image"},  # Main image
            ]
        },
        {
        "role": "assistant",
        "content": "Alright, please provide the subsequent images and the areas of interest (AOI) you are interested in, I will describe them according to your requirements and not any empty description or only [END].",
        },
        {
            "role": "user",
            "content": content
        }
    ]
    flag = True
    count = 0
    while True:
        # Apply chat template
        prompt = processor.apply_chat_template(chat)

        # Process inputs for the model
        inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=2048, 
                                          pad_token_id=processor.tokenizer.eos_token_id,
                                          do_sample = True,
                                          temperature = 0.4)

        # Decode output
        # 確保輸入和輸出在同一個設備上
        input_length = inputs["input_ids"].shape[1]  # 取得輸入的長度
        new_tokens = generate_ids[:, input_length:]  # 只保留新生成的部分
        count += 1
        # Decode only new tokens
        output = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if len(output.split("Image ")) - 1 == len(AOIs):
            flag = False
            for temp in output.split("Image "):
                if ': [END]' in temp: 
                    flag = True
            if not flag:
                break
            else:
                continue
        elif count == 4:
            return None
    return output

# Base directory
base_path = '/home/mvnl/code/cool/useful_course_not_EECS'

def process_all_course(course,video,page):
    # Process each course's images
    AOIs_path = f'{base_path}/{course}/AOIs/'
    video_path = f'{AOIs_path}/{video}'
    original_img = f'{video_path}/{page}/original.jpg'
    AOIs = [f'{video_path}/{page}/img/{i}.jpg' for i in range(len(os.listdir(f'{video_path}/{page}/img')))]
    ## Generate descriptions
    if os.path.exists(f'{video_path}/{page}/describes_pixtral_12B_8bit.txt'):
        return True
    response = process_images_and_generate(original_img, AOIs)
    if response is None:
        return False
    with open(f'{video_path}/{page}/describes_pixtral_12B_8bit.txt', "w", encoding="utf-8") as file:
        file.write(response)
    return True

new_error_list = []
with open('/home/mvnl/code/cool/document/describe_error.json','r') as f:
    error_list = json.load(f)
for course,video,page,error_AOI_id in tqdm.tqdm(error_list):
    result = process_all_course(course,video,page)
    if result == False:
        new_error_list.append([course,video,page])
with open('/home/mvnl/code/cool/document/describe_error.json','w') as f:
    json.dump(new_error_list,f,indent=4)