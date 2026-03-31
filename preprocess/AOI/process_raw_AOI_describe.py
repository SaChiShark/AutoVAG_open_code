import re
import json
import os
'''
def extract_image_descriptions(text):
    # 正則表達式匹配 "Image X:" 後的描述內容，直到遇到 "[END]"
    pattern = re.compile(r"\*\*?Image (\d+):\*?\*(.*?)\[END\]", re.DOTALL)
    
    # 存儲結果的字典
    image_descriptions = {}
    
    # 檢查是否有未結束的段落
    missing_end_flag = False
    
    for match in pattern.finditer(text):
        image_number = match.group(1)  # 提取 Image 編號
        description = match.group(2).strip()  # 去除前後空格
        image_descriptions[f"Image {image_number}"] = description
    
    # 檢查是否有 Image 開頭但沒有 [END]
    image_entries = re.findall(r"\*\*?Image \d+:\*?\*", text)
    end_entries = re.findall(r"\[END\]", text)
    
    if len(image_entries) != len(end_entries):
        missing_end_flag = True
    
    if missing_end_flag:
        print("Warning: Some entries are missing [END], skipping processing.")
        return {}
    
    return image_descriptions
'''

def extract_image_descriptions(text):
    # 去除 "Image" 前的多餘換行符
    text = re.sub(r"\n+(Image \d+:)", r"\1", text)
    text = text.replace('[END]','')
    # 正則表達式匹配 "Image X:" 後的描述內容，直到遇到 "Image X:" 或文本結束
    pattern = re.compile(r"Image (\d+):\s*(.*?)(?=Image \d+:|$)", re.DOTALL)
    
    # 存儲結果的字典
    image_descriptions = {}
    
    for match in pattern.finditer(text):
        image_number = match.group(1)  # 提取 Image 編號
        description = match.group(2).strip()  # 去除前後空格
        description = re.sub(r"^\*+|\*+$", "", description).strip()  # 去除開頭和結尾的 "*"
        description = description.strip("\n")  # 去除開頭和結尾的換行符
        image_descriptions[f"{image_number}"] = description
    return image_descriptions

def process_txt(file_path,target_count):

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
        
        # 提取描述內容
        descriptions = extract_image_descriptions(text_data)
        if len(descriptions.keys()) != target_count:
            return None
        else:
            
            return descriptions
    except FileNotFoundError:
        print(f"Error: 文件 '{file_path}' 未找到！")
    except Exception as e:
        print(f"Error: {e}")
base_path = '/home/mvnl/code/open_code/data/courses'
count = 0
count_aoi = 0
error_count = 0
error_list = []
for course in os.listdir(base_path):
    course_path = f'{base_path}/{course}/AOIs'
    for video in os.listdir(course_path):
        video_path = f'{course_path}/{video}'
        for page in sorted(os.listdir(video_path)):
            
            count += 1 
            page_path = f'{video_path}/{page}'
            describe_path = f'{page_path}/describes_pixtral_12B_8bit'
            os.makedirs(describe_path,exist_ok=True)
            txt_path = f'{page_path}/describes_pixtral_12B_8bit.txt'
            
            target_count = len(os.listdir(f'{page_path}/img'))
            count_aoi += target_count
            describes = process_txt(txt_path,target_count)
            if not describes is None:
                flag = True
                for key in describes.keys():
                    if describes[key] == '':
                        error_count += 1 
                        print("Error",course,video,page)
                        error_list.append([course,video,page,key])
                        #os.remove(txt_path)
                        flag = False
                        break
                if not flag:
                    continue
                for key in describes.keys():
                    #pass
                    with open(f'{describe_path}/{key}.json','w') as f:
                        json.dump({
                            'Original': describes[key]
                        },f,indent=4)
            else:
                error_count += 1 
                print("Error",course,video,page)
                error_list.append([course,video,page,None])
                #os.remove(txt_path)
print('Total 30396',count)
print('Total AOI',count_aoi)
print('Error',error_count)
print('Error ratio',error_count/count)
#with open('/home/mvnl/code/cool/document/describe_error.json','w') as f:
#    json.dump(error_list,f,indent=4)