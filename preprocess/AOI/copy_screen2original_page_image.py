import os
import json
from pathlib import Path

# 使用 Path 物件包裝基礎路徑
base_dir = Path('../../data/courses').resolve()

# 確保基礎路徑存在
if not base_dir.exists():
    print(f"錯誤：找不到基礎目錄 {base_dir}")
    exit(1)

for course_dir in base_dir.iterdir():
    if not course_dir.is_dir():
        continue
        
    split_results_dir = course_dir / "split_results"
    if not split_results_dir.exists():
        continue

    for json_file in split_results_dir.glob("*.json"):
        video_name = json_file.stem  
        aoi_video_path = course_dir / "AOIs" / video_name
        
        # 讀取 JSON
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                compare_result = json.load(f)
        except Exception as e:
            print(f"讀取 {json_file} 失敗: {e}")
            continue

        for page, screenshot_idx in compare_result.items():
            page_path = aoi_video_path / f"page_{page}"
            page_path.mkdir(parents=True, exist_ok=True)
            
            src_file = course_dir / "screenshots" / video_name / f"screenshot_{screenshot_idx}.jpg"
            dst_link = page_path / "original.jpg"
            if not src_file.exists():
                print(f"警告：來源圖檔不存在 {src_file}")
                continue
                
            try:
                # 使用 resolve() 確保是絕對路徑，避免 symlink 指向錯誤
                if dst_link.is_symlink() or dst_link.exists():
                    dst_link.unlink()  # 如果已存在，先刪除舊的 (比 try-except 更乾淨)
                
                os.symlink(src_file.resolve(), dst_link)
                
            except Exception as e:
                print(f"建立連結失敗 {dst_link} -> {src_file}: {e}")
