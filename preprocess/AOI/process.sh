python screenshot.py
python slide_splitter.py -f
cd dit/object_detection
python inference.py
cd ../../
python copy_screen2original_page_image.py
python describe.py
python process_raw_AOI_describe.py
python translate.py