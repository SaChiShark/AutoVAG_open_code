# AutoVAG: Automatic Video Analysis and Generation

AutoVAG is a comprehensive pipeline for extracting knowledge from educational videos (e.g., YouTube lectures) to train and evaluate multi-modal Large Language Models.

---

## 🛠 Environments & Installation

Due to conflicting dependencies (Detectron2, Faster-Whisper, and Transformers), this project uses multiple Conda environments.

### 1. Preprocessing Environments

| Name | Usage | Key Packages |
| :--- | :--- | :--- |
| `preprocess-ASR` | Downloading & Transcription | `pytubefix`, `faster-whisper` |
| `preprocess-AOI-det` | Vision/Object Detection | `detectron2`, `torch`, `cv2` |
| `autovag-vlm` | Image Description / Training | `transformers`, `llama-factory` |

### Environment Setup: `preprocess-AOI-det`
This environment requires a specific setup for **Detectron2** and **DiT**.
```bash
# 0. Create environment
conda create -n preprocess-AOI-det python==3.9
conda activate preprocess-AOI-det

# 1. Install PyTorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install requirements
cd preprocess/AOI/object_detection
pip install -r requirements.txt

# 3. Install Detectron2 
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
# 4. Install additional dependencies
pip install shapely
```

### Environment Setup: `autovag-vlm` (Shared with Training)
This environment is used for scene description (Pixtral/Phi-4) and model training using **Llama-Factory**.
```bash
# 0. Create environment (Python 3.10+ recommended)
conda create -n autovag-vlm python==3.10
conda activate autovag-vlm

# 1. Install Llama-Factory and its dependencies
cd train/llama_factory
pip install -e .
pip install -r requirements/metrics.txt
pip install -r requirements/liger-kernel.txt
pip install -r requirements/vllm.txt
pip install -r fp8.txt
pip install -r fp8-te.txt
```

---

## 📈 Pipeline: Preprocessing

The preprocessing workflow is split into several steps. Note the environment required for each.

### Step 1: Video Download
**Env:** `preprocess-ASR`
Downloads playlists and extracts audio.
```bash
python preprocess/download_video.py
```

### Step 2: ASR (Speech to Text)
**Env:** `preprocess-ASR`
Transcribes audio into Traditional Chinese SRT files.
```bash
cd preprocess/ASR
python check_srt_exist.py
python mk_subtitle.py
```

### Step 3: AOI Analysis (Visual)

#### 3.1 Object Detection
**Env:** `preprocess-AOI-det`
Extracts slides and identifies Areas of Interest (AOIs) using DiT.
1. **Screenshot & Split**: `screenshot.py`, `slide_splitter.py`
2. **Detection**: `object_detection/inference.py` (using DiT)

#### 3.2 Multimodal Description
**Env:** `autovag-vlm`
Generates detailed textual descriptions for each detected AOI.
- **Pixtral-12B**: Using `describe.py`
- **Phi-4 MM**: Using `phi4/phi_4.py` or `phi4/inference.py`

### Step 4: Dataset Generation
**Env:** Base/ASR
Synthesizes the final ShareGPT/Llama-Factory compatible dataset.
```bash
cd preprocess/dataset
python make_sharegpt_dataset.py
```

---

## 🚀 Training

Details for fine-tuning via **Llama-Factory** go here.

---

## 📊 Evaluation

Details for inference via **vLLM** go here.

---

## 📂 Project Structure

```text
AutoVAG_open_code/
├── data/               # Raw and processed data
├── datasets/           # Final training datasets
├── evaluation/         # vLLM evaluation scripts
├── preprocess/         # Preprocessing pipeline
│   ├── AOI/            # Vision analysis
│   ├── ASR/            # Audio transcription
│   └── dataset/        # Formatting tools
└── train/              # Training configs & Llama-Factory
```

---

## 📜 Acknowledgments
- Videos sourced from **NTU Hung-yi Lee's** courses.
- Powered by [Llama-Factory](https://github.com/hiyouga/LlamaFactory), [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper), and [vLLM](https://github.com/vllm-project/vllm).
