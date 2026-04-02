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
| `preprocess-VLM` | Image Description | `transformers`, `phi-4` |
| `autovag-train` | Model Training / Evaluation | `llama-factory`, `vllm` |

### Environment Setup: `preprocess-ASR`
Used for video downloading and speech-to-text.
```bash
# 0. Create environment
conda create -n preprocess-ASR python==3.11
conda activate preprocess-ASR

# 1. Install requirements
cd preprocess/ASR
pip install -r requirements.txt
```

### Environment Setup: `preprocess-AOI-det`
This environment requires a specific setup for **Detectron2** and **DiT**.
```bash
# 0. Create environment
conda create -n preprocess-AOI-det python==3.10
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
pip install psutil
```

### Environment Setup: `preprocess-VLM`
Used for scene description using **Phi-4 MM** or **Pixtral-12B**.
```bash
# 0. Create environment
conda create -n preprocess-VLM python==3.11
conda activate preprocess-VLM

# 1. Install requirements
cd preprocess/AOI/phi4
pip install -r requirements.txt

# 2. (Optional) Install Flash-Attention for better performance
pip install flash-attn --no-build-isolation
```

### Environment Setup: `autovag-train` (Llama-Factory)
This environment is used for model training and **vLLM** evaluation.
```bash
# 0. Create environment
conda create -n autovag-train python==3.11
conda activate autovag-train

# 1. Install Llama-Factory and its dependencies
cd train/llama_factory
pip install -e . && \
pip install -r requirements/vllm.txt && \
pip install -r requirements/liger-kernel.txt && \
pip install -r requirements/metrics.txt && \
pip install -r fp8.txt && \
pip install -r fp8-te.txt
```

---

## 📈 Pipeline: Preprocessing

Preprocessing is automated via a master script that handles environment switching. Use this script to run the entire flow or specific stages.

### 1. Make the script executable
```bash
chmod +x preprocess/run_pipeline.sh
```

### 2. Run the pipeline
Available stages: `all` (default), `asr`, `det`, `vlm`, `dataset`.

```bash
# Run all stages sequentially
./preprocess/run_pipeline.sh

# Run only transcription (ASR)
./preprocess/run_pipeline.sh asr

# Run only visual analysis (AOI)
./preprocess/run_pipeline.sh det

# Run only scene description (VLM)
./preprocess/run_pipeline.sh vlm
```

---

## 🚀 Training

Once the preprocessing is complete and the dataset is generated, you can start model training using **Llama-Factory**.

### Run Training
```bash
# 1. Activate the training environment
conda activate autovag-train

# 2. Navigate to the train directory
cd train

# 3. Start training using the provided config
llamafactory-cli train yaml/train.yaml
```

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
- Lectures from **NTU Hung-yi Lee's** and **NTU Yun-Nung Chen's** courses.
- ASR Powered by [MediaTek-Research/Breeze-ASR-25](https://huggingface.co/MediaTek-Research/Breeze-ASR-25).
- VLM Powered by [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct).
- LLM Powered by [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
- Object Detection by [Microsoft DiT](https://github.com/microsoft/unilm/tree/master/dit).
- Tools: [Llama-Factory](https://github.com/hiyouga/LlamaFactory), [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper), [vLLM](https://github.com/vllm-project/vllm).
