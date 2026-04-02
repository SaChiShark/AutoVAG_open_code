#!/bin/bash

# AutoVAG Preprocessing Pipeline
# This script automates the multi-environment preprocessing workflow.

# --- Configuration ---
ENV_ASR="preprocess-ASR"
ENV_DET="preprocess-AOI-det"
ENV_VLM="preprocess-VLM"

# Function to run a command in a specific conda environment
run_in_env() {
    local env_name=$1
    shift
    echo "----------------------------------------------------------------"
    echo "🚀 Running in environment: [$env_name]"
    echo "📂 Command: $@"
    echo "----------------------------------------------------------------"
    conda run --no-capture-output -n "$env_name" "$@"
    if [ $? -ne 0 ]; then
        echo "❌ Error occurred in stage using environment $env_name. Exiting."
        exit 1
    fi
}

# --- Stage Selection ---
STAGE=${1:-all}

echo "🌟 Starting AutoVAG Preprocessing Pipeline [Stage: $STAGE]"

# --- Stage 1: Download & ASR ---
if [[ "$STAGE" == "all" || "$STAGE" == "asr" ]]; then
    echo "🎬 Stage 1: Download & ASR"
    run_in_env "$ENV_ASR" python download_video.py
    
    cd ASR
    run_in_env "$ENV_ASR" python check_srt_exist.py
    run_in_env "$ENV_ASR" python mk_subtitle.py
    cd ../
fi

# --- Stage 2: AOI Detection ---
if [[ "$STAGE" == "all" || "$STAGE" == "det" ]]; then
    echo "🔍 Stage 2: AOI Detection"
    cd AOI
    # Note: These might use ASR or Base env
    run_in_env "$ENV_DET" python screenshot.py
    run_in_env "$ENV_DET" python slide_splitter.py -f
    
    cd object_detection
    run_in_env "$ENV_DET" python inference.py
    cd ../
    run_in_env "$ENV_DET" python copy_screen2original_page_image.py
fi

# --- Stage 3: VLM Description ---
if [[ "$STAGE" == "all" || "$STAGE" == "vlm" ]]; then
    echo "📝 Stage 3: VLM Description & Post-processing"
    cd phi4
    run_in_env "$ENV_VLM" python inference.py
    cd ../../
fi

# --- Stage 4: Dataset Synthesis ---
if [[ "$STAGE" == "all" || "$STAGE" == "dataset" ]]; then
    echo "📊 Stage 4: Dataset Synthesis"
    cd dataset
    run_in_env "$ENV_VLM" python make_base_dataset.py
    run_in_env "$ENV_VLM" python detect_laser.py
    run_in_env "$ENV_VLM" python find_laser_coresponse_AOI.py
    run_in_env "$ENV_VLM" python make_sharegpt_dataset.py
    cd ../..
fi

echo "✅ Preprocessing Pipeline Completed Successfully!"