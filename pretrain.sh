#!/bin/bash
# Pretraining with the combined multi-dataset loader.
# Set the paths below before running.

# ---- Paths (edit these) ----
llm_model_path="/path/to/llama"          # dir containing tokenizer.model
data_path="/path/to/pretrain/train.json" # used by the combined loader config
image_folder=""                          # leave empty if json paths are absolute
save_dir="./checkpoints_pretrain"
hf_home="$HOME/hf_cache"

# ---- Hyperparameters ----
train_epochs=100
learning_rate=0.00001
llama_layers=32
adapter_percentage=0.95
query_len=10
num_prompts=10
deep_prompt_layers=1
prompt_dim=256
batch_size=4
vision_model="biomedclip"

# ---- Parallelization ----
master_port=24005
num_processes=2
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME="$hf_home"
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=30
export NCCL_SOCKET_NTHREADS=8
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_TIMEOUT=600000

mkdir -p "$save_dir" logs_pretrain
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="logs_pretrain/pretrain_${timestamp}.log"
echo "Starting pretraining at $(date)" | tee "$log_file"

accelerate launch --multi_gpu --mixed_precision bf16 \
  --num_processes $num_processes --main_process_port $master_port main.py \
  --llm-model-path "$llm_model_path" \
  --vision-model $vision_model \
  --llm-layers $llama_layers \
  --adapter-percentage $adapter_percentage \
  --adapter-strategy "late" \
  --query-len $query_len \
  --use-deep-prompts \
  --num-deep-prompt-layers $deep_prompt_layers \
  --num-prompts $num_prompts \
  --prompt-dim $prompt_dim \
  --data-path "$data_path" \
  --image-folder "$image_folder" \
  --batch-size $batch_size \
  --epochs $train_epochs \
  --learning-rate $learning_rate \
  --lr $learning_rate \
  --warmup-epochs 5 \
  --gradient-accumulation-steps 4 \
  --warmup-steps 10 \
  --save-dir "$save_dir" \
  --save-steps 500 \
  --eval-steps 500 \
  --phase "pretrain" \
  --task-type "vqa" \
  --use-combined-loader \
  --mixed-precision "bf16" >> "$log_file" 2>&1

echo "Pretraining completed at $(date)" | tee -a "$log_file"
