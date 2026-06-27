#!/bin/bash
# Fine-tuning on EchoVQA, initialized from a pretrained checkpoint.
# Set the paths below before running.

# ---- Paths (edit these) ----
llm_model_path="/path/to/llama"              # dir containing tokenizer.model
data_path="/path/to/echovqa/train.json"      # EchoVQA train json
val_data_path="/path/to/echovqa/test.json"   # optional; remove flag below to skip eval
image_folder=""                              # leave empty if json paths are absolute
save_dir="./checkpoints_echovqa"
resume_from="/path/to/pretrained/checkpoint.pth"
hf_home="$HOME/hf_cache"

# ---- Hyperparameters ----
train_epochs=100
learning_rate=0.001
llama_layers=32
adapter_percentage=0.95
query_len=10
num_prompts=10
deep_prompt_layers=1
prompt_dim=256
batch_size=8
vision_model="biomedclip"

master_port=23411
num_processes=2
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME="$hf_home"
export NCCL_DEBUG=INFO

mkdir -p "$save_dir" echovqa_logs
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="echovqa_logs/finetune_${timestamp}.log"
echo "Starting fine-tuning at $(date)" | tee "$log_file"

accelerate launch --multi_gpu --mixed_precision bf16 \
  --num_processes $num_processes --main_process_port $master_port main.py \
  --llm-model-path "$llm_model_path" \
  --vision-model $vision_model \
  --llm-layers $llama_layers \
  --adapter-percentage $adapter_percentage \
  --adapter-strategy "late" \
  --query-len $query_len \
  --num-deep-prompt-layers $deep_prompt_layers \
  --num-prompts $num_prompts \
  --prompt-dim $prompt_dim \
  --data-path "$data_path" \
  --val-data-path "$val_data_path" \
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
  --phase "finetune" \
  --task-type "vqa" \
  --resume-from "$resume_from" \
  --mixed-precision "bf16" >> "$log_file" 2>&1

echo "Fine-tuning completed at $(date)" | tee -a "$log_file"
