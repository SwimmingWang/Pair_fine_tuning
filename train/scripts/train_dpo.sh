MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=1 accelerate launch \
  --config_file train/dpo/accelerate_config_dpo.yaml \
  --main_process_port 29514 \
    train/dpo/dpo.py \
    --model_name $MODEL_PATH \
    --learning_rate 5e-5 \
    --max_length 512 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 10 \
    --use_lora_train_dpo \
    --evaluation_steps 5 \
    --ref_model_path $MODEL_PATH \
    --dpo_data_path data/bqd/dpo_dataset/hallucination/explicit/hallucination_pos.json \
    --template_path train/tools/template.jinja \
    --checkpoint_dir models/Qwen2.5-7B-Instruct/dpo_output/hallucination/hallucination_pos \
    --wandb_project pft \
    --wandb_run_name dpo-hallucination-hallucination_pos \

