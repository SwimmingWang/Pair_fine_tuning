Preference_domain="Social_Strategy"
Target_value="Competitive"
num_samples=1000
CUDA_VISIBLE_DEVICES=1 python train/main.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --train_data_path data/vcd/sft_data/${Preference_domain}/${Target_value}_${num_samples}.json \
    --output_dir models/Qwen2.5-3B-Instruct/sft_output/${Preference_domain}_${Target_value}_${num_samples} \
    --max_length 512 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --warmup_steps 100 \
    --save_steps 50 \
    --eval_steps 5 

Target_value="Collaborative"
CUDA_VISIBLE_DEVICES=1 python train/main.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --train_data_path data/vcd/sft_data/${Preference_domain}/${Target_value}_${num_samples}.json \
    --output_dir models/Qwen2.5-3B-Instruct/sft_output/${Preference_domain}_${Target_value}_${num_samples} \
    --max_length 512 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --warmup_steps 100 \
    --save_steps 50 \
    --eval_steps 5 

CUDA_VISIBLE_DEVICES=1 python train/main.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --train_data_path data/bqd/sft_dataset/hallucination/hallucination_pos.json \
    --output_dir models/Qwen2.5-7B-Instruct/sft_output/hallucination_pos \
    --max_length 512 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --warmup_steps 100 \
    --save_steps 50 \
    --eval_steps 5 

CUDA_VISIBLE_DEVICES=1 python train/main.py \
    --model_name Qwen2.5-3B-Instruct \
    --train_data_path data/bqd/sft_dataset/hallucination/hallucination_neg.json \
    --output_dir models/Qwen2.5-7B-Instruct/sft_output/hallucination_neg \
    --max_length 512 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --warmup_steps 100 \
    --save_steps 50 \
    --eval_steps 5 