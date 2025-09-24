Preference_domain="Risk-taking"
num_samples=1000
MODEL_NAME_PATH="Qwen/Qwen2.5-3B-Instruct"
CUDA_VISIBLE_DEVICES=1 python train/main.py \
    --model_name $MODEL_NAME_PATH \
    --train_data_path /data/vcd/pair/vcd_train_Risk-taking_1000.json \
    --output_dir models/Qwen2.5-3B-Instruct/${Preference_domain}_${num_samples} \
    --max_length 512 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --warmup_steps 100 \
    --save_steps 50 \
    --eval_steps 5 \
    --paired \
    --loss_weight_a 1.0 \
    --loss_weight_b 1.0 

CUDA_VISIBLE_DEVICES=1 python train/main.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --train_data_path data/bqd/sft_dataset/sycophancy/dataset.json \
    --output_dir models/sft_output/sycophancy_pair \
    --max_length 512 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --warmup_steps 100 \
    --save_steps 50 \
    --eval_steps 5 \
    --paired \
    --loss_weight_a 1.0 \
    --loss_weight_b 1.0 
