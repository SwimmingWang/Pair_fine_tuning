num_samples_train=1000
Model_name="Qwen/Qwen2.5-7B-Instruct"
Dataset_name="data/vcd_test.json"

target_value="Competitive"

python evaluate/evaluate.py \
    --model_serve "vllm_serve" \
    --model_name ${Model_name} \
    --dataset_name ${Dataset_name} \
    --result_folder_name "results/Qwen2.5-7B-Instruct/sft_Competitive_${num_samples_train}" \
    --temperature 0.1 \
    --top_p 0.9 \
    --max_length 512 \
    --target_value ${target_value} \
    --port 7021 \
    --use_lora \
    --lora_name "lora1" \
    --parallel \
    --num_samples -1

python evaluate/evaluate.py \
    --model_serve "vllm_serve" \
    --model_name ${Model_name} \
    --dataset_name ${Dataset_name} \
    --result_folder_name "results/Qwen2.5-7B-Instruct/sft_Competitive_${num_samples_train}" \
    --temperature 0.1 \
    --top_p 0.9 \
    --max_length 512 \
    --target_value ${target_value} \
    --select_one \
    --port 7021 \
    --use_lora \
    --lora_name "lora1" \
    --parallel \
    --num_samples -1

python evaluate/eval_open.py \
    --model_serve "vllm_serve" \
    --model_name ${Model_name} \
    --dataset_name ${Dataset_name} \
    --result_folder_name "results/Qwen2.5-7B-Instruct/sft_Competitive_${num_samples_train}" \
    --temperature 0.1 \
    --top_p 0.9 \
    --max_length 512 \
    --target_value ${target_value} \
    --parallel \
    --use_lora \
    --lora_name "lora1" \
    --port 7021 \
    --num_samples -1

python evaluate/evaluate.py \
    --model_serve "vllm_serve" \
    --model_name ${Model_name} \
    --dataset_name ${Dataset_name} \
    --result_folder_name "results/Qwen2.5-7B-Instruct/sft_Collaborative_${num_samples_train}" \
    --temperature 0.1 \
    --top_p 0.9 \
    --max_length 512 \
    --target_value ${target_value} \
    --port 7021 \
    --use_lora \
    --lora_name "lora2" \
    --parallel \
    --num_samples -1

python evaluate/evaluate.py \
    --model_serve "vllm_serve" \
    --model_name ${Model_name} \
    --dataset_name ${Dataset_name} \
    --result_folder_name "results/Qwen2.5-7B-Instruct/sft_Collaborative_${num_samples_train}" \
    --temperature 0.1 \
    --top_p 0.9 \
    --max_length 512 \
    --target_value ${target_value} \
    --select_one \
    --port 7021 \
    --use_lora \
    --lora_name "lora2" \
    --parallel \
    --num_samples -1


python evaluate/eval_open.py \
    --model_serve "vllm_serve" \
    --model_name ${Model_name} \
    --dataset_name ${Dataset_name} \
    --result_folder_name "results/Qwen2.5-7B-Instruct/sft_Collaborative_${num_samples_train}" \
    --temperature 0.1 \
    --top_p 0.9 \
    --max_length 512 \
    --target_value ${target_value} \
    --parallel \
    --use_lora \
    --lora_name "lora6" \
    --port 7021 \
    --num_samples -1