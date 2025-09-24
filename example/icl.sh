num_samples_train=1000
Model_name="Qwen/Llama-3.1-8B-Instruct"
Dataset_name="data/vcd_test.json"

target_value="Immediate_gratification"

python evaluate/icl/icl.py \
    --model_serve "vllm_serve" \
    --model_name ${Model_name} \
    --dataset_name ${Dataset_name} \
    --result_folder_name "results/Llama-3.1-8B-Instruct/icl/Immediate_gratification_${num_samples_train}" \
    --temperature 0.1 \
    --top_p 0.9 \
    --max_length 2048 \
    --target_value ${target_value} \
    --select_one \
    --port 7021 \
    --num_samples -1

python evaluate/icl/icl.py \
    --model_serve "vllm_serve" \
    --model_name ${Model_name} \
    --dataset_name ${Dataset_name} \
    --result_folder_name "results/Llama-3.1-8B-Instruct/icl/Immediate_gratification_${num_samples_train}" \
    --temperature 0.1 \
    --top_p 0.9 \
    --max_length 2048 \
    --target_value ${target_value} \
    --select_one \
    --port 7021 \
    --use_lora \
    --lora_name "lora4" \
    --num_samples -1
