Port=7021
Mode="sft"
Direction="pos"
Behavior="hallucination"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
Dataset_type="ab"

python evaluate/bqd/eval.py \
    --model_name $MODEL_PATH \
    --behavior $Behavior \
    --dataset_type $Dataset_type \
    --direction $Direction \
    --model_serve vllm_serve \
    --mode $Mode \
    --temperature 0.1 \
    --top_p 0.9 \
    --max_length 512 \
    --port $Port \
    --use_lora \
    --lora_name "lora1" \
    --result_folder_name results/Qwen2.5-7B-Instruct

python evaluate/bqd/eval_open.py \
    --model_name $MODEL_PATH \
    --model_serve vllm_serve \
    --behaviors $Behavior \
    --temperature 0.1 \
    --top_p 0.9 \
    --max_length 512 \
    --port $Port \
    --method $Direction \
    --mode $Mode \
    --use_lora \
    --lora_name "lora1" \
    --result_dir results/Qwen2.5-7B-Instruct

Mode="sft"
Direction="neg"
Behavior="hallucination"
python evaluate/bqd/eval.py \
    --model_name $MODEL_PATH \
    --behavior $Behavior \
    --dataset_type $Dataset_type \
    --direction $Direction \
    --model_serve vllm_serve \
    --mode $Mode \
    --temperature 0.1 \
    --top_p 0.9 \
    --max_length 512 \
    --port $Port \
    --use_lora \
    --lora_name "lora2" \
    --result_folder_name results/Qwen2.5-7B-Instruct

python evaluate/bqd/eval_open.py \
    --model_name $MODEL_PATH \
    --model_serve vllm_serve \
    --behaviors $Behavior \
    --temperature 0.1 \
    --top_p 0.9 \
    --max_length 512 \
    --port $Port \
    --method $Direction \
    --mode $Mode \
    --use_lora \
    --lora_name "lora2" \
    --result_dir results/Qwen2.5-7B-Instruct
