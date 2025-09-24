MODEL_NAME="/Qwen2.5-7B-Instruct"

LORA_NAME1="lora1"
LORA_PATH1="models/Qwen2.5-7B-Instruct/sft_output/Social_Strategy_Competitive_1000/final_adapter"
LORA_NAME2="lora2"
LORA_PATH2="models/Qwen2.5-7B-Instruct/sft_output/Social_Strategy_Collaborative_1000/final_adapter"

PORT=7021

CUDA_VISIBLE_DEVICES=4 vllm serve ${MODEL_NAME} \
  --enable-lora \
  --lora-modules ${LORA_NAME1}=${LORA_PATH1} ${LORA_NAME2}=${LORA_PATH2} \
  --port ${PORT} 