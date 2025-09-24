BEHAVIORS="hallucination"
MODEL_NAME_PATH="Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT="pos"
TYPE="open_ended" # ab, open-ended

CUDA_VISIBLE_DEVICES=1 python train/caa/caa.py \
    --model_name_path $MODEL_NAME_PATH \
    --behaviors $BEHAVIORS

CUDA_VISIBLE_DEVICES=1 python train/caa/normalized.py \
    --model_name_path $MODEL_NAME_PATH \
    --n_layers 36 \
    --behaviors $BEHAVIORS

CUDA_VISIBLE_DEVICES=1 python train/caa/prompt_with_steering.py \
    --behaviors $BEHAVIORS \
    --multipliers -2 -1 0 1 2\
    --type ab \
    --model_name_path $MODEL_NAME_PATH \
    --system_prompt pos

CUDA_VISIBLE_DEVICES=1 python train/caa/prompt_with_steering.py \
    --behaviors $BEHAVIORS \
    --multipliers -1 1\
    --layers 16 \
    --type $TYPE \
    --model_name_path $MODEL_NAME_PATH \
    --system_prompt $SYSTEM_PROMPT

CUDA_VISIBLE_DEVICES=1 python train/caa/plot.py \
    --behaviors $BEHAVIORS \
    --multipliers -2 -1 0 1 2 \
    --type $TYPE \
    --system_prompt $SYSTEM_PROMPT \
    --model_name_path $MODEL_NAME_PATH \

python train/caa/scoring.py \
    --behaviors $BEHAVIORS \
    --do_printing