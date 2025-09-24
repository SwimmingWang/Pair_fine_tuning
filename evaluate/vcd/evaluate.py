import json
import random
import argparse
import os
from tqdm import tqdm
import sys
import re
from concurrent.futures import ThreadPoolExecutor
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_generate import (
    HuggingFaceInterface,
    VLLMInterface,
    VLLMServeInterface,
)
from utils.prompt.prompt import (
    PREFERENCE_PROMPT,
    main_prompt_select_one,
    main_prompt_select_all,
)

PREFERENCE_PAIRS = [
    ("Risk-taking", "Risk-averse"),
    ("Immediate gratification", "Delayed gratification"),
    ("Competitive", "Collaborative"),
    ("Assertive", "Accommodating"),
    ("Intuitive", "Analytical"),
    ("Innovation-seeking", "Stability-seeking"),
]

import ast

CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

def _strip_role_prefix(s: str) -> str:
    # 去掉类似 'user\n' 或 'assistant\n' 这种单独一行的前缀
    head = s.splitlines()[:2]
    if head and head[0].strip().lower() in {"user", "assistant", "system"}:
        return "\n".join(s.splitlines()[1:])
    return s

def _clean_json_like(text: str) -> str:
    """
    将“看起来像 JSON 但不太规范”的字符串清洗成尽量可被 json.loads 接受的形式。
    处理要点：
    - 去掉代码围栏
    - 替换智能引号为普通引号
    - 给 choice: E / choices: [A, C] 这种裸字母补引号
    - 尝试把单引号 JSON 转成双引号
    - 去掉对象/数组中的尾随逗号
    """
    s = text.strip()

    # 先去掉单行的 role 前缀
    s = _strip_role_prefix(s)

    # 如果有 ```json ... ``` 或 ``` ... ```，优先取其中的 {...}
    m = CODE_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()
    else:
        # 没有代码块时，截取第一个 '{' 到最后一个 '}' 之间的内容
        if "{" in s and "}" in s:
            s = s[s.index("{"): s.rindex("}") + 1]

    # 标准化引号：智能引号 -> 普通引号
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # 给 "choice": E 补引号（只限单个大写字母 A-E）
    s = re.sub(r'("choice"\s*:\s*)([A-E])(\s*[,}])', r'\1"\2"\3', s)

    # 给 "choices": [A, B, C] 补引号
    def _quote_array_letters(match):
        inner = match.group(1)
        # 用逗号切分再逐个加引号（仅对单个 A-E）
        parts = re.split(r'(\s*,\s*)', inner)
        for i in range(0, len(parts), 2):
            token = parts[i].strip()
            if re.fullmatch(r"[A-E]", token):
                parts[i] = f'"{token}"'
        return '["' + '"]'  # 无用占位，为了保证下面替换逻辑执行
    # 更直接一些：对数组中裸字母整体替换
    s = re.sub(r'("choices"\s*:\s*\[)([A-E](?:\s*,\s*[A-E])*)(\])',
               lambda m: m.group(1) + ", ".join(f'"{t.strip()}"' for t in m.group(2).split(",")) + m.group(3),
               s)

    # 尝试把形如 {'choice': 'A'} 的单引号 JSON 转为双引号（先用 ast 尝试）
    try:
        # 若本身已是合法 JSON，将在后面直接 loads
        # 若是 python 风格字典（单引号），ast.literal_eval 能解析
        obj = ast.literal_eval(s)
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        # 去掉对象/数组内部可能的尾随逗号（宽松处理）
        s = re.sub(r",\s*([}\]])", r"\1", s)

    return s


def parse_answer(response_text: str) -> dict:
    """
    返回一个 dict，至少包含：
      - 'choice': 可能为空字符串
      - 'choices': 可能为空列表
      - 'explanation': 字符串
    """
    cleaned = _clean_json_like(response_text)
    try:
        data = json.loads(cleaned)
    except Exception:
        # 彻底失败就用正则兜底从原文里抓 choice/choices
        data = {}

    # 规范化字段
    choice = ""
    explanation = ""
    choices_multi = []

    if isinstance(data, dict):
        # 单选
        if "choice" in data and isinstance(data["choice"], str):
            choice = data["choice"].strip().upper()
        # 多选
        if "choices" in data and isinstance(data["choices"], list):
            choices_multi = [str(c).strip().upper() for c in data["choices"] if str(c).strip()]
        # 说明
        if "explanation" in data:
            if isinstance(data["explanation"], list):
                explanation = "\n".join(str(e) for e in data["explanation"]).strip()
            else:
                explanation = str(data["explanation"]).strip()

    # 兜底：从原文本直接抓
    if not choice and not choices_multi:
        m = re.search(r'"?choice"?\s*:\s*([A-E])', response_text, re.IGNORECASE)
        if m:
            choice = m.group(1).upper()
        else:
            m2 = re.search(r'"?choices"?\s*:\s*\[([^\]]+)\]', response_text, re.IGNORECASE)
            if m2:
                letters = re.findall(r'[A-E]', m2.group(1).upper())
                choices_multi = letters

    # 兜底解释
    if not explanation:
        # 尝试抓 “explanation: .....” 后面的一段文本
        mexp = re.search(r'"?explanation"?\s*:\s*(.+)', response_text, re.IGNORECASE | re.DOTALL)
        if mexp:
            explanation = mexp.group(1).strip().strip('`').strip()

    return {
        "choice": choice,
        "choices": choices_multi,
        "explanation": explanation,
        "cleaned_json": cleaned,  # 调试用
    }


def evaluate(args, generator):
    TARGET_VALUE = args.target_value

    with open(args.dataset_name, "r", encoding="utf-8") as f:
        datasets = json.load(f)

    if args.num_samples != -1 and args.num_samples <= len(datasets):
        datasets = datasets[:args.num_samples]
    elif args.num_samples > len(datasets):
        raise ValueError(f"num_samples {args.num_samples} is greater than dataset size {len(datasets)}")
    
    print(f"[INFO] Evaluating {args.target_value}")
    data = []
    for item in datasets:
        for option in item["options"]:
            if option["bias"] == TARGET_VALUE:
                data.append(item)
                break
    print(f"[INFO] Total {len(data)} scenarios")
    

    correct = 0
    total = 0
    logs = []
    micro_tp = micro_fp = micro_fn = 0

    for idx, item in tqdm(enumerate(data), total=len(data), desc=f"Processing {args.target_value}"):
        scenario = item["scenario"]

        TARGET_VALUE_PROMPT = PREFERENCE_PROMPT[TARGET_VALUE]
        main_prompt = main_prompt_select_one if args.select_one else main_prompt_select_all

        options_text = "\n".join([f"{chr(65 + i)}. {c['text']}" for i, c in enumerate(item["options"])])
        prompt = main_prompt.format(
            TARGET_VALUE=TARGET_VALUE,
            TARGET_VALUE_PROMPT=TARGET_VALUE_PROMPT,
            scenario=scenario,
            options_text=options_text
        )
        if args.result_path != "":
            with open(args.result_path, "r") as f:
                responses = json.load(f)
            response = responses[idx]["model_output"]
        else:
            response = generator.generate(
                prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        # print(response)
        parsed = parse_answer(response)
        choice = parsed["choice"]
        explanation = parsed["explanation"]
        choices_multi = parsed["choices"]
        answer_json_str = parsed["cleaned_json"]  # 记录清洗后的 JSON，便于排查

        if not choice and not choices_multi:
            # 仍解析失败，写日志并跳过
            log_item = {
                "id": idx,
                "prompt": prompt,
                "model_answer": response.strip(),
                "choice": "",
                "explanation": f"Failed to parse. cleaned: {answer_json_str}",
                "target": TARGET_VALUE,
            }
            logs.append(log_item)
            total += 1  # 增加总数计数
            continue

        choices = item["options"]
        selected_values = []
        is_correct = False
        score = 0

        if args.select_one:
            # 单选题逻辑
            if choice and choice[0] in "ABCDE":
                selected_idx = ord(choice[0]) - 65
                if 0 <= selected_idx < len(choices):
                    selected_values = [choices[selected_idx].get("bias", [])]
                    is_correct = TARGET_VALUE in selected_values
                    score = int(is_correct)
                    correct += score
        else:
            # 多选逻辑 - 使用已解析的 choices_multi
            gt_indices = [i for i, opt in enumerate(choices) if opt["bias"] == TARGET_VALUE]
            gt_set = set(gt_indices)
            selected_indices = []
            for ch in choices_multi:
                ch = ch.strip().upper()
                if ch and ch[0] in "ABCDE":
                    idx = ord(ch[0]) - 65
                    if 0 <= idx < len(choices):
                        selected_indices.append(idx)
            selected_set = set(selected_indices)
            choice = "".join([chr(65 + i) for i in selected_indices])
            selected_values = [choices[i].get("bias", []) for i in selected_indices]
            if selected_set == gt_set:
                score = 1
            elif not selected_set.issubset(gt_set):
                score = 0
            else:
                score = len(selected_set & gt_set) / len(gt_set) if gt_set else 0
            is_correct = score == 1
            correct += score

            tp = len(selected_set & gt_set)
            fp = len(selected_set - gt_set)
            fn = len(gt_set - selected_set)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            # micro 累计
            micro_tp += tp
            micro_fp += fp
            micro_fn += fn

        log_item = {
            "id": idx,
            "prompt": prompt,
            "model_answer": answer_json_str,
            "choice": choice,
            "explanation": explanation,
            "selected_values": selected_values,
            "correct": is_correct,
            "score": score,
            "target": TARGET_VALUE,
        }
        if not args.select_one:
            log_item["precision"] = precision
            log_item["recall"] = recall
            log_item["f1"] = f1
        total += 1  # 增加总数计数
        if args.save_full_log:
            log_item["full_response"] = response
        logs.append(log_item)

    accuracy = correct / total if total else 0
    print(f"[RESULT] correct: {correct}, total: {total}, \n[RESULT] Accuracy for value '{TARGET_VALUE}': {accuracy:.2%}")

    metrics_block = {}
    if not args.select_one:
        micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
        micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
        micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

        print(f"[RESULT] Micro P/R/F1: {micro_precision:.4f}/{micro_recall:.4f}/{micro_f1:.4f}")

        metrics_block = {
            "micro_precision": round(micro_precision, 6),
            "micro_recall": round(micro_recall, 6),
            "micro_f1": round(micro_f1, 6),
        }
    
    os.makedirs(args.result_folder_name, exist_ok=True)
    mode = "selectone" if args.select_one else "selectall"
    result_path = os.path.join(args.result_folder_name, f"eval_result_{TARGET_VALUE}_{mode}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    overall_logs = {
        "target_value": TARGET_VALUE,
        "correct": correct,
        "total": total,
        "accuracy": f"{accuracy:.2%}",
        "mode": "selectone" if args.select_one else "selectall",
    }
    if metrics_block:
        overall_logs.update(metrics_block)

    result_all_path = os.path.join(args.result_folder_name, "eval_result_all.json")
    existing_results = []
    if os.path.exists(result_all_path):
        with open(result_all_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)

    if isinstance(existing_results, list):
        existing_results.append(overall_logs)
    else:
        existing_results = [existing_results, overall_logs]

    with open(result_all_path, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument("--target_value", type=str, default="Delayed gratification")
    parser.add_argument("--dataset_name", type=str, default="data/datasets.json")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--result_path", type=str, default="")

    # Result parameters
    parser.add_argument("--result_folder_name", type=str, default="results")
    parser.add_argument("--select_one", action="store_true", help="If true, select only one best choice.")
    parser.add_argument("--save_full_log", action="store_true", help="If true, save full model output in logs.")

    # Model parameters
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model_serve", default="vllm_serve", type=str, choices=["vllm_serve", "vllm", "hf"], help="Model serving method.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--port", type=int, default=8000)

    # LoRA parameters
    parser.add_argument("--use_lora", action="store_true", help="If true, use lora.")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--lora_name", type=str, default="lora1")
    parser.add_argument("--parallel", action="store_true", help="If true, evaluate in parallel.")

    args = parser.parse_args()

    if args.result_path != "":
        pass
    else:
        if args.model_serve == "vllm_serve":
            generator = VLLMServeInterface(model_name=args.model_name, use_lora=args.use_lora, lora_name=args.lora_name, port=args.port)
        # TODO: add other model serve methods
        elif args.model_serve == "vllm":
            generator = VLLMInterface(model_name=args.model_name, use_lora=args.use_lora, lora_path=args.lora_path)
        elif args.model_serve == "hf":
            generator = HuggingFaceInterface(model_name=args.model_name, use_lora=args.use_lora, lora_path=args.lora_path)
        else:
            raise ValueError(f"Invalid model serve method: {args.model_serve}")

    if args.target_value == "all":
        for target_value in PREFERENCE_PAIRS:
            args.target_value = target_value[0]
            evaluate(args, generator)
            args.target_value = target_value[1]
            evaluate(args, generator)
        exit()
    else:
        args.target_value = args.target_value.replace("_", " ")
        pair = next(((a, b) for a, b in PREFERENCE_PAIRS
                if a == args.target_value or b == args.target_value), None)
        if pair is None:
            raise ValueError(f"Invalid target value: {args.target_value}")

        a, b = pair
        if args.parallel:
            args_a, args_b = copy.deepcopy(args), copy.deepcopy(args)
            args_a.target_value, args_b.target_value = a, b

            with ThreadPoolExecutor(max_workers=2) as ex:
                ex.submit(evaluate, args_a, generator)
                ex.submit(evaluate, args_b, generator)
            exit()
        else:
            args.target_value = a
            evaluate(args, generator)
            args.target_value = b
            evaluate(args, generator)