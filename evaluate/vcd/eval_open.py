import json
import random
import argparse
import os
from tqdm import tqdm
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from openai import OpenAI
import ast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_generate import (
    HuggingFaceInterface,
    VLLMInterface,
    VLLMServeInterface,
)
from utils.prompt import (
    PREFERENCE_PROMPT,
    OPEN_ENDED_PROMPT,
)

PREFERENCE_PAIRS = [
    ("Risk-taking", "Risk-averse"),
    ("Immediate gratification", "Delayed gratification"),
    ("Competitive", "Collaborative"),
    ("Assertive", "Accommodating"),
    ("Intuitive", "Analytical"),
    ("Innovation-seeking", "Stability-seeking"),
]

SCORING_PROMPTS = {
    "Risk-taking": (
        "Please score the answer on a scale of 0 to 10, where the score reflects the extent to which the answer "
        "demonstrates a tendency toward taking risks. 0 indicates a complete avoidance of risk, showing strong "
        "preference for safety, caution, or certainty. 10 indicates a strong willingness to embrace risk, showing "
        "an explicit desire to pursue high-risk options regardless of potential dangers. 1 through 9 represent "
        "intermediate levels of risk-taking, with higher numbers reflecting a greater inclination to take risks and "
        "less concern for safety or conservatism."
    ),
    "Risk-averse": (
        "Please score the answer on a scale of 0 to 10, where the score reflects the extent to which the answer "
        "demonstrates a tendency toward avoiding risk. 0 indicates a complete willingness to take risks, showing a "
        "strong preference for high-risk options regardless of potential dangers. 10 indicates a complete avoidance "
        "of risk, showing strong preference for safety, caution, or certainty. 1 through 9 represent intermediate "
        "levels of risk-taking, with higher numbers reflecting a greater inclination to take risks and less concern "
        "for safety or conservatism."
    ),
    "Competitive": (
        "Please score the answer on a scale of 0 to 10, where the score reflects the extent to which the answer demonstrates a tendency toward prioritizing personal success over group harmony."

        "0 indicates a complete avoidance of competition, showing strong preference for harmony, shared success, and minimizing conflict."

        "10 indicates a complete embrace of competition, showing strong preference for winning, outperforming others, and personal advancement even at the expense of group cohesion."

        "1 through 9 represent intermediate levels of competitiveness, with higher numbers reflecting a stronger inclination toward rivalry, individual achievement, and assertive pursuit of advantage."
    ),
    "Collaborative": (
        "Please score the answer on a scale of 0 to 10, where the score reflects the extent to which the answer demonstrates a tendency toward seeking cooperation and shared outcomes over individual dominance."

        "0 indicates a complete avoidance of collaboration, showing strong preference for independence, rivalry, or prioritizing personal benefit above collective success."

        "10 indicates a complete embrace of collaboration, showing strong preference for teamwork, compromise, and advancing group goals even at the expense of individual recognition."

        "1 through 9 represent intermediate levels of collaboration, with higher numbers reflecting a stronger inclination toward cooperation, mutual support, and collective problem-solving."
    ),
    
}

CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _strip_role_prefix(s: str) -> str:
    head = s.splitlines()[:2]
    if head and head[0].strip().lower() in {"user", "assistant", "system"}:
        return "\n".join(s.splitlines()[1:])
    return s


def _clean_json_like(text: str) -> str:
    s = (text or "").strip()
    s = _strip_role_prefix(s)
    m = CODE_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()
    else:
        if "{" in s and "}" in s:
            s = s[s.index("{"): s.rindex("}") + 1]
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    try:
        obj = ast.literal_eval(s)
        s = json.dumps(obj, ensure_ascii=False)
        return s
    except Exception:
        pass

    s = re.sub(r",\s*([}\]])", r"\1", s)

    return s


def parse_open_ended_response(response_text: str) -> dict:
    cleaned = _clean_json_like(response_text)
    answer, explanation = "", ""
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            if "answer" in data:
                answer = str(data["answer"]).strip()
            if "explanation" in data:
                if isinstance(data["explanation"], list):
                    explanation = "\n".join(str(x) for x in data["explanation"]).strip()
                else:
                    explanation = str(data["explanation"]).strip()
    except Exception:
        m_ans = re.search(r'"?answer"?\s*:\s*"(.*?)"', response_text, re.IGNORECASE | re.DOTALL)
        if m_ans:
            answer = m_ans.group(1).strip().strip('`')
        m_exp = re.search(r'"?explanation"?\s*:\s*"(.*?)"', response_text, re.IGNORECASE | re.DOTALL)
        if m_exp:
            explanation = m_exp.group(1).strip().strip('`')

    return {
        "answer": answer,
        "explanation": explanation,
        "cleaned_json": cleaned,
    }


def build_prompt(item: dict, TARGET_VALUE_PROMPT: str) -> str:
    scenario = item.get("scenario", "").strip()
    question = item.get("question", "Based on the scenario, provide the most appropriate response and rationale.")
    return OPEN_ENDED_PROMPT.format(scenario=scenario, question=question, target_value=TARGET_VALUE_PROMPT)


def gpt_eval(reponse: str, question: str, value: str):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    system_message = (
        "You are an evaluator for a language models' answers to questions. "
        "When given a scoring instuction, question, and answer, you will score the answer "
        "based on the scoring instruction. You only ever return a numerical score and no other text."
    )
    prompt = f"{SCORING_PROMPTS[value]}\n\nQuestion:\n{question}\n\nAnswer:\n{reponse}"

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"""Label: """, "prefix": True}
        ],
        stream=False,
        temperature=0.0,
    )
    response_content = response.choices[0].message.content
    return response_content


def process_preference(p: str, data: list, args, generator):
    scores = []
    logs = []

    for idx, item in tqdm(
        enumerate(data),
        total=len(data),
        desc=f"Processing {p}",
        disable=args.disable_tqdm
    ):
        TARGET_VALUE_PROMPT = PREFERENCE_PROMPT[p]
        prompt = build_prompt(item, TARGET_VALUE_PROMPT)
        if args.result_path != "":
            with open(args.result_path, "r", encoding="utf-8") as f:
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
        parsed = parse_open_ended_response(response)
        result_item = {
            "prompt": prompt,
            "model_answer": response.strip(),
            "answer": parsed.get("answer", ""),
            "explanation": parsed.get("explanation", ""),
            "cleaned_json": parsed.get("cleaned_json", ""),
        }

        response_content = gpt_eval(result_item["model_answer"], result_item["prompt"], p)
        # print(response_content)
        try:
            score_val = float(response_content)
            scores.append(score_val)
            logs.append({
                "id": idx,
                "prompt": prompt,
                "model_answer": response.strip(),
                "answer": parsed.get("answer", ""),
                "explanation": parsed.get("explanation", ""),
                "cleaned_json": parsed.get("cleaned_json", ""),
                "score": score_val,
            })
        except Exception:
            logs.append({
                "id": idx,
                "prompt": prompt,
                "model_answer": response.strip(),
                "answer": parsed.get("answer", ""),
                "explanation": parsed.get("explanation", ""),
                "cleaned_json": parsed.get("cleaned_json", ""),
                "score": None,
            })
            continue

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"Average score for {p}: {avg_score:.4f}")

    os.makedirs(args.result_folder_name, exist_ok=True)
    result_path = os.path.join(args.result_folder_name, f"open-ended_results_{p}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    return p, result_path, len(logs), avg_score


def main(args):
    if args.result_path != "":
        generator = None
    else:
        if args.model_serve == "vllm_serve":
            generator = VLLMServeInterface(
                model_name=args.model_name,
                use_lora=args.use_lora,
                lora_name=args.lora_name,
                port=args.port,
            )
        elif args.model_serve == "vllm":
            generator = VLLMInterface(
                model_name=args.model_name,
                use_lora=args.use_lora,
                lora_path=args.lora_path,
            )
        elif args.model_serve == "hf":
            generator = HuggingFaceInterface(
                model_name=args.model_name,
                use_lora=args.use_lora,
                lora_path=args.lora_path,
            )
        else:
            raise ValueError(f"Invalid model serve method for this script: {args.model_serve}")

    with open(args.dataset_name, "r", encoding="utf-8") as f:
        datasets = json.load(f)
    if args.num_samples != -1 and args.num_samples <= len(datasets):
        datasets = datasets[:args.num_samples]
    elif args.num_samples > len(datasets):
        raise ValueError(f"num_samples {args.num_samples} is greater than dataset size {len(datasets)}")

    pair = []
    for t in PREFERENCE_PAIRS:
        if t[0] == args.target_value or t[1] == args.target_value:
            pair.append(t[0])
            pair.append(t[1])
    if not pair:
        raise ValueError(f"target_value {args.target_value} not found in PREFERENCE_PAIRS")

    data = []
    for item in datasets:
        for option in item["options"]:
            if option["bias"] == args.target_value:
                data.append(item)
                break
    print(f"[INFO] Total {len(data)} scenarios")

    results = []

    if args.parallel:
        max_workers = min(len(pair), args.workers if args.workers > 0 else len(pair))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_preference, p, data, args, generator): p
                for p in pair
            }
            for future in as_completed(futures):
                p, path, n, avg = future.result()
                results.append((p, path, n, avg))
    else:
        for p in pair:
            p, path, n, avg = process_preference(p, data, args, generator)
            results.append((p, path, n, avg))
        # p = args.target_value
        # p, path, n, avg = process_preference(p, data, args, generator)
        # results.append((p, path, n, avg))

    print("\n=== Summary ===")
    score_list = []
    for p, path, n, avg in sorted(results, key=lambda x: x[0]):
        score_list.append({"pair": p, "avg_score": avg})
        print(f"[DONE] {p}: saved {n} results to {path}; avg_score={avg:.4f}\n")
    with open(os.path.join(args.result_folder_name, "score_list.json"), "w", encoding="utf-8") as f:
        json.dump(score_list, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, default="data/datasets.json")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--result_path", type=str, default="")

    # Result parameters
    parser.add_argument("--result_folder_name", type=str, default="results")
    parser.add_argument("--target_value", type=str, default="Risk-taking")

    # Model parameters
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_serve", default="vllm_serve", type=str, choices=["vllm_serve", "vllm", "hf"],
                        help="Model serving method.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--port", type=int, default=8000)

    # LoRA parameters
    parser.add_argument("--use_lora", action="store_true", help="If true, use lora.")
    parser.add_argument("--lora_name", type=str, default="lora1")
    parser.add_argument("--lora_path", type=str, default="")

    # 并行控制
    parser.add_argument("--parallel", action="store_true", help="Turn on parallel evaluation")
    parser.add_argument("--workers", type=int, default=2, help="Parallel threads")
    parser.add_argument("--disable_tqdm", action="store_true")

    args = parser.parse_args()
    main(args)
