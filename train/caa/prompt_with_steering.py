import json
import sys
from model_wrapper import LlamaWrapper
import os
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
from helpers import get_a_b_probs
from tokenizer import E_INST
from steering_settings import SteeringSettings
from behaviors import (
    get_open_ended_test_data,
    get_steering_vector,
    get_system_prompt,
    get_truthful_qa_data,
    get_mmlu_data,
    get_ab_test_data,
    ALL_BEHAVIORS,
    get_results_dir,
)

PREFERENCE_PAIRS = {
    "Risk-Orientation": ["Risk-taking", "Risk-averse"],
    "Delay-Orientation": ["Immediate gratification", "Delayed gratification"],
    "Competition-Orientation": ["Competitive", "Collaborative"],
    "Assertiveness-Orientation": ["Assertive", "Accommodating"],
    "Intuition-Orientation": ["Intuitive", "Analytical"],
    "Innovation-Orientation": ["Innovation-seeking", "Stability-seeking"],
}

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.prompt import (
    PREFERENCE_PROMPT,
    main_prompt_select_one,
    main_prompt_select_all,
    OPEN_ENDED_PROMPT,
)

load_dotenv()

def process_item_ab(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    question: str = item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]
    model_output = model.get_logits_from_text(
        user_input=question, model_output="(", system_prompt=system_prompt
    )
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
    }

def process_item_open_ended(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    question = item["question"]
    model_output = model.generate_text(
        user_input=question, system_prompt=system_prompt, max_new_tokens=512
    )
    return {
        "question": question,
        "model_output": model_output.split(E_INST)[-1].strip(),
        "raw_model_output": model_output,
    }


def process_item_tqa_mmlu(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    prompt = item["prompt"]
    correct = item["correct"]
    incorrect = item["incorrect"]
    category = item["category"]
    model_output = model.get_logits_from_text(
        user_input=prompt, model_output="(", system_prompt=system_prompt
    )
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": prompt,
        "correct": correct,
        "incorrect": incorrect,
        "a_prob": a_prob,
        "b_prob": b_prob,
        "category": category,
    }

def process_item_vcd(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: str,
    behavior: str,
    args: Optional[argparse.Namespace],
) -> Dict[str, str]:
    if system_prompt == "pos":
        TARGET_VALUE = PREFERENCE_PAIRS[behavior][0]
    else:
        TARGET_VALUE = PREFERENCE_PAIRS[behavior][1]

    TARGET_VALUE_PROMPT = PREFERENCE_PROMPT[TARGET_VALUE]
    if args.type == "ab":
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
        model_output = model.generate_text(
            user_input=prompt, max_new_tokens=512
        )
        return {
            "question": prompt,
            "model_output": model_output.split(E_INST)[-1].strip(),
            "raw_model_output": model_output,
        }
    elif args.type == "open_ended":
        scenario = item.get("scenario", "").strip()
        question = item.get("question", "Based on the scenario, provide the most appropriate response and rationale.")
        prompt = OPEN_ENDED_PROMPT.format(scenario=scenario, question=question, target_value=TARGET_VALUE_PROMPT)
        model_output = model.generate_text(
            user_input=prompt, max_new_tokens=512
        )
        return {
            "question": prompt,
            "model_output": model_output.split(E_INST)[-1].strip(),
            "raw_model_output": model_output,
        }

def get_vcd_data(behavior: str):
    behavior = behavior.replace("-", " ")
    with open(f"data/vcd/vcd_test.json", "r") as f:
        data = json.load(f)
    vcd_data = []
    for item in data:
        if item["preference domain"] == behavior:
            vcd_data.append(item)
    return vcd_data

def test_steering(
    layers: List[int], multipliers: List[int], settings: SteeringSettings, overwrite=False, args=None
):
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    settings: SteeringSettings object.
    """
    save_results_dir = get_results_dir(settings.behavior)
    if settings.dataset == "vcd":
        if args.type == "ab":
            select_one_str = "_select_one" if args.select_one else "_select_all"
            save_results_dir = f"{save_results_dir}_caa_{settings.system_prompt}{select_one_str}_"
        elif args.type == "open_ended":
            save_results_dir = f"{save_results_dir}_caa_{settings.system_prompt}_open_ended"
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    process_methods = {
        "ab": process_item_ab,
        "open_ended": process_item_open_ended,
        "truthful_qa": process_item_tqa_mmlu,
        "mmlu": process_item_tqa_mmlu,
        "vcd": process_item_vcd,
    }
    model = LlamaWrapper(
        model_name_path=settings.model_name_path,
        use_chat=not settings.use_base_model,
        override_model_weights_path=settings.override_model_weights_path,
    )
    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")
    model.set_save_internal_decodings(False)
    
    if settings.dataset == "vcd":
        test_data = get_vcd_data(settings.behavior)
    else:
        test_datasets = {
            "ab": get_ab_test_data(settings.behavior),
            "open_ended": get_open_ended_test_data(settings.behavior),
            "truthful_qa": get_truthful_qa_data(),
            "mmlu": get_mmlu_data(),
        }
        test_data = test_datasets[settings.type]
    for layer in layers:
        name_path = model.model_name_path
        if settings.override_vector_model is not None:
            name_path = settings.override_vector_model
        if settings.override_vector is not None:
            vector = get_steering_vector(settings.behavior, settings.override_vector, name_path, normalized=True)
        else:
            vector = get_steering_vector(settings.behavior, layer, name_path, normalized=True)
        vector = vector.to(model.device)
        for multiplier in multipliers:
            result_save_suffix = settings.make_result_save_suffix(
                layer=layer, multiplier=multiplier
            )
            save_filename = os.path.join(
                save_results_dir,
                f"results_{result_save_suffix}.json",
            )
            if os.path.exists(save_filename) and not overwrite:
                print("Found existing", save_filename, "- skipping")
                continue
            results = []
            for item in tqdm(test_data, desc=f"Layer {layer}, multiplier {multiplier}"):
                model.reset_all()
                model.set_add_activations(
                    layer, multiplier * vector
                )
                if settings.dataset == "vcd":
                    result = process_item_vcd(
                        item=item,
                        model=model,
                        system_prompt=settings.system_prompt,
                        behavior=settings.behavior,
                        args=args,
                    )
                    results.append(result)
                else:
                    result = process_methods[settings.type](
                        item=item,
                        model=model,
                        system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
                        a_token_id=a_token_id,
                        b_token_id=b_token_id,
                    )
                    results.append(result)
            with open(
                save_filename,
                "w",
            ) as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(36)))
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument(
        "--behaviors",
        type=str,
        nargs="+",
        default=ALL_BEHAVIORS
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["ab", "open_ended", "truthful_qa", "mmlu"],
    )
    parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"], required=False)
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_name_path", type=str, required=True)
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="vcd", choices=["vcd", "bqd"])
    parser.add_argument("--select_one", action="store_true", default=False)
    
    args = parser.parse_args()

    steering_settings = SteeringSettings()
    steering_settings.type = args.type
    steering_settings.system_prompt = args.system_prompt
    steering_settings.override_vector = args.override_vector
    steering_settings.override_vector_model = args.override_vector_model
    steering_settings.use_base_model = args.use_base_model
    steering_settings.model_name_path = args.model_name_path
    steering_settings.override_model_weights_path = args.override_model_weights_path
    steering_settings.dataset = args.dataset

    for behavior in args.behaviors:
        steering_settings.behavior = behavior
        test_steering(
            layers=args.layers,
            multipliers=args.multipliers,
            settings=steering_settings,
            overwrite=args.overwrite,
            args=args,
        )