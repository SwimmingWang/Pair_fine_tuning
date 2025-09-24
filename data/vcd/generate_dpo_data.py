import json
import argparse
import os
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.prompt.prompt import PREFERENCE_PROMPT, main_prompt_select_one
PREFERENCE_PAIRS = [
    {
        "Risk_Orientation": ["Risk-taking", "Risk-averse"],
        "Time Preference": ["Immediate gratification", "Delayed gratification"],
        "Communication Style": ["Assertive", "Accommodating"],
        "Social Strategy": ["Competitive", "Collaborative"],
    },
]
answer_prompt = """
```json
{{
    "choice": {choice},
    "explanation": {explanation}
}}
```
"""

def main(args):
    with open(args.file_name, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    dpo_data = []
    side_A = []
    side_B = []
    if args.target_domain == "Risk Orientation":
        side_A_preference = "Risk-taking"
        side_B_preference = "Risk-averse"
    elif args.target_domain == "Time Preference":
        side_A_preference = "Immediate gratification"
        side_B_preference = "Delayed gratification"
    elif args.target_domain == "Communication Style":
        side_A_preference = "Assertive"
        side_B_preference = "Accommodating"
    elif args.target_domain == "Social Strategy":
        side_A_preference = "Competitive"
        side_B_preference = "Collaborative"
    for scenario in scenarios:
        if scenario["preference_domain"] == args.target_domain:
            side_A_list = []
            side_B_list = []
            for option in scenario["options"]:
                if option["bias"] == side_A_preference:
                    side_A_list.append(option)
                else:
                    side_B_list.append(option)
            options_text = "\n".join([f"{chr(65 + i)}. {c['option_text']}" for i, c in enumerate(scenario["options"])])
            for i in range(len(side_A_list)):
                for j in range(len(side_B_list)):
                    side_A.append({
                        "prompt": main_prompt_select_one.format(scenario=scenario["scenario"], options_text=options_text, TARGET_VALUE=side_A_preference, TARGET_VALUE_PROMPT=PREFERENCE_PROMPT[side_A_preference]),
                        "chosen": answer_prompt.format(choice=side_A_list[i]["option"], explanation=side_A_list[i]["explanation"]),
                        "rejected": answer_prompt.format(choice=side_B_list[j]["option"], explanation=side_B_list[j]["explanation"])
                    })
                    side_B.append({
                        "prompt": main_prompt_select_one.format(scenario=scenario["scenario"], options_text=options_text, TARGET_VALUE=side_B_preference, TARGET_VALUE_PROMPT=PREFERENCE_PROMPT[side_B_preference]),
                        "chosen": answer_prompt.format(choice=side_B_list[j]["option"], explanation=side_B_list[j]["explanation"]),
                        "rejected": answer_prompt.format(choice=side_A_list[i]["option"], explanation=side_A_list[i]["explanation"])
                    })
                    dpo_data.append(side_A[-1])
                    dpo_data.append(side_B[-1])
    random.shuffle(dpo_data)
    random.shuffle(side_A)
    random.shuffle(side_B)
    if args.num_samples == -1:
        args.num_samples = len(dpo_data)
    
    dpo_data = dpo_data[:args.num_samples]
    side_A = side_A[:args.num_samples]
    side_B = side_B[:args.num_samples]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.target_domain = args.target_domain.replace(" ", "_")
    if not os.path.exists(os.path.join(args.output_dir, args.target_domain)):
        os.makedirs(os.path.join(args.output_dir, args.target_domain))
    with open(os.path.join(args.output_dir, args.target_domain, f"all_{args.num_samples}.json"), "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=4)
    with open(os.path.join(args.output_dir, args.target_domain, f"{side_A_preference}_{args.num_samples}.json"), "w", encoding="utf-8") as f:
        json.dump(side_A, f, ensure_ascii=False, indent=4)
    with open(os.path.join(args.output_dir, args.target_domain, f"{side_B_preference}_{args.num_samples}.json"), "w", encoding="utf-8") as f:
        json.dump(side_B, f, ensure_ascii=False, indent=4)
    print(f"dpo: {len(dpo_data)}, side_A: {len(side_A)}, side_B: {len(side_B)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="data/vcd_train.json")
    parser.add_argument("--target_domain", type=str, default="Time Preference")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="data/dpo_data")
    args = parser.parse_args()
    main(args)