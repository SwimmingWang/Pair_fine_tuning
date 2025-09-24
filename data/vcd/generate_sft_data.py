import json
import argparse
import os
import random

PREFERENCE_PAIRS = [
    {
        "Risk_Orientation": ["Risk-taking", "Risk-averse"],
        "Time Preference": ["Immediate gratification", "Delayed gratification"],
        "Communication Style": ["Assertive", "Accommodating"],
        "Social Strategy": ["Competitive", "Collaborative"],
    },
]

def main(args):
    with open(args.file_name, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    sft_data = []
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
        if scenario["preference domain"] == args.target_domain:
            sft_data.append(scenario)
            if scenario["bias"] == side_A_preference:
                side_A.append(scenario)
            else:
                side_B.append(scenario)
    random.shuffle(sft_data)
    random.shuffle(side_A)
    random.shuffle(side_B)
    if args.num_samples == -1:
        args.num_samples = len(sft_data)
    sft_data = sft_data[:args.num_samples]
    side_A = side_A[:args.num_samples]
    side_B = side_B[:args.num_samples]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.target_domain = args.target_domain.replace(" ", "_")
    if not os.path.exists(os.path.join(args.output_dir, args.target_domain)):
        os.makedirs(os.path.join(args.output_dir, args.target_domain))
    with open(os.path.join(args.output_dir, args.target_domain, f"all_{args.num_samples}.json"), "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=4)
    with open(os.path.join(args.output_dir, args.target_domain, f"{side_A_preference}_{args.num_samples}.json"), "w", encoding="utf-8") as f:
        json.dump(side_A, f, ensure_ascii=False, indent=4)
    with open(os.path.join(args.output_dir, args.target_domain, f"{side_B_preference}_{args.num_samples}.json"), "w", encoding="utf-8") as f:
        json.dump(side_B, f, ensure_ascii=False, indent=4)
    print(f"sft: {len(sft_data)}, side_A: {len(side_A)}, side_B: {len(side_B)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="data/vcd_sft.json")
    parser.add_argument("--target_domain", type=str, default="Time Preference")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="data/sft_data")
    args = parser.parse_args()
    main(args)