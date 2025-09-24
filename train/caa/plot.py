#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dump consolidated evaluation results to stdout + TSV/LaTeX (no plotting).

Example:
python dump_results.py --layers $(seq 0 31) --multipliers -1 0 1 --type ab \
  --model_name_path llama-3.1-8b --behaviors kindness honesty
"""

import json
import os
import argparse
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from steering_settings import SteeringSettings
from behaviors import ANALYSIS_PATH, HUMAN_NAMES, get_results_dir, get_analysis_dir, ALL_BEHAVIORS

# ---------------------------
# Utilities for loading & stats
# ---------------------------
def _open_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def get_data(layer: int, multiplier: float, settings: SteeringSettings):
    directory = get_results_dir(settings.behavior)
    if settings.type == "open_ended":
        directory = directory.replace("results", os.path.join("results", "open_ended_scores"))
    filenames = settings.filter_result_files_by_suffix(directory, layer=layer, multiplier=multiplier)
    if len(filenames) == 0:
        print(f"[WARN] no file for {settings.behavior} layer={layer} mult={multiplier} {settings}")
        return []
    if len(filenames) > 1:
        print(f"[WARN] >1 file for {settings.behavior} layer={layer} mult={multiplier} {settings}: {filenames}")
    return _open_json(filenames[0])

def get_avg_score(results: List[Dict[str, Any]]) -> float:
    score_sum, tot = 0.0, 0
    for r in results:
        try:
            score_sum += float(r["score"])
            tot += 1
        except Exception:
            pass
    return 0.0 if tot == 0 else score_sum / tot

def get_avg_key_prob(results: List[Dict[str, Any]], key: str) -> float:
    if not results:
        return 0.0
    acc = 0.0
    for r in results:
        try:
            match = r[key]
            a_prob, b_prob = float(r["a_prob"]), float(r["b_prob"])
            denom = a_prob + b_prob if (a_prob + b_prob) != 0 else 1e-9
            if "A" in match:
                acc += a_prob / denom
            elif "B" in match:
                acc += b_prob / denom
        except Exception:
            pass
    return acc / len(results)

def metric_label(eval_type: str) -> str:
    if eval_type == "open_ended":
        return "mean_score_0to10"
    elif eval_type in ("ab", "mmlu", "truthful_qa"):
        return "p_match_or_correct"
    else:
        return "metric"

def compute_metric(settings: SteeringSettings, results: List[Dict[str, Any]], system_prompt: str) -> float:
    if settings.type == "open_ended":
        return get_avg_score(results)
    elif settings.type == "ab":
        if system_prompt == "pos":
            return get_avg_key_prob(results, "answer_matching_behavior")
        elif system_prompt == "neg":
            return get_avg_key_prob(results, "answer_not_matching_behavior")
        else:
            raise ValueError(f"Unsupported system prompt: {system_prompt}")
    elif settings.type in ("mmlu", "truthful_qa"):
        return get_avg_key_prob(results, "correct")
    else:
        raise ValueError(f"Unsupported eval type: {settings.type}")

# ---------------------------
# Dumpers
# ---------------------------
def print_table(header: List[str], rows: List[List[str]]):
    # Simple aligned text table
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    def fmt_line(vals):
        return " | ".join(str(v).ljust(w) for v, w in zip(vals, widths))
    sep = "-+-".join("-" * w for w in widths)
    print(fmt_line(header))
    print(sep)
    for r in rows:
        print(fmt_line(r))
    print()

def write_tsv(path: str, header: List[str], rows: List[List[str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")

def write_latex_table(path: str, caption: str, header: List[str], rows: List[List[str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{" + " ".join(["l"] + ["r"] * (len(header)-1)) + "}\n\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(map(str, r)) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write("\\label{tab:" + os.path.splitext(os.path.basename(path))[0] + "}\n")
        f.write("\\end{table}\n")

# ---------------------------
# Core: dump results
# ---------------------------
def dump_per_behavior(settings: SteeringSettings, layers: List[int], multipliers: List[float]) -> Tuple[List[str], List[List[str]]]:
    """
    Returns header, rows for one behavior across layers x multipliers.
    Row = (layer, multiplier, metric)
    """
    rows = []
    for layer in sorted(layers):
        for mult in sorted(multipliers):
            res = get_data(layer, mult, settings)
            val = compute_metric(settings, res, settings.system_prompt)
            # pretty print for prob metrics
            if settings.type == "open_ended":
                val_str = f"{val:.2f}"
            else:
                val_str = f"{val:.3f}"
            rows.append([str(layer), str(mult), val_str])
    header = ["layer", "multiplier", metric_label(settings.type)]
    return header, rows

def dump_per_layer_summary(settings: SteeringSettings, layer: int, multipliers: List[float]) -> Tuple[List[str], List[List[str]]]:
    """
    For a single layer: one row per multiplier; useful for quick view.
    """
    rows = []
    for mult in sorted(multipliers):
        res = get_data(layer, mult, settings)
        val = compute_metric(settings, res)
        rows.append([str(mult), f"{val:.3f}" if settings.type != "open_ended" else f"{val:.2f}"])
    header = ["multiplier", metric_label(settings.type)]
    return header, rows

def dump_category_breakdown_if_applicable(settings: SteeringSettings, layer: int, multipliers: List[float]):
    """
    For truthful_qa/mmlu with 'category' in results, dump per-category rows (multiplier x metric).
    """
    if settings.type not in ("mmlu", "truthful_qa"):
        return None

    # Build category => {multiplier: metric}
    cat_map = defaultdict(dict)
    cats = set()
    for mult in sorted(multipliers):
        res = get_data(layer, mult, settings)
        # group by category
        by_cat = defaultdict(list)
        for it in res:
            if "category" in it:
                by_cat[it["category"]].append(it)
        for c, items in by_cat.items():
            cats.add(c)
            cat_map[c][mult] = compute_metric(settings, items)

    cats = sorted(list(cats))
    header = ["category"] + [f"mult={m}" for m in sorted(multipliers)]
    rows = []
    for c in cats:
        row = [c]
        for m in sorted(multipliers):
            v = cat_map[c].get(m, None)
            row.append(f"{v:.3f}" if v is not None and settings.type != "open_ended" else ("-" if v is None else f"{v:.2f}"))
        rows.append(row)
    return header, rows

# ---------------------------
# Arg handling / entry
# ---------------------------
def steering_settings_from_args(args, behavior: str) -> SteeringSettings:
    s = SteeringSettings()
    s.type = args.type
    s.behavior = behavior
    s.override_vector = args.override_vector
    s.override_vector_model = args.override_vector_model
    s.system_prompt = args.system_prompt
    s.use_base_model = args.use_base_model
    s.model_name_path = args.model_name_path
    if len(args.override_weights) > 0:
        s.override_model_weights_path = args.override_weights[0]
    return s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(36)))
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--behaviors", type=str, nargs="+", default=ALL_BEHAVIORS)
    parser.add_argument("--type", type=str, default="ab", choices=["ab", "open_ended", "truthful_qa", "mmlu"])
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_name_path", type=str, required=True)
    parser.add_argument("--override_weights", type=str, nargs="+", default=[])
    parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"])
    args = parser.parse_args()

    # Run
    for behavior in args.behaviors:
        settings = steering_settings_from_args(args, behavior)
        save_dir = get_analysis_dir(behavior)
        os.makedirs(save_dir, exist_ok=True)

        print("=" * 80)
        print(f"[Behavior] {behavior} ({HUMAN_NAMES.get(behavior, behavior)}) | model={settings.model_name_path} | type={settings.type}")
        print("=" * 80)

        # (A) Full dump rows (layer x multiplier)
        header, rows = dump_per_behavior(settings, args.layers, args.multipliers)
        print_table(header, rows)
        tsv_path = os.path.join(save_dir, f"DUMP_{settings.type}_{behavior}_{args.system_prompt}.tsv")
        write_tsv(tsv_path, header, rows)

        # (B) If only one layer specified, print concise summary
        if len(args.layers) == 1:
            h2, r2 = dump_per_layer_summary(settings, args.layers[0], args.multipliers)
            print(f"--- Summary @ layer={args.layers[0]} ---")
            print_table(h2, r2)
            write_tsv(os.path.join(save_dir, f"SUMMARY_layer{args.layers[0]}_{settings.type}_{behavior}_{args.system_prompt}.tsv"), h2, r2)

        # (C) Category breakdown for MMLU/TruthfulQA (if categories exist)
        cat_dump = dump_category_breakdown_if_applicable(settings, args.layers[0], args.multipliers) if len(args.layers) >= 1 else None
        if cat_dump is not None:
            h3, r3 = cat_dump
            print("--- Per-category breakdown (if available) ---")
            print_table(h3, r3)
            write_tsv(os.path.join(save_dir, f"CATEGORY_{settings.type}_{behavior}_{args.system_prompt}.tsv"), h3, r3)

        print(f"[Saved] {tsv_path}")
        print()

if __name__ == "__main__":
    main()
