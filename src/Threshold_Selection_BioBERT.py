# Threshold_Selection_BioBERT.py

"""
Threshold Optimization for BioBERT Triple Matching (Full Triple Embedding)
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 30/07/2025

Description:
    Evaluates cosine similarity thresholds to align GPT-predicted full semantic triples
    (subject–predicate–object) with CBM gold standard using BioBERT. Full triples are compared
    as single normalized text strings.

Usage:
    python src/Threshold_Selection_BioBERT.py --gold <gold_file_path> --eval <eval_file_path>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import re
from transformers import AutoTokenizer, AutoModel

# === Load BioBERT ===
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# === Helper Functions ===

def normalize(text):
    if pd.isna(text):
        return ""
    text = text.lower().replace('_', ' ').replace('-', ' ')
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

def format_triple(s, p, o):
    return f"{normalize(s)} {normalize(p)} {normalize(o)}"

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def group_full_triples(df):
    grouped = {}
    for _, row in df.iterrows():
        if all(pd.notna([row['Subject'], row['Predicate'], row['Object']])):
            triple = format_triple(row['Subject'], row['Predicate'], row['Object'])
            grouped.setdefault(row['Image_number'], []).append(triple)
    return grouped

def compare_full_triples(gold_dict, eval_dict, threshold):
    TP = FP = FN = 0

    for image_id in sorted(set(gold_dict) & set(eval_dict), key=lambda x: int(x.split('_')[-1])):
        gold_triples = gold_dict[image_id]
        eval_triples = eval_dict[image_id]
        matched_gold = set()

        print(f"\n--- Comparing triples for image: {image_id} ---")

        for idx_pred, pred in enumerate(eval_triples):
            emb_pred = get_embedding(pred)
            best_score = 0
            best_idx = None
            best_gold = ""

            for idx_gold, gold in enumerate(gold_triples):
                if idx_gold in matched_gold:
                    continue
                emb_gold = get_embedding(gold)
                sim = cosine_similarity(emb_pred, emb_gold)
                if sim > best_score:
                    best_score = sim
                    best_idx = idx_gold
                    best_gold = gold

            match = best_score >= threshold
            print(f"\nTriple {idx_pred + 1}:")
            print(f"GPT:   {pred}")
            print(f"CBM:   {best_gold}")
            print(f"Sim:   {best_score:.3f} → {'✅ MATCH' if match else '❌ NO MATCH'}")

            if match:
                TP += 1
                matched_gold.add(best_idx)
            else:
                FP += 1

        FN += len(gold_triples) - len(matched_gold)
        print(f"Image Summary: TP={TP}, FP={FP}, FN={FN}")

    return TP, FP, FN

# === Threshold Evaluation ===

def evaluate_thresholds(gold_path, eval_path):
    df_gold = pd.read_excel(gold_path)
    df_eval = pd.read_excel(eval_path)

    gold_dict = group_full_triples(df_gold)
    eval_dict = group_full_triples(df_eval)

    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    results = []

    print("\n=== Evaluating BioBERT Full-Triple Similarity Thresholds ===")
    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold:.2f} ---")
        TP, FP, FN = compare_full_triples(gold_dict, eval_dict, threshold)
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        results.append((threshold, precision, recall, f1))
        print(f"\nThreshold: {threshold:.2f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

    # Plot Results
    thresholds, precisions, recalls, f1s = zip(*results)
    plt.figure(figsize=(7, 5), dpi=600)
    plt.plot(thresholds, f1s, marker='o', label='F1 Score', linewidth=2)
    plt.plot(thresholds, precisions, marker='s', linestyle='--', label='Precision')
    plt.plot(thresholds, recalls, marker='^', linestyle='--', label='Recall')
    plt.xlabel("Semantic Similarity Threshold (Full Triple)", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Threshold vs. Precision / Recall / F1", fontsize=13)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(thresholds)
    plt.tight_layout()
    plt.savefig("data/figures_output/Threshold_Evaluation_BioBERT_FullTriple.tiff", dpi=600)
    plt.close()

# === CLI Interface ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioBERT full-triple threshold optimization.")
    parser.add_argument("--gold", required=True, help="Path to CBM gold standard Excel file")
    parser.add_argument("--eval", required=True, help="Path to GPT-predicted triple Excel file")
    args = parser.parse_args()
    evaluate_thresholds(args.gold, args.eval)

# python src/Threshold_Selection_BioBERT.py --gold data/prompt_engineering/cbm_files/CBM_subset_50_URL_triples.xlsx --eval data/prompt_engineering/gpt_files/GPT_subset_triples_prompt1_param0_0.xlsx