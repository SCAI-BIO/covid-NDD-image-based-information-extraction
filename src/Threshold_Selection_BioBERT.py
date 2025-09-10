# Threshold_Selection_BioBERT.py

"""
Threshold Optimization for BioBERT Triple Matching (Full Triple Embedding)
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 30/07/2025

Description:
    Evaluates cosine similarity thresholds to align GPT-predicted full semantic triples
    (subject–predicate–object) with CBM gold standard using BioBERT. Full triples are compared
    as single normalized text strings, and global best-match selection is performed using
    the Hungarian algorithm for optimal assignment.

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
from scipy.optimize import linear_sum_assignment
import os

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
    comparison_records = []

    for image_id in sorted(set(gold_dict) & set(eval_dict), key=lambda x: int(x.split('_')[-1])):
        print(f"\n--- Comparing triples for image: {image_id} ---")

        gold_triples = gold_dict[image_id]
        eval_triples = eval_dict[image_id]

        emb_gold = [get_embedding(t) for t in gold_triples]
        emb_eval = [get_embedding(t) for t in eval_triples]

        sim_matrix = np.zeros((len(eval_triples), len(gold_triples)))

        for i, e_emb in enumerate(emb_eval):
            for j, g_emb in enumerate(emb_gold):
                sim = cosine_similarity(e_emb, g_emb)
                sim_matrix[i, j] = sim
                comparison_records.append({
                    "Image": image_id,
                    "GPT_Triple": eval_triples[i],
                    "CBM_Triple": gold_triples[j],
                    "Similarity": sim,
                })

        # Hungarian algorithm
        cost_matrix = 1 - sim_matrix  # maximize similarity = minimize (1 - sim)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_gpt = set()
        matched_cbm = set()

        print("\n✓ Matched triples:")
        for i, j in zip(row_ind, col_ind):
            sim = sim_matrix[i, j]
            if sim >= threshold:
                print(f"GPT: {eval_triples[i]}\nCBM: {gold_triples[j]}\nSimilarity: {sim:.4f} ✅ MATCH\n")
                matched_gpt.add(i)
                matched_cbm.add(j)

        unmatched_gpt = [eval_triples[i] for i in range(len(eval_triples)) if i not in matched_gpt]
        unmatched_cbm = [gold_triples[j] for j in range(len(gold_triples)) if j not in matched_cbm]

        if unmatched_gpt:
            print("\n❌ Unmatched GPT triples:")
            for triple in unmatched_gpt:
                print(f"GPT: {triple}")

        if unmatched_cbm:
            print("\n❌ Unmatched CBM triples:")
            for triple in unmatched_cbm:
                print(f"CBM: {triple}")

        TP += len(matched_gpt)
        FP += len(eval_triples) - len(matched_gpt)
        FN += len(gold_triples) - len(matched_cbm)

        print(f"\nImage Summary: TP={len(matched_gpt)}, FP={len(eval_triples) - len(matched_gpt)}, FN={len(gold_triples) - len(matched_cbm)}")

    return TP, FP, FN, comparison_records

# === Threshold Evaluation ===

def evaluate_thresholds(gold_path, eval_path):
    df_gold = pd.read_excel(gold_path)
    df_eval = pd.read_excel(eval_path)

    gold_dict = group_full_triples(df_gold)
    eval_dict = group_full_triples(df_eval)

    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    results = []
    all_records = []  # store for saving once

    output_dir = "data/figures_output"
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Evaluating BioBERT Full-Triple Similarity Thresholds ===")
    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold:.2f} ---")
        TP, FP, FN, records = compare_full_triples(gold_dict, eval_dict, threshold)

        # store only once — first run will have the same records as others
        if not all_records:
            all_records = records

        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        results.append((threshold, precision, recall, f1))

        print(f"\nThreshold: {threshold:.2f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

    # Save one Excel file with all similarity comparisons
    pd.DataFrame(all_records).to_excel(
        "data/prompt_engineering/statistical_data/Similarity_Threshold_Report.xlsx",
        index=False
    )

    # Define custom colors (hex codes or named colors)
    colors = {
        "F1": "#5c0b23",        
        "Precision": "#db7221", 
        "Recall": "#176e54"     
    }

    # Plot results with custom colors and line styles
    thresholds, precisions, recalls, f1s = zip(*results)
    plt.figure(figsize=(7, 5), dpi=600)

    plt.plot(thresholds, f1s, marker='o', label='F1 Score',
            linewidth=2, color=colors["F1"], linestyle='-')   # solid line

    plt.plot(thresholds, precisions, marker='s', label='Precision',
            linewidth=2, color=colors["Precision"], linestyle='--')  # dashed

    plt.plot(thresholds, recalls, marker='^', label='Recall',
            linewidth=2, color=colors["Recall"], linestyle='-.')      # dotted

    plt.xlabel("Semantic Similarity Threshold (Full Triple)", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(thresholds)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Threshold_Evaluation.tiff", dpi=600)
    plt.close()

# === CLI Interface ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioBERT full-triple threshold optimization.")
    parser.add_argument("--gold", required=True, help="Path to CBM gold standard Excel file")
    parser.add_argument("--eval", required=True, help="Path to GPT-predicted triple Excel file")
    args = parser.parse_args()
    evaluate_thresholds(args.gold, args.eval)

# python src/Threshold_Selection_BioBERT.py --gold data/prompt_engineering/cbm_files/CBM_subset_50_URL_triples.xlsx --eval data/prompt_engineering/gpt_files/GPT_subset_triples_prompt1_param0_0.xlsx
