# Gold_Standard_Comparison_BioBERT.py

"""
Semantic Triple Comparison using BioBERT (Full Triple Embedding)
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 30/07/2025

Description:
    This version compares full semantic triples (subject–predicate–object) by encoding them as full sentences
    using BioBERT and computing cosine similarity.

    A predicted triple is considered a match if the cosine similarity with any gold triple exceeds a threshold.

Input:
    - Gold standard triples (Excel)
    - GPT-extracted triples (Excel)

Output:
    - Console logs with matches and similarity
    - Global evaluation metrics
    - Similarity histogram plot

Usage:
    python src/Gold_Standard_Comparison_BioBERT.py --gold <gold_file> --eval <eval_file>
"""

import pandas as pd
import numpy as np
import torch
import re
import argparse
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

# === Load BioBERT ===
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# === Helper Functions ===

def normalize(text):
    """Basic normalization: lowercase, replace symbols, clean whitespace."""
    if pd.isna(text):
        return ""
    text = text.lower().replace('_', ' ').replace('-', ' ')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_triple(subject, predicate, obj):
    """Join normalized triple into a sentence-like phrase."""
    return f"{normalize(subject)} {normalize(predicate)} {normalize(obj)}"

def get_embedding(text):
    """Generate BioBERT embedding for input string."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def group_triples(df):
    """Group (Subject, Predicate, Object) triples by Image_number."""
    grouped = {}
    for _, row in df.iterrows():
        if all(pd.notna([row['Subject'], row['Predicate'], row['Object']])):
            grouped.setdefault(row['Image_number'], []).append(
                (row['Subject'], row['Predicate'], row['Object'])
            )
    return grouped

def compare_triples_full_biobert(gold_triples, eval_triples, image_id, threshold=0.85):
    TP, FP = 0, 0
    matched_gold = set()
    all_similarities = []
    matched_similarities = []

    print(f"\n--- Comparing triples for image: {image_id} ---")

    for idx_pred, (s_pred, p_pred, o_pred) in enumerate(eval_triples):
        pred_str = format_triple(s_pred, p_pred, o_pred)
        emb_pred = get_embedding(pred_str)

        best_score = 0
        best_idx = None
        best_gold_str = ""

        for idx_gold, (s_gold, p_gold, o_gold) in enumerate(gold_triples):
            if idx_gold in matched_gold:
                continue
            gold_str = format_triple(s_gold, p_gold, o_gold)
            emb_gold = get_embedding(gold_str)

            sim = cosine_similarity(emb_pred, emb_gold)

            if sim > best_score:
                best_score = sim
                best_idx = idx_gold
                best_gold_str = gold_str

        all_similarities.append(best_score)
        is_match = best_score >= threshold

        print(f"\nTriple {idx_pred+1}:")
        print(f"GPT:   {pred_str}")
        print(f"CBM:   {best_gold_str}")
        print(f"Score: {best_score:.3f} → {'✅ MATCH' if is_match else '❌ NO MATCH'}")

        if is_match:
            TP += 1
            matched_gold.add(best_idx)
            matched_similarities.append(best_score)
        else:
            FP += 1

    FN = len(gold_triples) - len(matched_gold)
    return TP, FP, FN, all_similarities, matched_similarities

def evaluate_images(df_gold, df_eval, threshold=0.85):
    gold_dict = group_triples(df_gold)
    eval_dict = group_triples(df_eval)

    common_images = sorted(set(gold_dict.keys()) & set(eval_dict.keys()), key=lambda x: int(x.split('_')[-1]))
    total_TP, total_FP, total_FN = 0, 0, 0
    all_scores, matched_scores = [], []

    for image_id in common_images:
        TP, FP, FN, sims, matched = compare_triples_full_biobert(
            gold_dict[image_id], eval_dict[image_id], image_id, threshold
        )
        total_TP += TP
        total_FP += FP
        total_FN += FN
        all_scores.extend(sims)
        matched_scores.extend(matched)

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n=== Overall Evaluation ===")
    print(f"TP: {total_TP}, FP: {total_FP}, FN: {total_FN}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")
    print(f"Avg Similarity (All): {np.mean(all_scores):.4f}")
    if matched_scores:
        print(f"Avg Similarity (Matched): {np.mean(matched_scores):.4f}")

    # Plot similarity histogram
    plt.figure(figsize=(6, 4), dpi=600)
    plt.hist(all_scores, bins=np.linspace(0.5, 1.0, 11), edgecolor='black', color='teal')
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Distribution of Triple Similarity Scores (Full Sentence)")
    plt.tight_layout()
    plt.savefig("data/figures_output/Fig_FullTriple_Semantic.tiff", dpi=600)
    plt.close()

# === CLI Execution ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare full semantic triples using BioBERT.")
    parser.add_argument("--gold", required=True, help="Path to gold standard (.xlsx)")
    parser.add_argument("--eval", required=True, help="Path to GPT triples (.xlsx)")
    parser.add_argument("--threshold", type=float, default=0.85, help="Cosine similarity threshold")
    args = parser.parse_args()

    df_gold = pd.read_excel(args.gold)
    df_eval = pd.read_excel(args.eval)

    evaluate_images(df_gold, df_eval, threshold=args.threshold)
