# Gold_Standard_Comparison_BioBERT.py

"""
Semantic Triple Comparison using BioBERT (Hungarian Matching)
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 30/07/2025

Description:
    This version compares full semantic triples (subject–predicate–object) by encoding them as full sentences
    using BioBERT and computing cosine similarity. All GPT-CBM triple pairs are compared within each image,
    and global one-to-one matching is computed using the Hungarian algorithm.

    A predicted triple is considered a match if its similarity to the assigned gold triple exceeds a threshold.

Input:
    - Gold standard triples (Excel)
    - GPT-extracted triples (Excel)

Output:
    - Console logs with matches and similarity
    - Global evaluation metrics
    - Similarity histogram plot
    - Full comparison log per triple (saved as Excel)

Usage:
    python src/Gold_Standard_Comparison_BioBERT.py --gold <gold_file> --eval <eval_file> --threshold 0.85
"""

import pandas as pd
import numpy as np
import torch
import re
import argparse
import matplotlib.pyplot as plt
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
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_triple(s, p, o):
    return f"{normalize(s)} {normalize(p)} {normalize(o)}"

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def group_triples(df):
    grouped = {}
    for _, row in df.iterrows():
        if all(pd.notna([row['Subject'], row['Predicate'], row['Object']])):
            grouped.setdefault(row['Image_number'], []).append(
                (row['Subject'], row['Predicate'], row['Object'])
            )
    return grouped

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def evaluate_images(df_gold, df_eval, threshold=0.85):
    gold_dict = group_triples(df_gold)
    eval_dict = group_triples(df_eval)

    common_images = sorted(set(gold_dict.keys()) & set(eval_dict.keys()), key=lambda x: int(x.split('_')[-1]))
    total_TP = total_FP = total_FN = 0
    all_scores = []
    matched_scores = []
    comparison_records = []

    for image_id in common_images:
        print(f"\n--- Comparing triples for image: {image_id} ---")

        gold_triples = gold_dict[image_id]
        eval_triples = eval_dict[image_id]

        gold_sentences = [format_triple(*t) for t in gold_triples]
        eval_sentences = [format_triple(*t) for t in eval_triples]

        emb_gold = [get_embedding(t) for t in gold_sentences]
        emb_eval = [get_embedding(t) for t in eval_sentences]

        sim_matrix = np.zeros((len(eval_triples), len(gold_triples)))

        for i, e_emb in enumerate(emb_eval):
            for j, g_emb in enumerate(emb_gold):
                sim = cosine_similarity(e_emb, g_emb)
                sim_matrix[i, j] = sim
                comparison_records.append({
                    "Image": image_id,
                    "GPT_Triple": eval_sentences[i],
                    "CBM_Triple": gold_sentences[j],
                    "Similarity": sim
                })

        cost_matrix = 1 - sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_gpt = set()
        matched_cbm = set()

        print("\n✓ Matched triples:")
        for i, j in zip(row_ind, col_ind):
            sim = sim_matrix[i, j]
            if sim >= threshold:
                print(f"GPT: {eval_sentences[i]}\nCBM: {gold_sentences[j]}\nSimilarity: {sim:.4f} ✅ MATCH\n")
                matched_gpt.add(i)
                matched_cbm.add(j)

        unmatched_gpt = [eval_sentences[i] for i in range(len(eval_sentences)) if i not in matched_gpt]
        unmatched_cbm = [gold_sentences[j] for j in range(len(gold_sentences)) if j not in matched_cbm]

        if unmatched_gpt:
            print("\n❌ Unmatched GPT triples:")
            for triple in unmatched_gpt:
                print(f"GPT: {triple}")

        if unmatched_cbm:
            print("\n❌ Unmatched CBM triples:")
            for triple in unmatched_cbm:
                print(f"CBM: {triple}")

        TP = len(matched_gpt)
        FP = len(eval_triples) - TP
        FN = len(gold_triples) - len(matched_cbm)

        total_TP += TP
        total_FP += FP
        total_FN += FN

        print(f"\nImage Summary: TP={TP}, FP={FP}, FN={FN}")

    # === Global metrics ===
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n=== Overall Evaluation ===")
    print(f"TP: {total_TP}, FP: {total_FP}, FN: {total_FN}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")

    # === Plot histogram ===
    plt.figure(figsize=(6, 4), dpi=600)
    plt.hist([rec["Similarity"] for rec in comparison_records], bins=np.linspace(0.5, 1.0, 11),
             edgecolor='black', color='teal')
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    #plt.title("Distribution of Triple Similarity Scores (Full Sentence)")
    #plt.tight_layout()
    os.makedirs("data/figures_output", exist_ok=True)
    plt.savefig("data/figures_output/Fig_FullTriple_Semantic.tiff", dpi=600)
    plt.close()

    # === Save log ===
    out_dir = "data/gold_standard_comparison"
    os.makedirs(out_dir, exist_ok=True)
    threshold_str = str(int(threshold * 100))
    log_path = f"{out_dir}/CBM_comparison_Report_Threshold_{threshold_str}.xlsx"
    pd.DataFrame(comparison_records).to_excel(log_path, index=False)
    print(f"\nSaved comparison log to {log_path}")

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare full semantic triples using BioBERT + Hungarian matching")
    parser.add_argument("--gold", required=True, help="Path to gold standard (.xlsx)")
    parser.add_argument("--eval", required=True, help="Path to GPT triples (.xlsx)")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold")
    args = parser.parse_args()

    df_gold = pd.read_excel(args.gold)
    df_eval = pd.read_excel(args.eval)

    evaluate_images(df_gold, df_eval, threshold=args.threshold)


# python src/Gold_Standard_Comparison_BioBERT.py --gold data/gold_standard_comparison/Triples_CBM_Gold_Standard.xlsx --eval data/gold_standard_comparison/Triples_GPT_for_comparison.xlsx