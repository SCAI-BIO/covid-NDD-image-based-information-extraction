# Prompt_Assessment.py

"""
Prompt Comparison using BioBERT Full-Triple Embedding Matching (Hungarian Optimization)
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 06/08/2025

Description:
    Compares GPT triples from multiple prompts to a gold standard using BioBERT.
    Each triple (subject–predicate–object) is embedded as a normalized sentence and compared.
    Uses Hungarian algorithm for optimal one-to-one alignment between GPT and gold triples.
    Prints per-prompt metrics and generates precision / recall / F1 plots.

Usage:
    python src/Prompt_Assessment.py
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.optimize import linear_sum_assignment

# === Configuration ===
GOLD_PATH = "data/prompt_engineering/cbm_files/CBM_subset_50_URL_triples.xlsx"
PROMPT_PATHS = {
    "Prompt 1": "data/prompt_engineering/gpt_files/GPT_subset_triples_prompt1_param0_0.xlsx",
    "Prompt 2": "data/prompt_engineering/gpt_files/GPT_subset_triples_prompt2_param0_0.xlsx",
    "Prompt 3": "data/prompt_engineering/gpt_files/GPT_subset_triples_prompt3_param0_0.xlsx"
}
SIM_THRESHOLD = 0.85

# === Load BioBERT ===
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# === Helper Functions ===
def normalize(text):
    if pd.isna(text):
        return ''
    text = text.lower().replace('_', ' ').replace('-', ' ')
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

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
            triple = f"{normalize(row['Subject'])} {normalize(row['Predicate'])} {normalize(row['Object'])}"
            grouped.setdefault(row['Image_number'], []).append(triple)
    return grouped

# === Core Comparison with Hungarian Algorithm ===
def compare_triples_hungarian(gold_dict, eval_dict, threshold):
    TP = FP = FN = 0

    for image_id in sorted(set(gold_dict) & set(eval_dict), key=lambda x: int(x.split('_')[-1])):
        gold_triples = gold_dict[image_id]
        eval_triples = eval_dict[image_id]

        if not gold_triples or not eval_triples:
            FN += len(gold_triples)
            FP += len(eval_triples)
            continue

        emb_gold = [get_embedding(t) for t in gold_triples]
        emb_eval = [get_embedding(t) for t in eval_triples]

        sim_matrix = np.zeros((len(eval_triples), len(gold_triples)))
        for i, emb_pred in enumerate(emb_eval):
            for j, emb_gold_j in enumerate(emb_gold):
                sim_matrix[i, j] = cosine_similarity(emb_pred, emb_gold_j)

        cost_matrix = 1 - sim_matrix  # convert similarity to cost
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = 0
        for i, j in zip(row_ind, col_ind):
            if sim_matrix[i, j] >= threshold:
                matched += 1

        TP += matched
        FP += len(eval_triples) - matched
        FN += len(gold_triples) - matched

    return TP, FP, FN

# === Evaluation ===
def evaluate_prompt(prompt_name, gold_df, eval_df, threshold):
    gold_dict = group_full_triples(gold_df)
    eval_dict = group_full_triples(eval_df)

    print(f"\n==================== {prompt_name} ====================")
    TP, FP, FN = compare_triples_hungarian(gold_dict, eval_dict, threshold)
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"\n=== Summary for {prompt_name} ===")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")

    return prompt_name, precision, recall, f1, TP, FP, FN

# === Run Evaluation for All Prompts ===
if __name__ == "__main__":
    df_gold = pd.read_excel(GOLD_PATH)
    results = []

    for prompt_name, eval_path in PROMPT_PATHS.items():
        df_eval = pd.read_excel(eval_path)
        result = evaluate_prompt(prompt_name, df_gold, df_eval, SIM_THRESHOLD)
        results.append(result)

    # === Convert to DataFrame ===
    df_stats = pd.DataFrame(results, columns=["Prompt", "Precision", "Recall", "F1 Score", "TP", "FP", "FN"])
    x = np.arange(len(df_stats))
    width = 0.25

    # === Plot Results ===
    plt.figure(figsize=(8, 5), dpi=600)
    plt.bar(x + width, df_stats["F1 Score"], width, label="F1 Score", color="#5c0b23")
    plt.bar(x - width, df_stats["Precision"], width, label="Precision", color="#db7221")
    plt.bar(x, df_stats["Recall"], width, label="Recall", color="#176e54")

    plt.xticks(x, df_stats["Prompt"], fontsize=10)
    plt.ylabel("Score", fontsize=11)
    plt.ylim(0, 1)
    #plt.title("Prompt Comparison via BioBERT Triple Matching (Threshold 0.85)", fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    os.makedirs("data/figures_output", exist_ok=True)
    plt.tight_layout()
    plt.savefig("data/figures_output/Prompt_Comparison.tiff", dpi=600)
    plt.savefig("data/figures_output/Prompt_Comparison.png", dpi=600)
    plt.close()

    # === Save Evaluation Results ===
    os.makedirs("data/prompt_engineering/statistical_data", exist_ok=True)
    df_stats.to_excel("data/prompt_engineering/statistical_data/Prompt_Comparison.xlsx", index=False)
