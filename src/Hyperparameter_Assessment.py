# Hyperparameter_Assessment.py
"""
Hyperparameter Tuning for GPT-Generated Triple Extraction (Full-Triple Matching with Hungarian Algorithm)
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 30/07/2025

Description:
    Compares GPT triples generated with different decoding parameters (temperature & top_p)
    to a CBM gold standard using BioBERT-based cosine similarity. 
    Matches are evaluated over full triples (subject–predicate–object) as sentences.

Usage:
    python src/Hyperparameter_Assessment.py
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer, AutoModel

# === File Paths ===
GOLD_PATH = "data/prompt_engineering/cbm_files/CBM_subset_50_URL_triples.xlsx"
HYPERPARAM_PATHS = {
    "Temp=1.0, Top_p=1.0": "data/prompt_engineering/gpt_files/GPT_subset_triples_prompt1_param1_0.xlsx",
    "Temp=0.75, Top_p=0.75": "data/prompt_engineering/gpt_files/GPT_subset_triples_prompt1_param0_75.xlsx",
    "Temp=0.5, Top_p=0.5": "data/prompt_engineering/gpt_files/GPT_subset_triples_prompt1_param0_5.xlsx",
    "Temp=0.25, Top_p=0.25": "data/prompt_engineering/gpt_files/GPT_subset_triples_prompt1_param0_25.xlsx",
    "Temp=0.0, Top_p=0.0": "data/prompt_engineering/gpt_files/GPT_subset_triples_prompt1_param0_0.xlsx"
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

# === Core Matching Function ===
def compare_triples(gold_dict, eval_dict, threshold):
    TP = FP = FN = 0

    for image_id in sorted(set(gold_dict) & set(eval_dict), key=lambda x: int(x.split('_')[-1])):
        gold_triples = gold_dict[image_id]
        pred_triples = eval_dict[image_id]

        emb_gold = [get_embedding(t) for t in gold_triples]
        emb_pred = [get_embedding(t) for t in pred_triples]

        sim_matrix = np.zeros((len(pred_triples), len(gold_triples)))
        for i in range(len(pred_triples)):
            for j in range(len(gold_triples)):
                sim_matrix[i][j] = cosine_similarity(emb_pred[i], emb_gold[j])

        cost_matrix = 1 - sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = 0
        for i, j in zip(row_ind, col_ind):
            if sim_matrix[i][j] >= threshold:
                matches += 1

        TP += matches
        FP += len(pred_triples) - matches
        FN += len(gold_triples) - matches

    return TP, FP, FN

# === Evaluation Routine ===
def evaluate_setting(setting_name, gold_df, eval_df, threshold):
    gold_dict = group_full_triples(gold_df)
    eval_dict = group_full_triples(eval_df)

    print(f"\n==================== {setting_name} ====================")
    TP, FP, FN = compare_triples(gold_dict, eval_dict, threshold)
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")

    return setting_name, precision, recall, f1, TP, FP, FN

# === Run All Settings ===
if __name__ == "__main__":
    df_gold = pd.read_excel(GOLD_PATH)
    results = []

    for setting, path in HYPERPARAM_PATHS.items():
        df_eval = pd.read_excel(path)
        result = evaluate_setting(setting, df_gold, df_eval, SIM_THRESHOLD)
        results.append(result)

    # Save stats
    df_stats = pd.DataFrame(results, columns=["Setting", "Precision", "Recall", "F1 Score", "TP", "FP", "FN"])
    os.makedirs("data/prompt_engineering/statistical_data", exist_ok=True)
    df_stats.to_excel("data/prompt_engineering/statistical_data/Hyperparameter_Assessment.xlsx", index=False)

    # Plot
    x = np.arange(len(df_stats))
    width = 0.25

    plt.figure(figsize=(8, 5), dpi=600)
    plt.bar(x - width, df_stats["Precision"], width, label="Precision", color="#E69F00")
    plt.bar(x, df_stats["Recall"], width, label="Recall", color="#56B4E9")
    plt.bar(x + width, df_stats["F1 Score"], width, label="F1 Score", color="#009E73")

    plt.xticks(x, df_stats["Setting"], rotation=30, ha='right')
    plt.ylabel("Score")
    plt.ylim(0, 1)
    #plt.title("Hyperparameter Comparison via BioBERT Triple Matching (Threshold 0.85)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    os.makedirs("data/figures_output", exist_ok=True)
    plt.savefig("data/figures_output/Hyperparameter_Assessment.tiff", dpi=600)
    plt.close()
