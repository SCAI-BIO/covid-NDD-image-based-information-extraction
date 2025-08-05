# Prompt_Assessment.py

"""
Prompt Comparison using BioBERT Full-Triple Embedding Matching
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 30/07/2025

Description:
    Compares GPT triples from multiple prompts to a gold standard using BioBERT.
    Each triple (subject–predicate–object) is embedded as a normalized sentence and compared.
    Prints matching logs, evaluates per prompt, saves statistics and visualizes results.

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

# === Configuration ===
GOLD_PATH = "data/prompt_engineering/cbm_files/CBM_subset_50_URL_triples.xlsx"
PROMPT_PATHS = {
    "Prompt 1": "data/prompt_engineering/gpt_files/GPT_subset_triples_prompt1_param0_0.xlsx",
    "Prompt 2": "data/prompt_engineering/gpt_files/GPT_subset_triples_prompt2_param0_0.xlsx",
    "Prompt 3": "data/prompt_engineering/gpt_files/GPT_subset_triples_prompt3_param0_0.xlsx"
}
SIM_THRESHOLD = 0.8

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

# === Core Comparison ===
def compare_triples(gold_dict, eval_dict, threshold):
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
            print(f"GPT:  {pred}")
            print(f"CBM:  {best_gold}")
            print(f"Sim:  {best_score:.3f} → {'✅ MATCH' if match else '❌ NO MATCH'}")

            if match:
                TP += 1
                matched_gold.add(best_idx)
            else:
                FP += 1

        FN += len(gold_triples) - len(matched_gold)
        print(f"Image Summary: TP={TP}, FP={FP}, FN={FN}")

    return TP, FP, FN

# === Evaluation ===
def evaluate_prompt(prompt_name, gold_df, eval_df, threshold):
    gold_dict = group_full_triples(gold_df)
    eval_dict = group_full_triples(eval_df)

    print(f"\n==================== {prompt_name} ====================")
    TP, FP, FN = compare_triples(gold_dict, eval_dict, threshold)
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
    plt.bar(x - width, df_stats["Precision"], width, label="Precision", color="#D95F02")
    plt.bar(x, df_stats["Recall"], width, label="Recall", color="#1B9E77")
    plt.bar(x + width, df_stats["F1 Score"], width, label="F1 Score", color="#7570B3")

    plt.xticks(x, df_stats["Prompt"], fontsize=10)
    plt.ylabel("Score", fontsize=11)
    plt.ylim(0, 1)
    plt.title("Prompt Comparison via BioBERT Full Triple Matching (Threshold 0.8)", fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    os.makedirs("data/figures_output", exist_ok=True)
    plt.tight_layout()
    plt.savefig("data/figures_output/Prompt_Comparison_BioBERT_FullTriple.tiff", dpi=600)
    plt.close()

    # === Save Evaluation Results ===
    os.makedirs("data/prompt_engineering/statistical_data", exist_ok=True)
    df_stats.to_excel("data/prompt_engineering/statistical_data/Prompt_Comparison_BioBERT_FullTriple.xlsx", index=False)
