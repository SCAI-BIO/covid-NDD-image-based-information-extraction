# Gold_Standard_Comparison_BioBERT.py

"""
Semantic Triple Comparison using BioBERT
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 06/05/2025

Description:
    This script compares semantic triples (subject–object) extracted by GPT from biomedical images
    to a curated gold standard (CBM) using BioBERT-based semantic similarity.

    For each triple, the subject and object are embedded using BioBERT and compared to all gold triples.
    A predicted triple is considered a match if:
        - Both subject and object cosine similarities ≥ 0.85
        - Both subject and object lexical similarity ≥ 0.25

    MeSH normalization is applied to improve synonym handling across biomedical terms.

Key functionalities:
    1. Loads GPT-predicted and gold standard triples from Excel files.
    2. Normalizes all entities, including synonym resolution using MeSH data.
    3. Computes semantic similarity (BioBERT) and lexical similarity (difflib).
    4. Matches predicted to gold triples and calculates evaluation metrics.
    5. Displays full comparison output with similarities and matching decisions.
    6. Visualizes similarity score distribution across all predictions.

Input:
    - Excel file of gold standard triples (column: Image_number, Subject, Object)
    - Excel file of GPT-predicted triples (same format)

Output:
    - Console report of all matched and unmatched triples
    - Evaluation metrics: precision, recall, F1 score
    - Histogram of cosine similarity distribution

Requirements:
    - pandas
    - torch
    - matplotlib
    - transformers (BioBERT)
    - MeSH synonym JSON file

Usage:
    python src/Gold_Standard_Comparison_BioBERT.py --gold data/Triples_CBM_Gold_Standard.xlsx --eval data/Triples_GPT_for_comparison.xlsx
"""

import pandas as pd
import numpy as np
import argparse
import torch
import re
import json
import matplotlib.pyplot as plt
import difflib
from transformers import AutoTokenizer, AutoModel

# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Compare GPT triples to gold standard using BioBERT semantic similarity.")
parser.add_argument("--gold", required=True, help="Path to gold standard Excel file")
parser.add_argument("--eval", required=True, help="Path to GPT/eval triples Excel file")
args = parser.parse_args()

# === Load Data ===
df_gold = pd.read_excel(args.gold)
df_eval = pd.read_excel(args.eval)

# === Load BioBERT ===
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# === Load MeSH Synonyms ===
with open("data/MeSh_data/mesh_triples_synonyms.json", "r", encoding="utf-8") as f:
    mesh_data = json.load(f)

mesh_lookup = {}
for key, entry in mesh_data.items():
    canonical = entry["normalized"]
    for synonym in entry.get("synonyms", []):
        mesh_lookup[synonym.lower()] = canonical
    mesh_lookup[key.lower()] = canonical

# === Utility Functions ===

def normalize(text):
    """
    Cleans and normalizes biomedical entity text.

    Steps:
    - Converts to lowercase
    - Removes special characters and extra spaces
    - Applies MeSH synonym mapping if available

    Args:
        text (str): Raw subject or object text.

    Returns:
        tuple:
            original (str): Original input text.
            normalized (str): Cleaned and possibly synonym-resolved form.
            mesh_matched (bool): True if MeSH synonym was applied.
    """
    if pd.isna(text):
        return '', '', False

    original = text
    text_clean = text.lower().replace('_', ' ').replace('-', ' ')
    text_clean = re.sub(r'[^\w\s]', '', text_clean)
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()

    if text_clean in mesh_lookup:
        return original, mesh_lookup[text_clean], True

    if text_clean == "sars cov 2":
        return original, "covid 19", False

    return original, text_clean, False

def get_embedding(text):
    """
    Generates a BioBERT embedding for the input text.

    Args:
        text (str): Normalized biomedical term.

    Returns:
        torch.Tensor: Sentence embedding vector (mean-pooled).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def cosine_similarity(a, b):
    """
    Computes cosine similarity between two vectors.

    Args:
        a (Tensor): Embedding vector A.
        b (Tensor): Embedding vector B.

    Returns:
        float: Cosine similarity score between 0 and 1.
    """
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def lexical_similarity(a, b):
    """
    Computes lexical similarity between two strings using difflib.

    Args:
        a (str): String A.
        b (str): String B.

    Returns:
        float: Similarity ratio between 0 and 1.
    """
    return difflib.SequenceMatcher(None, a, b).ratio()

def group_triples(df):
    """
    Groups subject–object triples by image identifier.

    Args:
        df (DataFrame): Triple data with 'Image_number', 'Subject', and 'Object'.

    Returns:
        dict: Mapping from image ID to list of (subject, object) pairs.
    """
    grouped = {}
    for _, row in df.iterrows():
        image = row['Image_number']
        subj = row['Subject']
        obj = row['Object']
        if pd.notna(subj) and pd.notna(obj):
            grouped.setdefault(image, []).append((subj, obj))
    return grouped

def compare_triples_biobert(gold_triples, eval_triples, image_id, sim_threshold=0.85, lex_thr=0.25):
    """
    Compares predicted triples to gold standard triples for one image using BioBERT.

    For each predicted triple:
        - Normalizes subject and object
        - Finds the best-matching gold triple using semantic and lexical similarity
        - Applies thresholds to decide if it's a match

    Args:
        gold_triples (list): List of (subject, object) gold triples.
        eval_triples (list): List of (subject, object) predicted triples.
        image_id (str): Image identifier.
        sim_threshold (float): Minimum cosine similarity required for match.
        lex_thr (float): Minimum lexical similarity required for match.

    Returns:
        tuple: (TP, FP, FN, all_sim_scores, matched_sim_scores)
    """
    TP, FP = 0, 0
    matched_gold = set()
    all_sim_scores = []
    matched_sim_scores = []

    print(f"\n Comparing triples for image: {image_id}")

    for idx_pred, (s_pred_raw, o_pred_raw) in enumerate(eval_triples):
        s_pred_orig, s_pred_norm, s_pred_mesh = normalize(s_pred_raw)
        o_pred_orig, o_pred_norm, o_pred_mesh = normalize(o_pred_raw)

        sub_pred_emb = get_embedding(s_pred_norm)
        obj_pred_emb = get_embedding(o_pred_norm)

        best_score = 0
        best_idx = None
        best_pair = ("", "")
        sub_sim_best = 0
        obj_sim_best = 0
        best_s_gold_norm = ""
        best_o_gold_norm = ""
        s_gold_mesh = False
        o_gold_mesh = False

        for idx_gold, (s_gold_raw, o_gold_raw) in enumerate(gold_triples):
            if idx_gold in matched_gold:
                continue
            s_gold_orig, s_gold_norm, s_gold_mesh_tmp = normalize(s_gold_raw)
            o_gold_orig, o_gold_norm, o_gold_mesh_tmp = normalize(o_gold_raw)

            sub_gold_emb = get_embedding(s_gold_norm)
            obj_gold_emb = get_embedding(o_gold_norm)

            sub_sim = cosine_similarity(sub_pred_emb, sub_gold_emb)
            obj_sim = cosine_similarity(obj_pred_emb, obj_gold_emb)
            sim_avg = (sub_sim + obj_sim) / 2

            if sim_avg > best_score:
                best_score = sim_avg
                sub_sim_best = sub_sim
                obj_sim_best = obj_sim
                best_idx = idx_gold
                best_pair = (s_gold_orig, o_gold_orig)
                best_s_gold_norm = s_gold_norm
                best_o_gold_norm = o_gold_norm
                s_gold_mesh = s_gold_mesh_tmp
                o_gold_mesh = o_gold_mesh_tmp

        all_sim_scores.append(best_score)
        lex_sub_sim = lexical_similarity(s_pred_norm, best_s_gold_norm)
        lex_obj_sim = lexical_similarity(o_pred_norm, best_o_gold_norm)

        is_match = (
            sub_sim_best >= sim_threshold and
            obj_sim_best >= sim_threshold and
            lex_sub_sim >= lex_thr and
            lex_obj_sim >= lex_thr  # Both subject and object must pass lexical threshold
        )

        match_str = "✅ MATCH" if is_match else "❌ NO MATCH"

        print(f"\nTriple {idx_pred + 1}:")
        print(f'Original: "{s_pred_raw} → {o_pred_raw}" (GPT) ↔ "{best_pair[0]} → {best_pair[1]}" (CBM)')
        print(f'Normalized: "{s_pred_norm}{" (MeSH)" if s_pred_mesh else ""} → {o_pred_norm}{" (MeSH)" if o_pred_mesh else ""}"'
              f' ↔ "{best_s_gold_norm}{" (MeSH)" if s_gold_mesh else ""} → {best_o_gold_norm}{" (MeSH)" if o_gold_mesh else ""}"')
        print(f"sub_sim={sub_sim_best:.3f}, obj_sim={obj_sim_best:.3f}, avg={best_score:.4f} | "
              f"lex_sub={lex_sub_sim:.2f}, lex_obj={lex_obj_sim:.2f} → {match_str}")

        if is_match:
            TP += 1
            matched_gold.add(best_idx)
            matched_sim_scores.append(best_score)
        else:
            FP += 1

    FN = len(gold_triples) - len(matched_gold)
    return TP, FP, FN, all_sim_scores, matched_sim_scores

# === Evaluation ===

def evaluate_all_images(df_gold, df_eval):
    """
    Evaluates triple matching across all shared images between gold and predicted datasets.

    - Aggregates TP, FP, FN across images
    - Prints per-image comparison details
    - Computes global precision, recall, and F1
    - Plots cosine similarity histogram

    Args:
        df_gold (DataFrame): Gold standard triples.
        df_eval (DataFrame): GPT-predicted triples.
    """
    gold_dict = group_triples(df_gold)
    eval_dict = group_triples(df_eval)

    image_keys = set(gold_dict.keys()) & set(eval_dict.keys())
    total_TP, total_FP, total_FN = 0, 0, 0
    all_similarities = []
    matched_similarities = []

    for image_id in sorted(image_keys, key=lambda x: int(x.split('_')[-1])):
        gold_triples = gold_dict[image_id]
        eval_triples = eval_dict[image_id]

        TP, FP, FN, all_sims, matched_sims = compare_triples_biobert(
            gold_triples, eval_triples, image_id
        )
        total_TP += TP
        total_FP += FP
        total_FN += FN
        all_similarities.extend(all_sims)
        matched_similarities.extend(matched_sims)

    # Metrics
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n=== Overall Evaluation ===")
    print(f"Total images compared: {len(image_keys)}")
    print(f"TP = {total_TP}, FP = {total_FP}, FN = {total_FN}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")

    print(f"\n Similarity Stats:")
    print(f"  Avg similarity (all predicted triples): {np.mean(all_similarities):.4f}")
    if matched_similarities:
        print(f"  Avg similarity (only matched triples): {np.mean(matched_similarities):.4f}")
    else:
        print("  No matched triples.")

    # Plot
    plt.figure(figsize=(6, 4), dpi=600)
    plt.hist(all_similarities, bins=[0, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0], edgecolor='black', color='#4A9BB5')

    plt.xlabel("Cosine Similarity", fontsize=10, fontname="Arial")
    plt.ylabel("Frequency", fontsize=10, fontname="Arial")
    plt.xticks(fontsize=8, fontname="Arial")
    plt.yticks(fontsize=8, fontname="Arial")
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig("data/figures_output/Fig7.tiff", format='tiff', dpi=600)
    plt.close()

# === Run ===
if __name__ == "__main__":
    evaluate_all_images(df_gold, df_eval)

