# Hyperparameters_and_Prompt_Assessment.py

"""
Hyperparameters and Prompt Assessment Script
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 15/02/2025

Description:
    This script evaluates the semantic similarity of triples extracted from images using different GPT prompts
    and hyperparameter configurations. It compares them against a manually curated gold standard and
    calculates a Combined Similarity Score (CSS).

Features:
    1. Prompt Performance Assessment
       - Compares Prompt_1, Prompt_2, Prompt_3 against manual triples.
    2. Hyperparameter Analysis
       - Evaluates the impact of temperature/top-p on extraction quality.

Input:
    - Excel file with sheets:
        - Manual (gold standard)
        - Prompt_1, Prompt_2, Prompt_3
        - Prompt_1_Parameters_0.0, ..., 0.75

Output:
    - Console output of CSS scores per setting
    - Visualization: Line plot of hyperparameters vs. CSS

Metric:
        - Combined Similarity Score (CSS):
          CSS = 0.6 × Average Cosine Similarity + 0.4 × F1-Score
    
Run from project root:
    >>> python src/Hyperparameters_and_Prompt_Assessment.py --input data/Supplementary_material_Table_1.xlsx

Dependencies:
    - sentence-transformers
    - pandas
    - numpy
    - matplotlib, seaborn
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import matplotlib.pyplot as plt
import argparse

# === Argument Parser ===
parser = argparse.ArgumentParser(description="Evaluate GPT prompt and hyperparameter performance against gold standard triples.")
parser.add_argument("--input", required=True, help="Path to the Excel file containing all prompt/hyperparameter results")
args = parser.parse_args()
path = args.input

def create_triples_dict(df):
    triples_dict = {}
    for _, row in df.iterrows():
        url = row['URL']
        subject = row['Subject'].replace('_', ' ') if pd.notna(row['Subject']) else ''
        predicate = row['Predicate'].replace('_', ' ') if pd.notna(row['Predicate']) else ''
        obj = row['Object'].replace('_', ' ') if pd.notna(row['Object']) else ''
        if subject and predicate and obj:
            triple = f"{subject} {predicate} {obj}"
            triples_dict.setdefault(url, []).append(triple)
    return triples_dict

def similarity_score(df_gold, df_test, url):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    gold_triples = create_triples_dict(df_gold).get(url, [])
    extracted_triples = create_triples_dict(df_test).get(url, [])
    if not gold_triples or not extracted_triples:
        return 0

    sim_matrix = util.cos_sim(
        model.encode(extracted_triples, convert_to_tensor=True),
        model.encode(gold_triples, convert_to_tensor=True)
    ).cpu().numpy()

    TP, FP = 0, 0
    SIM_THRESHOLD = 0.6
    best_scores = []

    for i in range(len(extracted_triples)):
        if sim_matrix.shape[1] > 0:
            best_score = max(sim_matrix[i])
            best_scores.append(best_score)
            if best_score >= SIM_THRESHOLD:
                TP += 1
            else:
                FP += 1
        else:
            best_scores.append(0)

    FN = max(0, len(gold_triples) - TP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_sim = np.mean(best_scores)
    return 0.6 * avg_sim + 0.4 * f1

def similarity_across_images(df_gold, df_test):
    urls = df_gold['URL'].unique()
    css_scores = [similarity_score(df_gold, df_test, url) for url in urls if url in df_test['URL'].unique()]
    return sum(css_scores) / len(css_scores) if css_scores else 0

def plot_hyperparameters(scores):
    hyperparams = [0.0, 0.25, 0.5, 0.75]
    if len(scores) != len(hyperparams):
        raise ValueError("Length of scores must match hyperparameter settings")

    plt.figure(figsize=(8, 5))
    plt.plot(hyperparams, scores, marker='o', linewidth=2)
    plt.xlabel("Temperature = Top_P", fontsize=12)
    plt.ylabel("Average CSS", fontsize=12)
    plt.title("Impact of Hyperparameters on CSS", fontsize=14)
    for i, score in enumerate(scores):
        plt.annotate(f"{score:.3f}", (hyperparams[i], scores[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.grid(True)
    plt.xticks(hyperparams)
    plt.show()

if __name__ == "__main__":
    # Load Sheets
    manual = pd.read_excel(path, sheet_name='Manual')
    prompts = {
        'Prompt_1': pd.read_excel(path, sheet_name='Prompt_1'),
        'Prompt_2': pd.read_excel(path, sheet_name='Prompt_2'),
        'Prompt_3': pd.read_excel(path, sheet_name='Prompt_3'),
    }
    temps = [0.0, 0.25, 0.5, 0.75]
    temp_sheets = [f'Prompt_1_Parameters_{str(t)}' for t in temps]
    param_runs = [pd.read_excel(path, sheet_name=sheet) for sheet in temp_sheets]

    print("Prompt Performance Assessment:")
    for name, df in prompts.items():
        css = similarity_across_images(manual, df)
        print(f"Average CSS (Manual vs. {name}): {css:.3f}")

    print("\nHyperparameters Analysis:")
    scores = []
    for i, df in enumerate(param_runs):
        css = similarity_across_images(manual, df)
        print(f"Temp = Top_P = {temps[i]}: CSS = {css:.3f}")
        scores.append(css)

    plot_hyperparameters(scores)

# === Example usage ===
# python src/Hyperparameters_and_Prompt_Assessment.py --input data/Supplementary_material_Table_1.xlsx