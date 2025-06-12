# Triples_Categorization.py

"""
BERT-Based Categorization Script for Biomedical Triples using MeSH Keywords
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 09/04/2025

Description:
    This script classifies biomedical triples into mechanistic categories using BERT embeddings and
    MeSH-derived keyword dictionaries. It supports two classification modes:

    1. Pathophysiological Process (PP) based classification (recommended when available).
    2. Subject + Object based classification (used as fallback or when PP is missing).

Categories:
    1. Viral Entry and Neuroinvasion
    2. Immune and Inflammatory Response
    3. Neurodegenerative Mechanisms
    4. Vascular Effects
    5. Psychological and Neurological Symptoms
    6. Systemic Cross-Organ Effects

Input:
    - MeSH descriptors file: desc2025.xml
    - Dataset file: Triples_Final_All_Relevant.csv

Output:
    - Categorized dataset as CSV and Excel files
    - Category counts per process

Requirements:
    - sentence-transformers
    - pandas
    - torch
    - openpyxl (for Excel export)
"""

import pandas as pd
import json
import re
import torch
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import argparse
import os

def normalize_text(text):
    text = re.sub(r"[;_\-]", " ", str(text))
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def combine_subject_object(row):
    fields = [row.get('Subject'), row.get('Object')]
    return ' '.join([normalize_text(f) for f in fields if pd.notna(f) and str(f).strip().lower() != 'missing'])

def bert_keyword_classify(text, category_keyword_embeddings, model, threshold=0.5, aggregation="max"):
    process_embedding = model.encode(text, convert_to_tensor=True)
    best_category = None
    best_score = 0.0

    for category, keyword_embeddings in category_keyword_embeddings.items():
        if keyword_embeddings is None or len(keyword_embeddings) == 0:
            continue

        cosine_scores = util.pytorch_cos_sim(process_embedding, keyword_embeddings)[0]

        if aggregation == "max":
            score = torch.max(cosine_scores).item()
        elif aggregation == "mean":
            score = torch.mean(cosine_scores).item()
        else:
            raise ValueError("Unsupported aggregation type. Use 'max' or 'mean'.")

        if score > best_score:
            best_score = score
            best_category = category

    return best_category if best_score >= threshold else "Uncategorized"

def run_classification(input_path, mesh_json_path, output_path, mode="pp"):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_excel(input_path)

    with open(mesh_json_path, 'r') as f:
        category_terms = json.load(f)

    print("Normalizing keywords...")
    for category in category_terms:
        category_terms[category] = [normalize_text(term) for term in category_terms[category]]

    print("Initializing BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Embedding category keywords...")
    category_keyword_embeddings = {
        category: model.encode(terms, convert_to_tensor=True) if terms else None
        for category, terms in category_terms.items()
    }

    if mode == "pp":
        print("Running PP-based classification...")
        df['Normalized_Input'] = df['Pathophysiological Process'].apply(normalize_text)
    elif mode == "subjobj":
        print("Running Subject+Object-based classification...")
        df['Normalized_Input'] = df.apply(combine_subject_object, axis=1)
    else:
        raise ValueError("Mode must be either 'pp' or 'subjobj'")

    print("Classifying using BERT + Keywords...")
    df['Category'] = df['Normalized_Input'].apply(
        lambda x: bert_keyword_classify(x, category_keyword_embeddings, model)
    )

    category_counts = Counter(df['Category'])
    print("=== Category Counts ===")
    for cat, count in category_counts.items():
        print(f"{cat}: {count}")

    df.to_csv(output_path + ".csv", index=False)
    df.to_excel(output_path + ".xlsx", index=False)
    print("Classification complete and results exported!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify biomedical triples using BERT and MeSH keywords.")
    parser.add_argument("--input", required=True, help="Path to input CSV or XLSX file")
    parser.add_argument("--mesh", default="data/MeSh_data/mesh_category_terms.json", help="Path to MeSH keyword JSON")
    parser.add_argument("--output", required=True, help="Path prefix for output files (no extension)")
    parser.add_argument("--mode", default="pp", choices=["pp", "subjobj"], help="Classification mode")
    args = parser.parse_args()

    run_classification(args.input, args.mesh, args.output, args.mode)

# === Example usage ===
# Run PP-based classification:
# python src/Triples_Categorization.py --input data/triples_output/Triples_Final_All_Relevant.csv --output data/triples_output/Triples_Final_All_Relevant_Categorized --mode pp
#
# Run Subject+Object-based classification:
# python src/Triples_Categorization.py --input data/gold_standard_comparison/Triples_CBM_Gold_Standard.xlsx --output data/gold_standard_comparison/Triples_CBM_Gold_Standard_SubjObj_Categorized --mode subjobj   
# python src/Triples_Categorization.py --input data/gold_standard_comparison/Triples_GPT_for_comparison.xlsx --output data/gold_standard_comparison/Triples_GPT_for_comparison_SubjObj_Categorized --mode subjobj
