# URL_subset_selector.py

"""
Random Subset Selector for CBM Evaluation
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 24/07/2025

Description:
    This script randomly selects 50 unique image URLs from the CBM dataset,
    saves the list of selected URLs and image numbers, and extracts the corresponding
    triples from the CBM gold-standard file.

Inputs:
    - Excel file: data/CBM_data/Data_CBM.xlsx
    - Excel file: data/CBM_data/Triples_CBM_Gold_Standard.xlsx

Outputs:
    - Excel file: data/prompt_engineering/cbm_files/CBM_subset_50_URLs.xlsx
    - Excel file: data/prompt_engineering/cbm_files/CBM_subset_50_URL_triples.xlsx
"""

import pandas as pd
import random
import os

# === File Paths ===
cbm_metadata_path = "data/CBM_data/Data_CBM_with_GitHub_URLs.xlsx"
cbm_triples_path = "data/CBM_data/Triples_CBM_Gold_Standard.xlsx"
output_urls_path = "data/prompt_engineering/cbm_files/CBM_subset_50_URLs.xlsx"
output_triples_path = "data/prompt_engineering/cbm_files/CBM_subset_50_URL_triples.xlsx"

# === Load the metadata ===
df_metadata = pd.read_excel(cbm_metadata_path)

# === Randomly sample 50 unique image URLs ===
unique_urls = df_metadata['URL'].dropna().unique().tolist()
sampled_urls = random.sample(unique_urls, 50)

# === Get corresponding Image_number and URL ===
df_subset_urls = df_metadata[df_metadata['URL'].isin(sampled_urls)][['Image_number', 'URL', 'GitHub_URL']].drop_duplicates()

# === Save subset of URLs ===
os.makedirs(os.path.dirname(output_urls_path), exist_ok=True)
df_subset_urls.to_excel(output_urls_path, index=False)

# === Load the full CBM gold-standard triples ===
df_triples = pd.read_excel(cbm_triples_path)

# === Filter triples for selected Image_numbers ===
selected_images = df_subset_urls['Image_number'].unique().tolist()
df_subset_triples = df_triples[df_triples['Image_number'].isin(selected_images)]

# === Save filtered triples ===
df_subset_triples.to_excel(output_triples_path, index=False)

print(f"Saved 50-image URL subset to: {output_urls_path}")
print(f"Saved corresponding triples to: {output_triples_path}")

