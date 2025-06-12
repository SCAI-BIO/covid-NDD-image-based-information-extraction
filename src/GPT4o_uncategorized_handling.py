# GPT4o_uncategorized_handling.py

"""
This script assigns mechanistic categories to previously uncategorized Pathophysiological Processes (PP) using GPT-4o.

This script scans a categorized triple file, identifies entries marked as "Uncategorized" by the BERT+MeSH system,
and uses the OpenAI GPT-4o model to assign one of six predefined mechanistic categories.

The final output contains two new columns:
- Category_GPT: GPT-assigned category for previously uncategorized entries
- Final_Category: Consolidated category using BERT (if present) or GPT fallback

Authors: Elizaveta Popova, Negin Babaiha  
Institution: University of Bonn, Fraunhofer SCAI  
Date: 2025-04-10

Usage:
    python src/GPT4o_uncategorized_handling.py \
        --input data/triples_output/Triples_Final_All_Relevant_Categorized.xlsx \
        --output data/triples_output/Triples_Final_All_Relevant_Categorized_GPT4o \
        --api_key YOUR_API_KEY

Requirements:
    - openai
    - pandas
    - Python 3.8+

Environment variable option:
    export OPENAI_API_KEY=sk-...
    python src/GPT4o_uncategorized_handling.py --input ...
"""

import pandas as pd
import os
from openai import OpenAI
import argparse

# === GPT setup ===
def gpt_authenticate(API_key):
    """
    Authenticates with the OpenAI GPT API using the provided API key.

    Args:
        API_key (str): OpenAI API key.

    Returns:
        OpenAI: Authenticated API client.
    """
    return OpenAI(api_key=API_key)

def gpt_categorize(client, process_text):
    """
    Sends a pathophysiological process string to GPT-4o and retrieves a mechanistic category.

    Args:
        client (OpenAI): Authenticated API client.
        process_text (str): Pathophysiological process string.

    Returns:
        str: One of six category names or 'Uncategorized'
    """
    prompt = f"""
You are a biomedical expert. Categorize the following pathophysiological process into exactly one of the six predefined 
mechanistic categories listed below. Respond only with the category name — do not explain your reasoning.

Available categories:
1. Viral Entry and Neuroinvasion
2. Immune and Inflammatory Response
3. Neurodegenerative Mechanisms
4. Vascular Effects
5. Psychological and Neurological Symptoms
6. Systemic Cross-Organ Effects

If the input does not clearly fit any category, return: Uncategorized

Process: "{process_text}"

Your answer:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            top_p=0.0,
            max_tokens=50,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing: {process_text}\n{e}")
        return "GPT_Error"

def categorize_uncategorized(input_file, output_file, api_key):
    """
    Main function to categorize previously uncategorized PP entries using GPT-4o.

    Args:
        input_file (str): Path to input Excel file with BERT-based categorization.
        output_file (str): Path stem for saving categorized output as .xlsx and .csv.
        api_key (str): OpenAI API key.
    """
    client = gpt_authenticate(api_key)
    df = pd.read_excel(input_file)

    df['Category_GPT'] = ""
    for idx, row in df.iterrows():
        if row['Category'] == 'Uncategorized':
            pp_text = str(row['Pathophysiological Process'])
            gpt_cat = gpt_categorize(client, pp_text)
            df.at[idx, 'Category_GPT'] = gpt_cat
            print(f"Uncategorized Pathophysiological Process: '{pp_text}' -> GPT category: {gpt_cat}")

    # Combine to final category: use BERT first, fallback to GPT
    df['Final_Category'] = df['Category']
    df.loc[df['Final_Category'] == 'Uncategorized', 'Final_Category'] = df['Category_GPT']

    df.to_csv(output_file + ".csv", index=False)
    df.to_excel(output_file + ".xlsx", index=False)

    print(f"Saved categorized file to {output_file} in .xlsx and .csv")


# === CLI entry ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use GPT to categorize uncategorized PP triples.")
    parser.add_argument("--input", required=True, help="Path to input .xlsx file with 'Uncategorized' entries")
    parser.add_argument("--output", default="Triples_Final_All_Relevant_Categorized_GPT4o", help="Output .xlsx and .csv file path")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("No API key provided. Use --api_key or set the OPENAI_API_KEY environment variable.")

    categorize_uncategorized(args.input, args.output, args.api_key)

# === Example usage ===
# python src/GPT4o_uncategorized_handling.py --input data/triples_output/Triples_Final_All_Relevant_Categorized.xlsx --output data/triples_output/Triples_Final_All_Relevant_Categorized_GPT4o --api_key YOUR_API_KEY
