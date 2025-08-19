"""
Triple_Extraction_FullText.py

Semantic Triple Extraction from Full-Text Biomedical Articles using GPT-4o

Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 2025-07-11

Description:
    This script extracts semantic triples from full-text biomedical articles focused on the comorbidity between COVID-19 and neurodegeneration.
    It uses OpenAI's GPT-4o model to process paragraphs and extract structured triples in the format (subject|predicate|object).

    Key functionalities:
    1. Reads paragraph-level full text from a JSON input.
    2. Sends each paragraph to GPT-4o with a carefully designed prompt.
    3. Parses GPT output and validates triple format.
    4. If malformed, replaces output with a fallback standard triple.
    5. Removes non-informative results (e.g., Not_found).
    6. Saves valid triples to CSV and Excel.

Input:
    - JSON file with article texts and metadata.
    - OpenAI API key (via CLI argument or environment variable).

Output:
    - Triples_Full_Text_GPT_for_comp.csv: All extracted semantic triples.
    - Triples_Full_Text_GPT_for_comp.xlsx: Excel version of the same data.

Usage:
    python Triple_Extraction_FullText.py \
        --input data/CBM_data/full_text_articles.json \
        --output_dir ./data/gold_standard_comparison \
        --api_key sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXX
"""

import json
import os
import pandas as pd
from openai import OpenAI

def gpt_authenticate(API_key):
    """
    Authenticates with the OpenAI GPT API using the provided API key.

    Args:
        API_key (str): OpenAI API key.

    Returns:
        OpenAI: Authenticated API client.
    """
    return OpenAI(api_key=API_key)

def gpt_extract_from_text(client, paragraph):
    """
    Sends a text paragraph to GPT-4o to extract structured pathophysiological triples.

    Args:
        client (OpenAI): Authenticated API client.
        paragraph (str): Text paragraph to process.

    Returns:
        str: Raw GPT output with extracted mechanisms and triples.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": '''Describe the following scientific paragraph from an article on comorbidity between COVID-19 and Neurodegeneration.
                                      1. Name potential mechanisms (pathophysiological processes) of Covid-19's impact on the brain described in the text. 
                                      2. Describe each process described in the text as semantic triples (subject–predicate–object).  
                                      Example: 
                                      Pathophysiological Process: Astrocyte_Activation 
                                      Triples:
                                      SARS-CoV-2_infection|triggers|astrocyte_activation

                                      If the paragraph does not contain relevant biological content (e.g. acknowledgments, funding, conflicts of interest, publisher notes, metadata, disclaimers, or any non-scientific content), or if no such mechanisms or valid triples are present in the paragraph, return exactly:
                                      Pathophysiological Process: Not_found  
                                      Triples:  
                                      Not_found|Not_found|Not_found
                                      
                                      Use ONLY the information provided in the text! Follow the structure precisely and don't write anything else! Replace spaces in names with _ sign, make sure that words "Pathophysiology Process:" and "Triples:" are presented, don't use bold font and margins. Each triple must contain ONLY THREE elements separated by a | sign, four and more are not allowed!'''},
                    {
                        "type": "text",
                        "text": paragraph
                    }
                ]
            }
        ],
        max_tokens=2000,
        temperature=0.0,
        top_p=0.0
    )
    return response.choices[0].message.content

def triples_extraction_from_articles(input_path, output_dir, API_key):
    """
    Main function that orchestrates full pipeline:
        1. Loads articles
        2. Extracts triples via GPT-4o
        3. Validates and filters triples
        4. Saves to CSV and Excel

    Args:
        input_path (str): Path to JSON file with article data.
        output_dir (str): Directory to store output files.
        API_key (str): OpenAI API key for GPT-4o access.
    """
    client = gpt_authenticate(API_key)

    with open(input_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    parsed_data = []

    for article_id, article in articles.items():
        title = article.get("title", "")
        paragraphs = article.get("paragraphs", [])

        for paragraph in paragraphs:
            try:
                content = gpt_extract_from_text(client, paragraph)
                print(f"Processed paragraph from article {article_id} | Title: {title}\n{content}\n")

                if content.strip() == "Pathophysiological Process: Not_found\nTriples:\nNot_found|Not_found|Not_found":
                    parsed_data.append([article_id, title, paragraph, "Not_found", "Not_found", "Not_found", "Not_found"])
                    continue

                mechanisms = content.strip().split('Pathophysiological Process: ')
                for mechanism_block in mechanisms[1:]:
                    lines = mechanism_block.strip().split('\n')
                    mechanism_name = lines[0].strip()
                    triples = lines[2:]  # skip the "Triples:" line

                    malformed = False
                    for triple in triples:
                        parts = triple.strip().split('|')
                        if len(parts) != 3:
                            malformed = True
                            break

                    if malformed:
                        parsed_data.append([article_id, title, paragraph, "Not_found", "Not_found", "Not_found", "Not_found"])
                        break
                    else:
                        for triple in triples:
                            subject, predicate, obj = triple.strip().split('|')
                            parsed_data.append([article_id, title, paragraph, mechanism_name, subject, predicate, obj])

            except Exception as e:
                print(f"Error processing article {article_id}: {e}")
                continue

    parsed_df = pd.DataFrame(parsed_data, columns=[
        'PMID', 'Title', 'Paragraph', 'Pathophysiological Process', 'Subject', 'Predicate', 'Object'])

    # Define the filtering condition
    def is_valid(row):
        return row["Pathophysiological Process"] != "Not_found"

    parsed_df = parsed_df[parsed_df.apply(is_valid, axis=1)].reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "Triples_Full_Text_GPT_for_comp.csv")
    xlsx_path = os.path.join(output_dir, "Triples_Full_Text_GPT_for_comp.xlsx")

    parsed_df.to_csv(csv_path, index=False)
    parsed_df.to_excel(xlsx_path, index=False)

    print(f"Triples saved to:\n- {csv_path}\n- {xlsx_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract semantic triples from full-text articles using OpenAI GPT-4o.")
    parser.add_argument("--input", required=True, help="Path to full_text_articles.json.")
    parser.add_argument("--output_dir", default="./data/triples_output", help="Directory to save output files.")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key (or set OPENAI_API_KEY env variable).")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("No API key provided. Use --api_key or set the OPENAI_API_KEY environment variable.")

    triples_extraction_from_articles(args.input, args.output_dir, args.api_key)
    