# Triple_Extraction_GPT4o.py

"""
Semantic Triple Extraction from Biomedical Images using GPT
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 06/11/2024

Description:
    This script extracts semantic triples from images related to the comorbidity between COVID-19 and neurodegeneration,
    using OpenAI's GPT-4o model. Each image is processed to identify pathophysiological mechanisms represented visually,
    and for each mechanism, structured triples (subject|predicate|object) are extracted.

    Key functionalities:
    1. Reads image URLs from an input Excel file.
    2. Sends each image to GPT with a strict prompt to ensure triple consistency.
    3. Parses the model output and extracts structured triples.
    4. Saves the results to CSV and Excel files.

Input:
    - Excel file containing image URLs in a column named "URL".
    - OpenAI API key (via CLI argument or environment variable).

Output:
    - Triples_Final_All.csv: All extracted semantic triples.
    - Triples_Final_All.xlsx: Excel version of the same data.

Requirements:
    - openai
    - pandas
    - Internet connection and OpenAI API access.

Usage:
    python src/Triple_Extraction_GPT4o.py --input data/URL_relevance_analysis/Final_Relevant_URLs.xlsx --output data/triples_output/Triples_Final_All_Relevant --api_key YOUR_API_KEY
"""

from openai import OpenAI
import pandas as pd
import os
import requests
import time

def gpt_authenticate(API_key):
    return OpenAI(api_key=API_key)

def is_url_accessible(url, timeout=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    try:
        head = requests.head(url, headers=headers, timeout=timeout)
        if head.status_code == 200 and 'image' in head.headers.get("Content-Type", ""):
            return True
        # Fallback to GET
        get = requests.get(url, headers=headers, stream=True, timeout=timeout)
        return get.status_code == 200 and 'image' in get.headers.get("Content-Type", "")
    except requests.RequestException as e:
        print(f"⚠️ URL access error for {url}: {e}")
        return False

def gpt_extract(client, url):
    """
    Sends an image URL to GPT-4o to extract structured pathophysiological triples.

    Args:
        client (OpenAI): Authenticated API client.
        url (str): Image URL.

    Returns:
        str: Raw GPT output with extracted mechanisms and triples.
    """
    response = client.chat.completions.create(
    model="gpt-4o",

## Prompt_3
    # messages=[
    #   {
    #     "role": "user",
    #     "content": [
    #       {"type": "text", "text": '''Describe the image (Figure/Graphical abstract) from an article on comorbidity between COVID-19 and Neurodegeneration.   
    #                                   1. Name potential mechanisms (pathophysiological processes) of Covid-19's impact on the brain depicted in the image. 
    #                                   2. Describe each process depicted in the image as semantic triples (subject–predicate–object).  
    #                                   Example: 
    #                                   Pathophysiological Process: Astrocyte_Activation 
    #                                   Triples:
    #                                   SARS-CoV-2_infection|triggers|astrocyte_activation
                                      
    #                                   Use ONLY the information shown in the image! Follow the structure precisely and don't write anything else! Replace spaces in names with _ sign, make sure that words "Pathophysiology Process:" and "Triples:" are presented, don't use bold font and margins. Each triple must contain ONLY THREE elements separated by a | sign, four and more are not allowed! The predicates should be chosen according to the provided example, and the objects should represent specific biological elements or conditions.
    #                                   Predicate examples: increases, decreases, causes, may_cause, leads_to, may_lead_to, affects, may_affect. 
                                      
    #                                   Structure your answer exactly as it is shown in the example. Don't write anything else!'''},

## Prompt_2
    # messages=[
    #   {
    #     "role": "user",
    #     "content": [
    #       {"type": "text", "text": '''Describe the image (Figure/Graphical abstract) from an article on comorbidity between COVID-19 and Neurodegeneration.   
    #                                   1. Name potential mechanisms (pathophysiological processes) of Covid-19's impact on the brain depicted in the image. 
    #                                   2. Describe each process depicted in the image as semantic triples (subject–predicate–object).  
    #                                   Example: 
    #                                   Pathophysiological Process: Astrocyte_Activation 
    #                                   Triples:
    #                                   SARS-CoV-2_infection|triggers|astrocyte_activation
                                      
    #                                   Use ONLY the information shown in the image! Follow the structure precisely and don't write anything else! Replace spaces in names with _ sign, make sure that words "Pathophysiological Process:" and "Triples:" are presented, don't use bold font and margins. Each triple must contain ONLY THREE elements separated by a | sign, four and more are not allowed! The predicate for each triple must be taken only from the Predicate List! 
                                      
    #                                   Predicate List: increases, decreases, association, positive_correlation, negative_correlation, regulates, causes_no_change, directly_increases, is_a, directly_decreases, has_variant, biomarker_for, has_members, has_components, orthologous, equivalent_to, has_component, translated_to, prognostic_biomarker_for, has_member, transcribed_to, rate_limiting_step_of, analogous_to. 
    #                                   Structure your answer exactly as it is shown in the example. Don't write anything else!'''},
        
## Prompt_1
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": '''Describe the image (Figure/Graphical abstract) from an article on comorbidity between COVID-19 and Neurodegeneration.   
                                      1. Name potential mechanisms (pathophysiological processes) of Covid-19's impact on the brain depicted in the image. 
                                      2. Describe each process depicted in the image as semantic triples (subject–predicate–object).  
                                      Example: 
                                      Pathophysiological Process: Astrocyte_Activation 
                                      Triples:
                                      SARS-CoV-2_infection|triggers|astrocyte_activation
                                      
                                      Use ONLY the information shown in the image! Follow the structure precisely and don't write anything else! Replace spaces in names with _ sign, make sure that words "Pathophysiology Process:" and "Triples:" are presented, don't use bold font and margins. Each triple must contain ONLY THREE elements separated by a | sign, four and more are not allowed!'''},
          {
            "type": "image_url",
            "image_url": {
              "url": url,
            },
          },
        ],
      }
    ],
    max_tokens=2000,
    temperature=0.25,    #parameters    
    top_p=0.25   #parameters
  )
    content = response.choices[0].message.content
    return content

def triples_extraction_from_urls(input_path, output_path_base, API_key):
    client = gpt_authenticate(API_key)
    df = pd.read_excel(input_path)

    parsed_data = []

    for idx, row in df.iterrows():
        image_number = row["Image_number"]
        original_url = row["URL"]
        github_url = row["GitHub_URL"]

        if not is_url_accessible(github_url):
            print(f"❌ Skipping inaccessible URL: {github_url}")
            continue

        try:
            time.sleep(1.5)
            content = gpt_extract(client, github_url)
            print(f"\n{github_url}\n{content}")

            mechanisms = content.strip().split('Pathophysiological Process: ')
            for mechanism_block in mechanisms[1:]:
                lines = mechanism_block.strip().split('\n')
                mechanism_name = lines[0].strip()
                triples = lines[2:]  # Skip "Triples:" line

                for triple in triples:
                    subject, predicate, obj = triple.strip().split('|')
                    parsed_data.append([image_number, original_url, github_url,
                                        mechanism_name, subject, predicate, obj])
        except Exception as e:
            print(f"⚠️ Error processing {github_url}: {e}")
            continue

    parsed_df = pd.DataFrame(parsed_data, columns=[
        'Image_number', 'URL', 'GitHub_URL',
        'Pathophysiological Process', 'Subject', 'Predicate', 'Object'
    ])

    os.makedirs(os.path.dirname(output_path_base), exist_ok=True)
    parsed_df.to_csv(f"{output_path_base}.csv", index=False)
    parsed_df.to_excel(f"{output_path_base}.xlsx", index=False)

    print(f"✅ Triples saved to:\n- {output_path_base}.csv\n- {output_path_base}.xlsx")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract semantic triples from biomedical images using OpenAI GPT-4o.")
    parser.add_argument("--input", required=True, help="Path to Excel file with image URLs (columns: 'Image_number', 'URL', 'GitHub_URL').")
    parser.add_argument("--output", required=True, help="Output file base path without extension.")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key.")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("No API key provided. Use --api_key or set the OPENAI_API_KEY environment variable.")

    triples_extraction_from_urls(args.input, args.output, args.api_key)

# python src/Triple_Extraction_GPT4o.py --input data/prompt_engineering/cbm_files/CBM_subset_50_URLs.xlsx --output data/prompt_engineering/gpt_files/GPT_subset_triples_prompt1_param00 --api_key YOUR_API_KEY
# python src/Triple_Extraction_GPT4o.py --input data/prompt_engineering/cbm_files/CBM_subset_50_URLs.xlsx --output data/prompt_engineering/gpt_files/GPT_subset_triples_prompt2_param00 --api_key YOUR_API_KEY
# python src/Triple_Extraction_GPT4o.py --input data/prompt_engineering/cbm_files/CBM_subset_50_URLs.xlsx --output data/prompt_engineering/gpt_files/GPT_subset_triples_prompt3_param00 --api_key YOUR_API_KEY
# python src/Triple_Extraction_GPT4o.py --input data/prompt_engineering/cbm_files/CBM_subset_50_URLs.xlsx --output data/prompt_engineering/gpt_files/GPT_subset_triples_prompt1_param0_25 --api_key YOUR_API_KEY
