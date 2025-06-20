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
    python src/Triple_Extraction_GPT4o.py --input data/URL_relevance_analysis/Final_Relevant_URLs.xlsx --output_dir ./triples_output --api_key YOUR_API_KEY
"""

from openai import OpenAI
import pandas as pd
import os

def gpt_authenticate(API_key):
    """
    Authenticates with the OpenAI GPT API using the provided API key.

    Args:
        API_key (str): OpenAI API key.

    Returns:
        OpenAI: Authenticated API client.
    """
    return OpenAI(api_key=API_key)


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
    temperature=0.0,    #parameters    
    top_p=0.0   #parameters
  )
    content = response.choices[0].message.content
    return content

def triples_extraction_from_urls(input_path, output_dir, API_key):
    """
    Main function to process image URLs, extract semantic triples using GPT, and save results.

    Args:
        input_path (str): Path to the input Excel file with image URLs.
        output_dir (str): Directory where the output files will be saved.
        API_key (str): OpenAI API key.
    """
    client = gpt_authenticate(API_key)
    df = pd.read_excel(input_path)

    parsed_data = []

    for idx, row in df.iterrows(): 
        try:
            url = row["URL"]
            content = gpt_extract(client, url)
            print(url, content)

            # Parse mechanisms
            mechanisms = content.strip().split('Pathophysiological Process: ')
            for mechanism_block in mechanisms[1:]:
                lines = mechanism_block.strip().split('\n')
                mechanism_name = lines[0].strip()
                triples = lines[2:]  # skip the "Triples:" line

                for triple in triples:
                    subject, predicate, obj = triple.strip().split('|')
                    parsed_data.append([url, mechanism_name, subject, predicate, obj])
        except Exception as e:
            print(f"Error processing {url}: {e}")
            continue

    parsed_df = pd.DataFrame(parsed_data, columns=['URL', 'Pathophysiological Process', 'Subject', 'Predicate', 'Object'])

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "Triples_Final_All_Relevant.csv")
    xlsx_path = os.path.join(output_dir, "Triples_Final_All_Relevant.xlsx")

    parsed_df.to_csv(csv_path, index=False)
    parsed_df.to_excel(xlsx_path, index=False)

    print(f"Triples saved to:\n- {csv_path}\n- {xlsx_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract semantic triples from images using OpenAI GPT-4o.")
    parser.add_argument("--input", required=True, help="Path to Excel file with image URLs (column name: 'URL').")
    parser.add_argument("--output_dir", default="./data/triples_output", help="Directory to save output files.")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key (can also be set via OPENAI_API_KEY env variable).")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("No API key provided. Use --api_key or set the OPENAI_API_KEY environment variable.")

    triples_extraction_from_urls(args.input, args.output_dir, args.api_key)

