# URLs_Relevance_Check.py

"""
Relevance Classification of Image URLs using OpenAI GPT
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 23/10/2024

Description:
    This script performs automatic relevance assessment of image URLs extracted from biomedical publications.
    The goal is to identify images that are relevant to the topic of Covid-19 and neurodegeneration using GPT-4o.

    Key functionalities:
    1. Accessibility check: Ensures the image URLs point to valid, retrievable images.
    2. Relevance classification: Uses the OpenAI GPT-4o model to classify each image as relevant or irrelevant
       based on predefined bioinformatics-focused criteria.
    3. Result export: Saves full classification results and filters out only the relevant images.

Input:
    - Excel file containing image URLs (column name: 'image_url')
    - OpenAI API key (as argument or environment variable)

Output:
    - "Relevance_assignment_GPT_4o.xlsx": All image URLs with classification labels.
    - "Relevant_URLs_only_GPT_4o.xlsx": Filtered table with only the relevant URLs.

Requirements:
    - Python libraries: openai, pandas, requests
    - OpenAI API access (GPT-4o)

Usage:
    Provide input and output arguments via CLI:
    python src/URLs_Relevance_Check.py --input data/enrichment_data/Enrichment_Search_URLs.xlsx --api_key YOUR_API_KEY

    Or (preferred) store your key as an environment variable:
    export OPENAI_API_KEY=sk-...
    python src/URLs_Relevance_Check.py --input data/enrichment_data/Enrichment_Search_URLs.xlsx
"""

import os
import requests
import pandas as pd
import time
from openai import OpenAI
from http.client import RemoteDisconnected


def gpt_authenticate(API_key):
    """
    Authenticates with the OpenAI GPT API using a predefined API key and returns the authenticated client object.
    
    Returns:
        client (OpenAI): Authenticated OpenAI API client.
    """
    client = OpenAI(api_key = API_key)
    return client


def check_image_url(url, retries=3):
    """
    Checks the accessibility of a given image URL by sending an HTTP GET request and handling retries on failure.
    It validates whether the URL points to an image by examining the Content-Type header.

    Args:
        url (str): The URL of the image to be checked.
        retries (int, optional): Number of retry attempts in case of connection failures. Default is 3.

    Returns:
        bool: Returns False if the URL is accessible and contains an image; True if an error occurred or the content is not an image.
    """
    attempt = 0
    while attempt < retries:
        try:
            # Send a GET request to the image URL
            response = requests.get(url, timeout=5)  # Set a timeout to avoid hanging
            
            # Check if the response status code is 200 (OK)
            if response.status_code == 200:
                # Get the Content-Type header from the response
                content_type = response.headers.get('Content-Type', '')

                # Check if the Content-Type is an image format
                if 'image' in content_type:
                    print(f"Success: URL {url} is accessible and is of type {content_type}.")
                    return False  # No error
                else:
                    print(f"Error: URL {url} is not an image. Content-Type: {content_type}")
                    return True  # Error occurred
            else:
                print(f"Error: URL {url} returned status code {response.status_code}.")
                return True  # Error occurred

        except (requests.ConnectionError, requests.Timeout, RemoteDisconnected) as e:
            attempt += 1
            print(f"Attempt {attempt}/{retries} failed for {url}: {e}")
            if attempt < retries:
                print("Retrying after a short delay...")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                print(f"Failed to access URL after {retries} attempts: {url}")
                return True  # Error occurred after retries
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return True  # Error occurred due to unknown exception

    return True  # Fallback in case of any unexpected behavior


def URLs_access_check(dataframe, retries=3):
    """
    Iterates over a DataFrame containing image URLs, checks the accessibility of each URL, and stores the results.
    
    Args:
        dataframe (pandas.DataFrame): DataFrame containing a column 'image_url' with image URLs.
        retries (int, optional): Number of retry attempts in case of connection failures. Default is 3.

    Returns:
        pandas.DataFrame: DataFrame containing two columns: 'URL' and 'Access' (Yes/No), indicating the accessibility of each URL.
    """
    access_data = []
    for idx, row in dataframe.iterrows():
        url = row['image_url']

        # Check if the image URL is accessible with retry mechanism
        access = "No" if check_image_url(url, retries=retries) else "Yes"
        access_data.append([url, access])

    # Create a new DataFrame to store the results
    access_df = pd.DataFrame(access_data, columns=['URL', 'Access'])

    return access_df


def gpt_extract(client, url):
    """
    Uses GPT to analyze an image and determine its relevance based on predefined criteria. 
    The URL is passed to the GPT model, which returns either "Yes" or "No" indicating the relevance of the image.

    Args:
        client (OpenAI): The authenticated OpenAI API client.
        url (str): The URL of the image to be analyzed.

    Returns:
        str: The GPT's assessment of the image's relevance ("Yes" or "No").
    """
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": '''Image URL is given. Analyze this image and assign relevance to it: "Yes" for relevant images, "No" for irrelevant images, and "Uncertain" if the relevance cannot be assessed. Follow this classification:
                                      Relevant:
                                      Images which:
                                      Сlearly demonstrate relationship between Covid-19 and neurodegeneration (any neurological impacts).
                                      Don't contain lots of text (no more than 500 characters).
                                      Don't just depict a research outline.
                                      Don't be just graphs or represent photos derived from scientific tools (microscopic, histological images) and data visualization.
                                      Are likely to be cartoons drawn by article authors.
                                      
                                      Irrelevant:
                                      Unrelated images, for example just an image of a virus particle or a sick person.
                                      Images where correct interpretation of the data is impossible.
                                      Images which display insights into Covid-19 OR Neurodegeneration, if one is present and the other is missing.
                                      
                                      Uncertain:
                                      The relevance of the image cannot be confidently determined based on the visual content.
                                      
                                      Your answer should contain only a final decision in the following format: No/Yes/Uncertain (without dots)
                                      Don't write anything else!'''},
           {
            "type": "image_url",
            "image_url": {
              "url": url,
            },
          },
        ],
      }
    ],
    max_tokens=900,
  )
    content = response.choices[0].message.content
    return content


def get_GPT_answers(dataframe, API_key):
    """
    Authenticates with GPT, processes a DataFrame containing image URLs, and retrieves the relevance classification for each image.

    Args:
        dataframe (pandas.DataFrame): DataFrame containing a column 'image_url' with image URLs.

    Returns:
        list: A list of lists where each inner list contains the image URL and the GPT's relevance classification or "Error" in case of failure.
    """
    client = gpt_authenticate(API_key)
    parsed_data = []

    for idx, row in dataframe.iterrows():
        try:
            url = row["image_url"]  # Extract image URL
            relev_check = gpt_extract(client, url) # Extract content from the image using GPT
            print(url, relev_check)
            parsed_data.append([url, relev_check])

        except Exception as e:
            # Print the error and continue to the next row
            print(f"Error processing {idx, row}: {e}")
            parsed_data.append([url, "Error"])
            continue  # Skip to the next row in case of an error
    
    return parsed_data


def relevance_check_main(input_path, output_path, api_key):
    """
    Runs the full relevance-check pipeline for a dataset of image URLs.

    Steps:
    1. Loads an Excel file containing image URLs.
    2. Sends each image to GPT-4o for classification (Yes / No / Error).
    3. Exports two Excel files:
    - One with all URLs and GPT relevance labels.
    - One with only relevant URLs (no label column).

    Args:
        input_path (str): Full path to the input Excel file.
        output_path (str): Directory where output Excel files will be saved.
        api_key (str): OpenAI API key.

    Outputs:
        - Relevance_assignment_GPT_4o.xlsx
        - Relevant_URLs_only_GPT_4o.xlsx
    """

    # Convert the Excel file into dataframe
    data_raw = pd.read_excel(input_path)

    # Drop 'Unnamed: 0' only if it exists in the DataFrame
    if 'Unnamed: 0' in data_raw.columns:
        data_raw.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Get relevance data
    parsed_data_all = get_GPT_answers(data_raw, api_key)
    
    # Loop through the list and clean up 'Yes.' or 'No.' to 'Yes' or 'No'
    for item in parsed_data_all:
        # Check if the second element ends with a period and is 'Yes.' or 'No.'
        if item[1] in ['Yes.', 'No.']:
            item[1] = item[1].rstrip('.')
            
    # Create a new DataFrame to store parsed results
    relevance_GPT_all = pd.DataFrame(parsed_data_all, columns=['URL', 'Relevance_GPT'])
    relevance_GPT_all.to_excel(os.path.join(output_path, "Relevance_assignment_GPT_4o.xlsx"), index=False)
    
    # Store only relevant URLs
    relevant_URLs = relevance_GPT_all[relevance_GPT_all['Relevance_GPT'] == 'Yes']
    relevant_URLs.reset_index(drop=True, inplace=True)
    relevant_URLs[['URL']].to_excel(os.path.join(output_path, "Relevant_URLs_only_GPT_4o.xlsx"), index=False)
    
    # Number of images which GPT can't process
    print('Initial total number of URLs:', len(data_raw))
    print("Number of images which GPT can't process",len(relevance_GPT_all.loc[relevance_GPT_all['Relevance_GPT'] == 'Error']))
    print('Number of relevant images (GPT):', len(relevant_URLs))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GPT-4o-based relevance filtering for biomedical image URLs.")
    parser.add_argument("--input", required=True, help="Path to input Excel file (containing 'image_url' column).")
    parser.add_argument("--output_dir", default="./data/URL_relevance_analysis", help="Directory where result files will be saved.")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key. Can also be set as OPENAI_API_KEY environment variable.")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("No API key provided. Use --api_key or set the OPENAI_API_KEY environment variable.")

    # Run main processing
    API_key = args.api_key
    relevance_check_main(args.input, args.output_dir, args.api_key)


