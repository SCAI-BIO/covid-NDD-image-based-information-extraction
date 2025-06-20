# Image_Enrichment_Analysis.py

"""
Image Enrichment and De-duplication Script
Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 23/10/2024

Description:
    This script automates image enrichment using Google Images and Selenium.
    It retrieves and compares image URLs, removes perceptual duplicates, and retains
    only the highest-resolution image among near-duplicates.

    Steps:
    1. Google image search via Selenium
    2. Retrieval of main and similar image URLs
    3. Perceptual hashing using imagehash
    4. Hamming distance to detect near-duplicates
    5. Duplicate removal based on resolution
    6. Final export to Excel files

Input (via command-line arguments):
    --query (str): Search query for Google Images
    --main (int): Number of main images to process
    --similar (int): Number of similar/faceted images to process per main image
    --output_raw (str): Output filename for raw scraped image URLs (without extension)
    --output_clean (str): Output filename for cleaned image list (without extension)
    --outdir (str): Output directory to save the Excel files

Output:
    - [output_raw].xlsx: Raw image list including all retrieved URLs
    - [output_clean].xlsx: Cleaned list of images with duplicates removed

Requirements:
    - selenium
    - pillow
    - requests
    - imagehash
    - pandas
    - matplotlib (for visual inspection)

Usage:
    python src/Image_Enrichment_Analysis.py \
        --query "Covid-19 and Neurodegeneration" \
        --main 5 \
        --similar 3 \
        --output_raw Enrichment_Search_URLs_0 \
        --output_clean Enrichment_Cleaned_0 \
        --outdir ./data/enrichment_data

Warning:
    When the Chrome browser window opens during scraping, you may need to manually accept cookies or close pop-ups
    (e.g., "Accept All", "Decline", etc.). Please monitor the browser briefly to ensure smooth operation.
"""


import os
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import StaleElementReferenceException, ElementClickInterceptedException, NoSuchElementException
import random
import requests
from PIL import Image
from io import BytesIO
import imagehash
from requests.exceptions import ConnectionError, Timeout
from http.client import RemoteDisconnected
import matplotlib.pyplot as plt


def find_thumbnails_on_page(driver, num_images):
    """
    Scrolls the page to load image thumbnails and attempts to find elements representing images using multiple CSS selectors.

    Args:
        driver (webdriver.Chrome): The WebDriver instance controlling the browser.
        num_images (int): Number of images to load on the page by scrolling.

    Returns:
        list: A list of WebElement objects representing the image thumbnails found.
    """
    for _ in range(int(num_images/4)):
        try:
            body = driver.find_element(By.TAG_NAME, "body")
            body.send_keys(Keys.PAGE_DOWN)
            #time.sleep(0.3)
            time.sleep(random.uniform(0.2, 0.5))
        except StaleElementReferenceException:
            print("StaleElementReferenceException encountered. Retrying...")
            continue
    
    selectors = [".rg_i.Q4LuWd", ".isv-r.PNCib.MSM1fd.BUooTd img", ".rg_i", ".H8Rx8c img"]
    thumbnails = []

    for selector in selectors:
        try:
            print(f"Trying selector '{selector}'")
            thumbnails = driver.find_elements(By.CSS_SELECTOR, selector)
            print(f"Tried selector '{selector}': found {len(thumbnails)} elements")
            if thumbnails:
                break
        except Exception as e:
            print(f"Error with selector '{selector}': {str(e)}")

    if not thumbnails:
        print("No thumbnails found with the selectors.")
        driver.quit()
        return
    
    return thumbnails


def find_image_URL(some_url_list, some_thumbnail, driver):
    """
    Extracts the thumbnail and full image URLs from a given thumbnail WebElement and appends them to a dictionary of URLs.

    Args:
        some_url_list (dict): A dictionary where 'thumbnail_url' and 'image_url' keys store the extracted URLs.
        some_thumbnail (WebElement): The thumbnail WebElement from which to retrieve URLs.
        driver (webdriver.Chrome): The WebDriver instance controlling the browser.

    Returns:
        None: Updates the `some_url_list` with the URLs.
    """
    
    # Retrieve thumbnail URL
    thumbnail_url = some_thumbnail.get_attribute("src")
    some_url_list['thumbnail_url'].append(thumbnail_url if thumbnail_url else 'Not_found')
            
    # Try to find the full image URL
    try:
        image_url_button = driver.find_element(By.CSS_SELECTOR, ".sFlh5c.FyHeAf.iPVvYb")
        image_url = image_url_button.get_attribute("src")
        some_url_list['image_url'].append(image_url if image_url else 'Not_found')
    except NoSuchElementException:
        some_url_list['image_url'].append('Not_found')
        print("No image URL found")


def google_image_search(query, download_path, num_images_main, num_images_similar, output_raw_filename):
    """
    Conducts a Google Image search for a given query, collects image URLs, and saves the results to an Excel file.
    It processes both main and similar images from the search results.

    Args:
        query (str): The search query for Google Images.
        download_path (str): The path where the output Excel file will be saved.
        num_images_main (int): Number of main images to process.
        num_images_similar (int): Number of similar images to process after clicking "See more".
        output_raw_filename (str): Desired filename (without extension) for the raw image URL list.

    Returns:
        pandas.DataFrame: A DataFrame containing the thumbnail and image URLs found during the search.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36")
    driver = webdriver.Chrome(options=options)
    
    search_url = f"https://www.google.com/search?tbm=isch&q={query}"
    driver.get(search_url)
    print(f"Opened URL: {search_url}")
    
    time.sleep(random.uniform(2, 5))

    thumbnails = find_thumbnails_on_page(driver, num_images_main) #integrate find_thumbnails_on_page function
    print("thumbnails_0 ",thumbnails[0])

    url_list = {'thumbnail_url': [], 'image_url': []}
    
    #for thumbnail_num in range(len(thumbnails[:num_images_main])+1):
    for thumbnail_num in range(len(thumbnails[:num_images_main])):
        print("Processing image...")
        
        thumbnail = thumbnails[thumbnail_num]

        try:
            thumbnail.click()
            time.sleep(random.uniform(2, 5))
            
            find_image_URL(url_list, thumbnail, driver)

            # Look for the "See more" button and click it if found
            try:
                see_more_button = driver.find_element(By.CSS_SELECTOR, ".DFaQu.T38yZ.cS4Vcb-pGL6qe-lfQAOe")
                if see_more_button:
                    see_more_button.click()
                    print("Clicked 'See more' button.")
                    
                    #collect thumbnails
                    thumbnails_ = find_thumbnails_on_page(driver, num_images_similar) #integrate find_thumbnails_on_page function
                    
                    for thumbnail_ in thumbnails_[:num_images_similar]:
                        print("Processing image (similar)...", thumbnail_)
                        
                        try:
                            thumbnail_.click()
                            time.sleep(random.uniform(2, 5))

                            #Image URL search
                            find_image_URL(url_list, thumbnail_, driver)
                            
                        except (StaleElementReferenceException, ElementClickInterceptedException, NoSuchElementException) as e:
                            print(f"Exception encountered: {str(e)}. Skipping...")
                            url_list['thumbnail_url'].append('Processing_problems')
                            url_list['image_url'].append('Processing_problems')
                            continue
                    
                    # Wait for additional images to load
                    time.sleep(random.uniform(2, 5))
            except NoSuchElementException:
                print("No 'See more' button found.")
                

            # Go back to the previous page (Google Image search results) using driver.back()
            try:
                back_to_main_search_button = driver.find_element(By.CSS_SELECTOR, ".xZaYFf.VAyT2")
                print("Trying to find 'Back to main search' button.")
                body = driver.find_element(By.TAG_NAME, "body")  # Locate the body element
                body.send_keys(Keys.HOME)  # Simulate pressing the "Home" key to scroll to the top
                time.sleep(random.uniform(2, 5))  # Pause for a second to allow the page to scroll up
                print("Scrolled to the top of the page.")
                if back_to_main_search_button:
                    back_to_main_search_button.click()
                    print("Clicked 'Back to main search' button.")
                    time.sleep(random.uniform(2, 5))  # Wait a moment for the main page to load again
                    
            except NoSuchElementException:
                print("No 'Back to main search' button found.")

            thumbnails = find_thumbnails_on_page(driver, num_images_main) #integrate find_thumbnails_on_page function
            print("thumbnails_1: ",thumbnails[0])

                
        except (StaleElementReferenceException, ElementClickInterceptedException, NoSuchElementException) as e:
            print(f"Exception encountered: {str(e)}. Skipping...")
            url_list['thumbnail_url'].append('Processing_problems')
            url_list['image_url'].append('Processing_problems')
            continue

    driver.quit()

    # Ensure all lists are the same length
    min_length = min(len(url_list['thumbnail_url']), len(url_list['image_url']))
    for key in url_list.keys():
        url_list[key] = url_list[key][:min_length]

    data_table = pd.DataFrame(url_list)

    file_path = os.path.join(download_path, f"{output_raw_filename}.xlsx")
    data_table.to_excel(file_path)
    print(f"Saved data to {file_path}")

    return data_table


# Comparing images and removing duplicates based on resolution.

def get_image_hash(image_url, retries=3, timeout=10):
    """
    Downloads an image from the provided URL and calculates its perceptual hash (pHash). Implements retry logic in case of network errors.

    Args:
        image_url (str): The URL of the image to download.
        retries (int, optional): Number of retry attempts in case of download failures. Default is 3.
        timeout (int, optional): Timeout for the image download request in seconds. Default is 10.

    Returns:
        ImageHash or None: The perceptual hash of the image, or None if an error occurs during download or processing.
    """
    attempt = 0
    
    while attempt < retries:
        try:
            # Send a request to the URL to get the image with a timeout
            response = requests.get(image_url, timeout=timeout)
            response.raise_for_status()  # Check if the request was successful
            
            # Open the image and calculate the hash
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return imagehash.phash(image)  # Or use .ahash(), .dhash(), etc.
        
        except (ConnectionError, Timeout, RemoteDisconnected) as e:
            attempt += 1
            print(f"Attempt {attempt}/{retries} failed for {image_url}: {e}")
            if attempt < retries:
                print(f"Retrying after a short delay...")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                print(f"Failed to download image after {retries} attempts: {image_url}")
                return None
        
        except Exception as e:
            print(f"Error processing {image_url}: {e}")
            return None

    
def process_images_from_dataframe(df, n=None):
    """
    Downloads images from URLs in the given DataFrame, calculates their perceptual hashes, and stores the results in a dictionary.

    Parameters:
        df (pd.DataFrame): DataFrame containing the image URLs.
        output_folder (str): Folder to store downloaded images (if needed).
        n (int): Number of images to process from the DataFrame.

    Returns:
        dict: Dictionary where keys are image filenames and values are the hashes.
    """

    # Dictionary to store image URLs and their hashes
    image_hashes = {}
    
    # Process only the first `n` rows if needed
    if n:
        df_subset = df.head(n)
        print(f"Taking the first {n} URLs")
    else:
        df_subset = df
        print("n is not provided or is None")
    
    # Loop through each image URL in the DataFrame
    for index, row in df_subset.iterrows():
        image_url = row['image_url']
        
        # Get the hash of the image
        img_hash = get_image_hash(image_url)

        if img_hash is not None:
                
            # Store the hash and the image URL or filename in the dictionary
            image_hashes[image_url] = img_hash
            
            print(f"Image {index+1} processed and hash assigned.")

    return image_hashes


def display_image_pairs(image1_url, image2_url):
    """
    Downloads and displays two images side by side in a Matplotlib figure for visual comparison.

    Args:
        image1_url (str): The URL of the first image.
        image2_url (str): The URL of the second image.

    Returns:
        None: Displays the two images in a single figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Download and display the first image
    response1 = requests.get(image1_url)
    image1 = Image.open(BytesIO(response1.content))
    axes[0].imshow(image1)
    axes[0].axis('off')
    axes[0].set_title(f'Image 1: {os.path.basename(image1_url)}')

    # Download and display the second image
    response2 = requests.get(image2_url)
    image2 = Image.open(BytesIO(response2.content))
    axes[1].imshow(image2)
    axes[1].axis('off')
    axes[1].set_title(f'Image 2: {os.path.basename(image2_url)}')

    plt.show()
    
    
# Function to download image and get its resolution (width x height)
def get_image_resolution(url):
    """
    Downloads an image from the provided URL and returns its resolution (width and height).

    Args:
        url (str): The URL of the image to download.

    Returns:
        tuple or None: A tuple (width, height) representing the image's resolution, or None if an error occurs during download or processing.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        image = Image.open(BytesIO(response.content))
        return image.size  # Returns (width, height)
    except Exception as e:
        print(f"Error downloading or processing image at {url}: {e}")
        return None

def remove_lower_resolution_duplicate(df, duplicates_dict):
    """
    Compares the resolutions of duplicate image URLs in the DataFrame and removes the one with the lower resolution.
    If both images have the same resolution, the second URL is removed. If an image cannot be downloaded, it skips the comparison.

    Args:
        df (pd.DataFrame): The DataFrame containing image URLs.
        duplicates_dict (dict): Dictionary of duplicate URL pairs (url1, url2) to be compared.

    Returns:
        pd.DataFrame: The DataFrame with the lower resolution duplicates removed.
    """
    # Iterate through the duplicate pairs
    for url1, url2 in duplicates_dict:
        # Check if both URLs are in the DataFrame
        if url1 in df['image_url'].values and url2 in df['image_url'].values:
            # Get resolutions of both images
            res1 = get_image_resolution(url1)
            res2 = get_image_resolution(url2)
            print(res1, res2)

            if res1 and res2:
                # Compare resolutions (width * height for total pixel count)
                pixels1 = res1[0] * res1[1]
                pixels2 = res2[0] * res2[1]

                # Remove the URL with the lower resolution
                if pixels1 < pixels2:
                    df = df[df['image_url'] != url1]  # Remove url1 if it has lower resolution
                elif pixels2 < pixels1:
                    df = df[df['image_url'] != url2]  # Remove url2 if it has lower resolution
                else:
                    df = df[df['image_url'] != url2]  # Remove url2 if it has equal resolution
            else:
                print(f"Skipping comparison for URLs {url1} and {url2} due to download issues.")
    
    return df


def dataset_purification(raw_df, download_path, output_clean_filename):
    """
    Cleans and deduplicates image URLs by:
    - Removing invalid and duplicate entries
    - Detecting near-duplicates using perceptual hashing
    - Keeping higher-resolution images
    - Saving the cleaned dataset to Excel

    Args:
        raw_df (pd.DataFrame): DataFrame with image URLs.
        download_path (str): Directory to save the final file.
        output_clean_filename (str): Name of the final cleaned Excel file (without extension).
    """

    # First purification
    raw_df.drop(raw_df[raw_df.image_url == "Processing_problems"].index, inplace=True)
    raw_df = raw_df.drop_duplicates(subset='image_url')
    raw_df = raw_df.drop(raw_df[raw_df.image_url == "Not_found"].index)
    raw_df.reset_index(drop=True, inplace=True)
    
    # Compare hashes to find near-duplicates
    duplicates = []
    image_hashes = process_images_from_dataframe(raw_df)
    hash_values = list(image_hashes.values())
    for i, hash1 in enumerate(hash_values):
        for j, hash2 in enumerate(hash_values[i+1:], i+1):
            if hash1 - hash2 < 8:
                duplicates.append((list(image_hashes.keys())[i], list(image_hashes.keys())[j]))
    
    print(f"Found {len(duplicates)} near-duplicate image pairs:")
    for dup in duplicates:
        print(f"Duplicate pair: {dup[0]} and {dup[1]}")
        display_image_pairs(dup[0], dup[1])
        
    df_cleaned = remove_lower_resolution_duplicate(raw_df, duplicates)
    
    file_path = os.path.join(download_path, f"{output_clean_filename}.xlsx")
    df_cleaned.to_excel(file_path)
    print(f"File saved successfully as {file_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image enrichment and duplicate removal using Google Images + perceptual hashing")
    parser.add_argument("--query", required=True, help="Search query for Google Images (use quotes for multi-word queries)")
    parser.add_argument("--main", type=int, default=100, help="Number of main search images to collect (default: 100)")
    parser.add_argument("--similar", type=int, default=100, help="Number of similar/faceted images to collect (default: 100)")
    parser.add_argument("--output_raw", required=True, help="Filename (without extension) for raw image output")
    parser.add_argument("--output_clean", required=True, help="Filename (without extension) for cleaned de-duplicated output")
    parser.add_argument("--outdir", default="./data/enrichment_data", help="Output directory (default: ./data/enrichment_data)")
    
    args = parser.parse_args()

    # Run image search and save raw URLs
    data_table = google_image_search(
        query=args.query,
        download_path=args.outdir,
        num_images_main=args.main,
        num_images_similar=args.similar,
        output_raw_filename=args.output_raw
    )

    # Run cleaning and save cleaned file
    dataset_purification(data_table, download_path=args.outdir, output_clean_filename=args.output_clean)

