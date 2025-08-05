import urllib.parse
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
def get_doi_from_springer(url):
    """
    Parse out the DOI from Springer image URLs.
    
    Handles variations like:
    - .../image/art%3A10.1007%2Fs12640-020-00219-8/MediaObjects/...
    - URLs with different encoding patterns
    - URLs with slight variations in structure
    
    Args:
        url (str): The Springer image URL
    
    Returns:
        str or None: Extracted DOI
    """
    try:
        # Try multiple extraction strategies
        
        # Strategy 1: Extract after "/image/" and before "/MediaObjects"
        if "/image/" in url:
            path_part = url.split("/image/")[-1]
            decoded = urllib.parse.unquote(path_part)
            
            # Look for DOI patterns
            doi_match = re.search(r'(10\.\d{4,9}/[^\s/]+)', decoded)
            if doi_match:
                return doi_match.group(1)
        
        # Strategy 2: Direct regex search in the full URL
        doi_match = re.search(r'(10\.\d{4,9}/[^\s]+)', urllib.parse.unquote(url))
        if doi_match:
            return doi_match.group(1)
        
        # Strategy 3: Handle art%3A encoded DOIs specifically
        if "art%3A" in url:
            decoded = urllib.parse.unquote(url)
            art_match = re.search(r'art:?(10\.\d{4,9}/[^\s/]+)', decoded)
            if art_match:
                return art_match.group(1)
    
    except Exception as e:
        # Optional: log the error if needed
        print(f"Error extracting DOI: {e}")
    
    return None

def get_doi_from_frontiers(url):
    """
    Attempt to parse the pattern:
      /Articles/<article_id>/<journal>-<volume>-<article_id>-HTML/...
    Then guess the year or build a 10.3389/<journal>.<year>.<article_id> style DOI.
    Example:
      https://www.frontiersin.org/files/Articles/583459/fneur-11-583459-HTML/image_m/fneur-11-583459-g001.jpg
    """
    try:
        parts = url.split("/")
        article_id = parts[-4]        # e.g. "583459"
        journal_vol_id = parts[-3]    # e.g. "fneur-11-583459-HTML"
        jv_parts = journal_vol_id.split("-")  # ["fneur", "11", "583459", "HTML"]

        journal_code = jv_parts[0]    # "fneur"
        volume_str = jv_parts[1]      # "11"

        # A rough guess: if volume=11 => year=2020, volume=12 => year=2021, etc.
        base_year = 2020
        year = base_year + (int(volume_str) - 11)

        doi = f"10.3389/{journal_code}.{year}.{article_id}"
        return doi
    except:
        return None

def get_doi_from_mdpi(url):
    """
    If the URL is something like:
      https://www.mdpi.com/cells/cells-12-00816/article_deploy/html/images/cells-12-00816-g002.png
    We can:
      1) strip off '/html/images...' 
      2) remove '/article_deploy' if present
      3) scrape the resulting page for the <meta name="citation_doi" content="..."> tag
    """
    try:
        base_part = url.split("/html/images")[0]  # e.g. ".../cells/cells-12-00816/article_deploy"
        if base_part.endswith("/article_deploy"):
            base_part = base_part[:-len("/article_deploy")]

        resp = requests.get(base_part)
        soup = BeautifulSoup(resp.text, "html.parser")
        meta_tag = soup.find("meta", attrs={"name": "citation_doi"})
        if meta_tag:
            return meta_tag["content"]
    except:
        pass
    return None

def find_doi(url):
    """
    Master function that tries each known extractor in turn.
    You can add more elif blocks for other publishers.
    """
    if "springernature.com" in url:
        return get_doi_from_springer(url)
    elif "frontiersin.org" in url:
        return get_doi_from_frontiers(url)
    elif "mdpi.com" in url:
        return get_doi_from_mdpi(url)
    # Add more as needed...
    return None

def main():
    # Example: reading from an Excel file named "input.xlsx"
    input_file = "data/Triples_Final_All_Relevant.xlsx"
    output_file = "output_with_doi.xlsx"

    df = pd.read_excel(input_file)

    # Ensure the column with URLs is named 'URL'
    if 'URL' not in df.columns:
        raise ValueError("Expected a column named 'URL' in the Excel file.")

    # Create a new column 'DOI' by applying our find_doi() function
    df['DOI'] = df['URL'].apply(find_doi)

    # Save the result
    df.to_excel(output_file, index=False)
    print(f"DOI extraction complete. See '{output_file}' for the new 'DOI' column.")

if __name__ == "__main__":
    main()
