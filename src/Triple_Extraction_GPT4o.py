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
    - Excel file containing image URLs. For your current file, columns are "Unnamed: 0" and "URL".
      "Unnamed: 0" will be used as image_number; "URL" will be duplicated into "github_url".
    - OpenAI API key (via CLI argument or environment variable).

Output:
    - <output_base>.csv and <output_base>.xlsx: All extracted semantic triples.
    - data/triples_output/triple_extraction_runlog.csv: Run log.

Requirements:
    - openai
    - pandas
    - requests
    - openpyxl
    - Internet connection and OpenAI API access.

Usage:
    python3 src/Triple_Extraction_GPT4o.py \
      --input data/URL_relevance_analysis/Final_Relevant_URLs_test.xlsx \
      --output data/triples_output/Triples_Final_All_Relevant_test \
      --api_key sk-...ugA
"""

import os
import re
import csv
import time
import requests
import pandas as pd
from datetime import datetime
from openai import OpenAI


# -----------------------------
# Utilities
# -----------------------------

def gpt_authenticate(API_key: str) -> OpenAI:
    # If API_key is None, the SDK will look for OPENAI_API_KEY in env
    return OpenAI(api_key=API_key)


def to_raw_github(url: str) -> str:
    """
    Convert GitHub /blob/ URLs to raw.githubusercontent.com for direct image content.
    Leave other URLs as-is.
    """
    m = re.match(r'^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)$', url)
    if m:
        user, repo, branch, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return url


def is_url_accessible(url: str, timeout: int = 10) -> bool:
    """
    Check if the URL returns image content. Tries raw GitHub conversion first.
    """
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/114.0.0.0 Safari/537.36")
    }
    try:
        trial_urls = [to_raw_github(url)]
        if trial_urls[0] != url:
            trial_urls.append(url)

        for u in trial_urls:
            r = requests.head(u, headers=headers, timeout=timeout, allow_redirects=True)
            if r.status_code == 200 and 'image' in r.headers.get("Content-Type", ""):
                return True
            r = requests.get(u, headers=headers, stream=True, timeout=timeout)
            if r.status_code == 200 and 'image' in r.headers.get("Content-Type", ""):
                return True
        return False
    except requests.RequestException as e:
        print(f"⚠️ URL access error for {url}: {e}")
        return False


# -----------------------------
# Parsing helpers (tolerant to minor format drift)
# -----------------------------

HEADER_REGEX = re.compile(r'^\s*Pathophysiolog(?:y|ical)\s+Process:\s*(.+?)\s*$', re.I)
TRIPLES_HEADER_REGEX = re.compile(r'^\s*Triples:\s*$', re.I)
TRIPLE_LINE_REGEX = re.compile(r'^\s*([^|]+)\|([^|]+)\|([^|]+)\s*$')

def parse_mechanisms_and_triples(text: str):
    """
    Parse model output like:

        Pathophysiological Process: Astrocyte_Activation
        Triples:
        SARS-CoV-2_infection|triggers|astrocyte_activation
        ...

    Returns a list of dicts: [{"mechanism": str, "triples": [(s,p,o), ...]}, ...]
    """
    blocks = []
    current = None
    collecting = False
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = HEADER_REGEX.match(line)
        if m:
            current = {"mechanism": m.group(1).strip(), "triples": []}
            blocks.append(current)
            collecting = False
            continue
        if TRIPLES_HEADER_REGEX.match(line):
            collecting = True
            continue
        if collecting and current:
            m3 = TRIPLE_LINE_REGEX.match(line)
            if m3:
                subj, pred, obj = (x.strip() for x in m3.groups())
                current["triples"].append((subj, pred, obj))
            # Silently ignore malformed lines
    return blocks


# -----------------------------
# GPT call
# -----------------------------

PROMPT_TEXT = (
    "Describe the image (Figure/Graphical abstract) from an article on comorbidity between COVID-19 and Neurodegeneration.\n"
    "1. Name potential mechanisms (pathophysiological processes) of Covid-19's impact on the brain depicted in the image.\n"
    "2. Describe each process depicted in the image as semantic triples (subject–predicate–object).\n"
    "Example:\n"
    "Pathophysiological Process: Astrocyte_Activation\n"
    "Triples:\n"
    "SARS-CoV-2_infection|triggers|astrocyte_activation\n\n"
    "Use ONLY the information shown in the image! Follow the structure precisely and don't write anything else!\n"
    "Replace spaces in names with _ sign, make sure that words \"Pathophysiological Process:\" and \"Triples:\" are presented,\n"
    "don't use bold font and margins. Each triple must contain ONLY THREE elements separated by a | sign; four or more are not allowed!"
)

def gpt_extract(client: OpenAI, url: str) -> str:
    """
    Sends an image URL to GPT-4o to extract structured pathophysiological triples.
    """
    response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEXT},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
        max_tokens=2000,
        temperature=0.25,
        top_p=0.25,
        seed=42
    )
    return response.choices[0].message.content or ""


# -----------------------------
# Main extraction pipeline
# -----------------------------

def triples_extraction_from_urls(input_path: str, output_path_base: str, API_key: str):
    # Ensure output directory and log directory exist
    out_dir = os.path.dirname(output_path_base)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    os.makedirs('data/triples_output', exist_ok=True)

    # Open log AFTER ensuring directory exists
    log_path = 'data/triples_output/triple_extraction_runlog.csv'
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Image_Number', 'URL', 'Outcome', 'Finish_Reason', 'HTTP_Status', 'Timestamp'])

    client = gpt_authenticate(API_key)

    # Load Excel and adapt to your current schema (['Unnamed: 0', 'URL'])
    df = pd.read_excel(input_path)

    # image_number from Unnamed: 0 if present; else 1-based index
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "image_number"})
    else:
        df["image_number"] = df.index + 1

    # Standardize URL column, and duplicate to github_url (single URL source)
    if "URL" not in df.columns:
        raise ValueError(f"Input file missing 'URL' column. Found: {list(df.columns)}")
    df = df.rename(columns={"URL": "url"})
    df["github_url"] = df["url"]

    parsed_data = []

    for _, row in df.iterrows():
        image_number = row["image_number"]
        original_url = row["url"]
        github_url = to_raw_github(row["github_url"])

        # Initialize variables for logging
        outcome = "unknown"
        finish_reason = "N/A"
        http_status = "N/A"

        # Check URL accessibility
        if not is_url_accessible(github_url):
            print(f"❌ Skipping inaccessible URL: {github_url}")
            outcome = "inaccessible"
            http_status = "Non-200 or timeout"
            log_writer.writerow([
                image_number,
                github_url,
                outcome,
                finish_reason,
                http_status,
                datetime.now().isoformat()
            ])
            continue

        try:
            time.sleep(1.5)  # small delay for rate-limiting hygiene
            http_status = 200  # URL was accessible

            # Call GPT-4o
            content = gpt_extract(client, github_url)

            # Extract finish reason (best-effort since we use helper)
            finish_reason = "stop"

            # Refusal / content filter heuristics
            refusal_keywords = [
                "i cannot", "i can't", "i am unable",
                "content policy", "i apologize", "i'm sorry", "inappropriate"
            ]
            if any(k in content.lower() for k in refusal_keywords):
                outcome = "refused"
                print(f"⚠️ GPT refused to process: {github_url}")
            else:
                # Parse triples
                blocks = parse_mechanisms_and_triples(content)
                if not blocks:
                    outcome = "no_parse"
                    print(f"⚠️ No mechanisms/triples parsed for: {github_url}\n--- RAW ---\n{content}\n-----------")
                else:
                    outcome = "ok"
                    print(f"\n{github_url}\n{content}")
                    for b in blocks:
                        mech = b["mechanism"]
                        for (subject, predicate, obj) in b["triples"]:
                            parsed_data.append([
                                image_number, original_url, github_url,
                                mech, subject, predicate, obj
                            ])

            # Log outcome
            log_writer.writerow([
                image_number,
                github_url,
                outcome,
                finish_reason,
                http_status,
                datetime.now().isoformat()
            ])

        except Exception as e:
            outcome = "error"
            finish_reason = str(e)
            print(f"⚠️ Error processing {github_url}: {e}")

            # Log error
            log_writer.writerow([
                image_number,
                github_url,
                outcome,
                finish_reason,
                http_status,
                datetime.now().isoformat()
            ])
            continue

    # Close log
    log_file.close()

    # Save parsed triples
    parsed_df = pd.DataFrame(parsed_data, columns=[
        'Image_number', 'URL', 'GitHub_URL',
        'Pathophysiological Process', 'Subject', 'Predicate', 'Object'
    ])

    parsed_df.to_csv(f"{output_path_base}.csv", index=False)
    parsed_df.to_excel(f"{output_path_base}.xlsx", index=False)

    print(f"✅ Triples saved to:\n- {output_path_base}.csv\n- {output_path_base}.xlsx")
    print(f"✅ Run log saved to: {log_path}")


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract semantic triples from biomedical images using OpenAI GPT-4o."
    )
    parser.add_argument("--input", required=True,
                        help="Path to Excel file with image URLs. "
                             "For current file, uses columns 'Unnamed: 0' and 'URL'.")
    parser.add_argument("--output", required=True,
                        help="Output file base path without extension.")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"),
                        help="OpenAI API key (or set OPENAI_API_KEY env var).")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("No API key provided. Use --api_key or set the OPENAI_API_KEY environment variable.")

    triples_extraction_from_urls(args.input, args.output, args.api_key)
