#!/usr/bin/env python3
# review_labels_neo4j.py
"""
Label Review and Correction of Neo4j Nodes via OpenAI GPT
Author: Elizaveta Popova
Institution: University of Bonn, Fraunhofer SCAI
Date: 06/08/2025

Description:
    Connects to a Neo4j database, reads nodes with a `name` property, and uses the OpenAI GPT API
    to verify or correct each node’s semantic label based on a controlled vocabulary.

    Minimal-tweak version:
      - Removes the log filter that previously hid most output
      - Adds progress logs: fetched count, sample entities, progress heartbeat, final summary

Requirements:
    config/config.ini — must contain the following sections:

    [neo4j]
    uri = bolt://localhost:7687
    user = neo4j
    password = your_password_here

    [openai]
    api_key = your_openai_api_key_here
      
Usage:
    python src/review_labels_neo4j.py
"""

import sys
import time
import logging
import configparser
from pathlib import Path

from neo4j import GraphDatabase, basic_auth
from openai import OpenAI

# ----------------------------
# Logging (INFO level, verbose)
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ----------------------------
# Controlled vocabulary
# ----------------------------
CONTROLLED_VOCAB = [
    "Anatomical_Structure",
    "Biological_Process",
    "Cell",
    "Cell_Phenotype",
    "Chemical",
    "Disease",
    "Gene",
    "Phenotype",
    "Protein",
    "Pathway"
]

def review_label(client: OpenAI, entity: str, current_label: str) -> str:
    """
    Use the OpenAI GPT API to verify or correct the semantic label of a given entity.

    Args:
        client (OpenAI): Initialized OpenAI API client.
        entity (str): The name of the biological entity to classify.
        current_label (str): The entity's current label in the Neo4j database.

    Returns:
        str: The most appropriate label, either unchanged or corrected, based on GPT output.
    """
    prompt = (
        "You are a biomedical ontology expert. Your task is to verify or correct the label for a biological entity.\n\n"
        f"Entity: \"{entity}\"\n"
        f"Current Label: \"{current_label}\"\n\n"
        "Choose the single most appropriate label from the following controlled vocabulary:\n"
        f"{', '.join(CONTROLLED_VOCAB)}\n\n"
        "Rules:\n"
        "- Return only ONE label as plain text (e.g., Gene, Disease).\n"
        "- Do not include punctuation, quotes, extra words, or explanations.\n"
        "- Return the current label exactly as-is if it is already correct.\n"
        "- If none apply, return a single new label that best describes the entity.\n"
        "- If the current label is Unknown, choose the most suitable label; avoid 'Unknown'.\n\n"
        "Output: The label string only."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You classify biological entities."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.0,
            max_tokens=20
        )
        new_label = (resp.choices[0].message.content or "").strip()
        return new_label or current_label
    except Exception:
        logging.exception(f"GPT API error for entity='{entity}'")
        return current_label

def main():
    logging.info("Starting review_labels_neo4j...")

    # ----------------------------
    # Read config.ini (fixed path)
    # ----------------------------
    config_path = Path("config/config.ini").resolve()
    if not config_path.is_file():
        logging.error(f"config.ini not found at {config_path}")
        sys.exit(1)

    cfg = configparser.ConfigParser()
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg.read_file(f)
    except Exception as e:
        logging.error(f"Failed to read config.ini: {e}")
        sys.exit(1)

    # Validate required sections and keys
    for section in ("neo4j", "openai"):
        if section not in cfg:
            logging.error(f"Missing section [{section}] in config.ini")
            sys.exit(1)

    neo4j_conf = cfg["neo4j"]
    openai_conf = cfg["openai"]

    uri      = neo4j_conf.get("uri")
    user     = neo4j_conf.get("user")
    password = neo4j_conf.get("password")
    api_key  = openai_conf.get("api_key")

    if not all([uri, user, password, api_key]):
        logging.error("One or more credentials are missing in config.ini")
        sys.exit(1)

    # ----------------------------
    # Initialize clients
    # ----------------------------
    logging.info("Connecting to OpenAI and Neo4j...")
    client = OpenAI(api_key=api_key)
    try:
        driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
    except Exception:
        logging.exception("Failed to connect to Neo4j")
        sys.exit(1)

    updated = 0
    unchanged = 0

    try:
        with driver.session() as session:
            query = (
                "MATCH (n) WHERE n.name IS NOT NULL "
                "RETURN id(n) AS id, n.name AS entity, labels(n) AS labels"
            )
            try:
                results = list(session.run(query))
            except Exception:
                logging.exception("Cypher query failed")
                return

            total = len(results)
            logging.info(f"Fetched {total} nodes with a 'name' property.")
            if total > 0:
                preview = [r["entity"] for r in results[:5]]
                logging.info(f"Sample entities: {preview}")

            for i, rec in enumerate(results, start=1):
                try:
                    nid     = rec["id"]
                    ent     = rec["entity"]
                    labs    = rec["labels"]

                    # Pick the first known label from CONTROLLED_VOCAB if any
                    current = next((L for L in labs if L in CONTROLLED_VOCAB), "")
                    new_label = review_label(client, ent, current)

                    if new_label != current:
                        # Build Cypher parts
                        remove_unknown = "REMOVE n:`Unknown`" if "Unknown" in labs else ""
                        remove_current = f"REMOVE n:`{current}`" if current else ""
                        add_clause     = f"SET n:`{new_label}`"

                        update_cypher = f"""
                            MATCH (n)
                            WHERE id(n) = $id
                            {remove_unknown}
                            {remove_current}
                            {add_clause}
                        """
                        session.run(update_cypher, id=nid)
                        updated += 1
                        logging.info(f"[UPDATED] '{ent}': {current or '<none>'} → {new_label}")
                    else:
                        unchanged += 1
                        logging.info(f"[UNCHANGED] '{ent}' remains '{current or '<none>'}")

                except Exception:
                    logging.exception(f"Error processing record #{i}")
                finally:
                    # Heartbeat
                    if i % 50 == 0 or i == total:
                        logging.info(f"Progress: {i}/{total} nodes processed...")
                    time.sleep(0.2)

    finally:
        driver.close()

    logging.info(f"Summary: UPDATED={updated}, UNCHANGED={unchanged}, TOTAL={updated + unchanged}")

if __name__ == "__main__":
    main()
