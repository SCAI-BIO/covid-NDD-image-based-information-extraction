# mesh_category_extraction.py

"""
MeSH Category Term Extraction Script

Authors: Elizaveta Popova, Negin Babaiha  
Institution: University of Bonn, Fraunhofer SCAI  
Date: 30/07/2025

Description:
    Parses a local MeSH descriptor XML file and extracts terms grouped by high-level 
    biomedical categories (relevant to COVID-19 and neurodegeneration) using keyword matching.

    Categories are built from predefined seed keywords. Matched MeSH terms and synonyms 
    are exported to JSON for downstream usage in triple annotation or ontology-based filtering.

Usage:
    python src/mesh_category_extraction.py --mesh_xml path/to/desc2025.xml --output path/to/output.json

Notes:
    • The file `desc2025.xml` must be downloaded locally by the user from the official MeSH website.
"""

import xml.etree.ElementTree as ET
import json
import re
import argparse
from collections import defaultdict

# === Define Category Seed Keywords ===
category_keywords = {
    "Viral Entry and Neuroinvasion": [
        "neuroinvasion", "receptor", "ACE2", "blood-brain barrier", "BBB", "virus entry", "olfactory",
        "retrograde transport", "endocytosis", "direct invasion", "cranial nerve", "neural pathway",
        "transcribrial", "neurotropic", "trans-synaptic", "neuronal route", "olfactory nerve",
        "hematogenous", "choroid plexus", "neuronal transmission", "entry into CNS"
    ],
    "Immune and Inflammatory Response": [
        "immune", "cytokine", "inflammation", "interferon", "TNF", "IL-6", "IL6", "cytokine storm",
        "immune response", "inflammatory mediators", "macrophage", "microglia", "neutrophil",
        "lymphocyte", "innate immunity", "immune dysregulation", "chemokine", "T cell", "NLRP3",
        "antibody", "immune activation", "immune imbalance", "immune-mediated", "complement"
    ],
    "Neurodegenerative Mechanisms": [
        "neurodegeneration", "protein aggregation", "apoptosis", "cell death", "synaptic loss",
        "neurotoxicity", "oxidative stress", "mitochondrial dysfunction", "tau", "amyloid",
        "α-synuclein", "prion", "demyelination", "neuron loss", "misfolded proteins",
        "chronic neuronal damage", "neurodegenerative", "neuroinflammation"
    ],
    "Vascular Effects": [
        "stroke", "thrombosis", "vascular", "ischemia", "coagulation", "blood clot", "microthrombi",
        "endothelial", "vasculitis", "hemorrhage", "blood vessel", "vascular damage", "capillary",
        "clotting", "hypoperfusion", "angiopathy", "vasculopathy"
    ],
    "Psychological and Neurological Symptoms": [
        "cognitive", "memory", "fatigue", "depression", "anxiety", "brain fog", "psychiatric",
        "mood", "confusion", "neuropsychiatric", "emotional", "behavioral", "neurocognitive",
        "insomnia", "psychosocial", "attention", "motivation", "executive function", "suicidality"
    ],
    "Systemic Cross-Organ Effects": [
        "lungs", "liver", "kidney", "systemic", "multi-organ", "gastrointestinal", "heart",
        "cardiovascular", "endocrine", "renal", "pancreas", "organ failure", "liver damage",
        "pulmonary", "myocardial", "respiratory", "hypoxia", "oxygen deprivation", "fibrosis"
    ]
}

def extract_category_terms(mesh_xml_path, category_keywords):
    """
    Parse MeSH XML and extract descriptors/synonyms matching seed keywords per category.

    Args:
        mesh_xml_path (str): Path to MeSH XML file
        category_keywords (dict): Category names and their keyword seeds

    Returns:
        dict: Mapping of category → list of matching MeSH terms
    """
    tree = ET.parse(mesh_xml_path)
    root = tree.getroot()
    category_terms = defaultdict(set)

    for descriptor in root.findall('DescriptorRecord'):
        descriptor_name_el = descriptor.find('DescriptorName/String')
        if descriptor_name_el is None:
            continue

        descriptor_name = descriptor_name_el.text
        term_elements = descriptor.findall('ConceptList/Concept/TermList/Term/String')
        synonyms = [term_el.text for term_el in term_elements if term_el is not None]
        all_text = f"{descriptor_name} " + ' '.join(synonyms)

        for category, keywords in category_keywords.items():
            if any(keyword.lower() in all_text.lower() for keyword in keywords):
                category_terms[category].update([descriptor_name] + synonyms)

    # Convert sets to sorted lists
    for category in category_terms:
        category_terms[category] = sorted(list(category_terms[category]))

    return category_terms

def preview_sample(category_terms, category="Immune and Inflammatory Response", n=25):
    """Print a preview of extracted terms from one category."""
    print(f"\n=== Preview: {category} ===")
    for term in category_terms[category][:n]:
        print("-", term)

def save_to_json(category_terms, output_path):
    """Save category-term dictionary to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(category_terms, f, indent=2)
    print(f"\nExtraction complete! Terms saved to: {output_path}")

# === Command-line interface ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Extract MeSH terms grouped by biomedical categories relevant to COVID-19 and neurodegeneration.
        NOTE: You must first download the official MeSH descriptor file locally (e.g. desc2025.xml).
        """
    )
    parser.add_argument("--mesh_xml", required=True, help="Path to local MeSH descriptor XML file (e.g., desc2025.xml)")
    parser.add_argument("--output", required=True, help="Path to output JSON file for category terms")

    args = parser.parse_args()

    print("Parsing MeSH XML and extracting category terms...")
    category_terms = extract_category_terms(args.mesh_xml, category_keywords)

    preview_sample(category_terms)
    save_to_json(category_terms, args.output)

# python src/mesh_category_extraction.py --mesh_xml data/MeSh_data/desc2025.xml --output data/MeSh_data/mesh_category_terms.json
