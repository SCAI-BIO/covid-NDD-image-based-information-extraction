# neo4j_upload_fulltext.py

"""
Neo4j Upload of Biomedical Triples with Ontology-Based Labeling
Authors: Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 07/08/2025

Description:
    Loads semantic triples extracted from biomedical text into a Neo4j graph database.
    Triples are processed to determine ontology-based labels and namespaces using external
    APIs (e.g., HGNC, GO, MESH, HPO). The script assigns semantic types (e.g., Gene, Disease,
    Phenotype) to each entity, cleans labels for Neo4j compatibility, and merges nodes and
    relationships accordingly.

    Supports robust input handling with CSV delimiter sniffing, optional metadata (e.g., PMID,
    Title, Evidence), and customized metadata fields like Cell and Anatomy. Triple matching
    and node creation follow a consistent and interpretable schema.

Usage:
    # Run with a custom input CSV
    python src/neo4j_upload_fulltext.py \
    --input data/gold_standard_comparison/Triples_Full_Text_GPT_for_comp_cleaned.csv \
    --password YOUR_Neo4j_PASSWORD

Note:
    After this script run gpt4o-correct-neo4j-labels-nodes.py to correct labels!
"""

# After this RUN GPT-CORRECT .. script to correct labels!!
from __future__ import annotations
from operator import delitem
import sys
from unicodedata import name
import pandas as pd
import requests
from rapidfuzz import fuzz, process
from neo4j import GraphDatabase
import os
import re
from difflib import get_close_matches
import urllib
import logging
from pathlib import Path
import argparse
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import csv
import re
print("starting")
def sanitize_label(label):
    """
    Normalize and sanitize a label to ensure compatibility with Neo4j node naming rules (only if necessary).
    Replaces spaces, hyphens, and dots with underscores.
    
    Args:
        label (str): Raw entity label or predicate.
    
    Returns:
        str: Cleaned and Neo4j-safe label.
    """
    label_str = str(label).strip()
    # Check if the label contains only allowed characters (letters, digits, and underscores)
    if re.fullmatch(r'\w+', label_str):
        return label_str
    # Otherwise, replace problematic characters
    return label_str.replace("-", "_").replace(" ", "_").replace(".", "_")


def clean_entity_text(entity: str) -> str:
    """
    Lowercase and remove special characters from an entity name to standardize it.
    
    Args:
        entity (str): Raw entity string.
    
    Returns:
        str: Cleaned and normalized entity string.
    """
    entity = entity.strip().lower()
    # Replace common separators with space or underscore
    entity = entity.replace("-", " ").replace("_", " ").replace("+", " plus ")
    # Remove or normalize any remaining special characters
    entity = re.sub(r"[^\w\s]", "", entity)  # keeps letters, numbers, spaces
    entity = re.sub(r"\s+", " ", entity)     # collapse multiple spaces
    return entity.strip()

def determine_entity_label(entity):
    """
    Determine the semantic type (e.g., Gene, Disease, Protein, etc.) and ontology source of a given entity.
    Uses a combination of heuristics, keyword detection, and external ontology APIs (e.g., HGNC, MESH, GO).
    
    Args:
        entity (str): A single biological entity name.
    
    Returns:
        tuple: (label_type, namespace, ontology_id)
            - label_type (str): Assigned semantic category (e.g., "Gene").
            - namespace (str): Ontology source (e.g., "HGNC", "GO").
            - ontology_id (str or None): Unique ontology identifier, if found.
    """
    entity = entity.strip().upper()
    
    # Add SNP pattern detection
    snp_patterns = [
        r'^RS\d+$',  # Standard RS ID format
        r'^[Rr][Ss]\d+$'  # Case insensitive RS ID
    ]
    
    # Check for SNP pattern first
    if any(re.match(pattern, entity) for pattern in snp_patterns):
        return False, "SNP"  # Not a gene, but a SNP

    # Expanded disease indicators
    disease_indicators = {
        "ALZHEIMER", "SYNDROME", "DISEASE", "DISORDER", "IMMUNODEFICIENCY", 
        "ITIS", "OSIS", "PATHY", "CANCER", "TUMOR", "DEFICIENCY",
        "COVID", "SARS", "SEQUELA", "LONG COVID", "POST-COVID",
        "CONDITION", "INFECTION", "COMPLICATION","CANCER", "TUMOR", "PATHY"
    }
    
    # Common phenotype terms
    phenotype_indicators = {
        "PHENOTYPE", "ABNORMAL", "MORPHOLOGY", "APPEARANCE", "TRAIT", 
        "PRESENTATION", "MANIFESTATION", "CLINICAL", "SYMPTOMS", "FEATURES",
        "LOSS", "MEGALY", "TROPHY", "PLASIA", "GRADE",
        "DISABILITY", "IMPAIRMENT", "DEFICIT", "DYSFUNCTION",
        "INHERITANCE", "DOMINANT", "RECESSIVE", "LINKED",
        "SEIZURE", "ATAXIA", "PALSY", "DYSTROPHY", "WEAKNESS",
        "RETARDATION", "DEGENERATION", "DEFICIT",
        "SEQUELA", "COMPLICATION", "AFTER", "POST"
    }
    
    # Process indicators
    process_indicators = {
        "PATHWAY", "SIGNALING", "INFLAMMATION", "RESPONSE", 
        "REGULATION", "METABOLISM", "SYNTHESIS", "ACTIVITY"
    }
    
    # Protein/molecule indicators
    protein_indicators = {
        "CYTOKINE", "CYTOKINES", "INTERLEUKIN", "CHEMOKINE",
        "FACTOR", "PROTEIN", "RECEPTOR", "HORMONE", "ENZYME"
    }
    
    # Known gene patterns
    gene_indicators = {"APOE4", "BRCA1", "TP53"}

    # Add more comprehensive gene pattern matching
    gene_patterns = [
        r"^LOC\d+$",
        r"^[A-Z0-9]+-AS\d+$",
        r"^LINC\d+$",
        r"^MIR\d+$",
        r"^SNOR\d+$",
        r"^(ATP|ABC|SLC|COL|IL|TNF|TGF|IGF|FGF|EGF|HOX|SOX|FOX)\d+[A-Z]?$",
        r"^[A-Z]{1,6}\d[A-Z0-9]*$",  # e.g., TP53, BRCA1
        r"^[A-Z]{2,6}\d[A-Z0-9]*-AS\d*$",  # antisense RNA with number
        r"^[A-Z]{2,4}\d{1,2}[A-Z]{0,1}\d{0,1}$",  # e.g., AKT1, MAPK14
        r"^CDK\d+[A-Z]?$",  # CDK family
        r"^CCN[A-Z]\d*$",   # CCN family
        r"^CD\d+[A-Z]?$"    # CD markers
    ]
    
    # First, check against explicit gene patterns
    if any(re.match(pattern, entity) for pattern in gene_patterns):
        return True, "Gene"
    
    # Then check for COVID-related terms
    if any(covid_term in entity for covid_term in ["COVID", "SARS-COV", "SEQUELA OF COVID"]):
        return False, "Disease"
    
    # Check for disease terms
    if any(term in entity for term in disease_indicators):
        return False, "Disease"
    
    # Check for phenotype indicators
    if any(term in entity for term in phenotype_indicators):
        return False, "Phenotype"
    
    # Check for protein/molecule indicators
    if any(term in entity for term in protein_indicators):
        return False, "Protein"
    
    # Check for process indicators
    if any(term in entity for term in process_indicators):
        return False, "Biological_Process"
    
    # Fallback gene check: only check for specific gene indicators (without the short-length assumption)
    if entity.startswith("APOE") or any(gene in entity for gene in gene_indicators):
        return True, "Gene"
    
    # Default case: not obviously any of the above
    return False, None


def determine_type_from_content(result):
    """Helper function to determine type based on content analysis"""
    label = result['label'].lower()
    description = result.get('description', '').lower()
    types = result.get('type', [])
    
    # Check for inheritance patterns
    if any(word in label or word in description for word in ['inheritance', 'hereditary', 'genetic pattern']):
        return "Inheritance"
    
    # Check for phenotypes
    if any(word in label or word in description for word in ['phenotype', 'symptom', 'clinical feature']):
        return "Phenotype"
    
    # Check for diseases
    if any(word in label or word in description for word in ['disease', 'syndrome', 'disorder']):
        return "Disease"
    
    # Default to trait if nothing else matches
    return "Trait"

def query_hpo_fuzzy(entity, threshold=0.6):
    """
    Enhanced query for terms with better ontology-based classification.
    Returns (label_type, namespace, ontology_id) tuple.
    """
    original_term = entity.strip()
    search_term = original_term.lower()
    
    print(f"Searching for term: {original_term}")
    
    # Define ontology-to-type mappings
    ontology_mappings = {
        'HP': ('Phenotype', 'HP'),      # Human Phenotype Ontology
        'SYMP': ('Phenotype', 'SYMP'),  # Symptom Ontology
        'MP': ('Phenotype', 'MP'),      # Mammalian Phenotype
        'OMIM': ('Disease', 'OMIM'),    # Online Mendelian Inheritance in Man
        'MONDO': ('Disease', 'MONDO'),  # Mondo Disease Ontology
        'DOID': ('Disease', 'DOID'),    # Disease Ontology
        'NCIT': ('Disease', 'NCIT'),    # National Cancer Institute Thesaurus
        'GENO': ('Inheritance', 'GENO'), # Genotype Ontology
        'EFO': ('Experimental Factor', 'EFO')       # Experimental Factor Ontology
    }

    # Try OLS general search first
    url = f"https://www.ebi.ac.uk/ols/api/search?q={search_term}&local=true"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            results = response.json()
            docs = results.get("response", {}).get("docs", [])
            
            if docs:
                # Process and score matches
                scored_matches = []
                for doc in docs:
                    ontology = doc.get("ontology_prefix", "")
                    label = doc.get("label", "").lower()
                    score = fuzz.ratio(search_term, label)
                    
                    # Get all possible IDs
                    obo_id = doc.get("obo_id")
                    iri = doc.get("iri", "")
                    short_form = doc.get("short_form")
                    
                    # Determine best ID
                    ontology_id = None
                    if obo_id:
                        ontology_id = obo_id
                    elif iri and "obo/" in iri:
                        # Extract ID from IRI
                        ontology_id = iri.split("/")[-1].replace("_", ":")
                    elif short_form and ":" in short_form:
                        ontology_id = short_form
                    elif iri:
                        # Create ID from IRI if nothing else available
                        last_part = iri.split("/")[-1]
                        ontology_id = f"{ontology}:{last_part}"

                    print(f"Found match: {label} ({ontology}) - Score: {score} - ID: {ontology_id}")
                    
                    if score >= threshold * 100 and ontology_id:
                        scored_matches.append({
                            'score': score,
                            'ontology': ontology,
                            'label': label,
                            'id': ontology_id,
                            'description': doc.get("description", [""])[0],
                            'semantic_types': doc.get("semantic_types", [])
                        })
                
                if scored_matches:
                    # Sort by score
                    scored_matches.sort(key=lambda x: x['score'], reverse=True)
                    best_match = scored_matches[0]
                    
                    # Use ontology mapping if available
                    if best_match['ontology'] in ontology_mappings:
                        label_type, namespace = ontology_mappings[best_match['ontology']]
                        return label_type, namespace, best_match['id']
                    
                    # Fallback classification based on semantic types and description
                    semantic_types = best_match.get('semantic_types', [])
                    description = best_match.get('description', '').lower()
                    
                    if any(t in semantic_types for t in ['Sign or Symptom']):
                        return "Symptom", best_match['ontology'], best_match['id']
                    elif any(t in semantic_types for t in ['Disease or Syndrome']):
                        return "Disease", best_match['ontology'], best_match['id']
                    
                    # Default to the ontology's primary domain
                    return "Phenotype", best_match['ontology'], best_match['id']
    
    except Exception as e:
        print(f"Error in OLS search: {str(e)}")
    
    return None, None, None
import requests
import urllib.parse

import requests
import urllib.parse
import xml.etree.ElementTree as ET


def query_chembl_compound(query):
    """
    Use the dedicated CHEMBL molecule search endpoint to find the best matching molecule.
    
    Endpoint used:
        GET /chembl/api/data/molecule/search?q=:query
    
    The function parses the XML response, scores each candidate using fuzzy matching
    against the input query (case-insensitive), and returns the CHEMBL ID of the best match
    if its fuzzy score exceeds the given threshold.
    
    Parameters:
        query (str): The search term (e.g., chemical name).
        score_threshold (int): Minimum fuzzy match score required.
    
    Returns:
        str or None: The CHEMBL ID of the best candidate or None if no candidate meets the threshold.
    """
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={encoded_query}"
    score_threshold = 0.90
    try:
        response = requests.get(url)
    except Exception as e:
        print(f"Error querying CHEMBL for '{query}': {str(e)}")
        return None
    
    if response.status_code == 200:
        xml_text = response.text
        # Optionally print the raw response (truncated)
        print(f"Raw CHEMBL API response for '{query}': {xml_text[:500]}...\n")
        
        root = ET.fromstring(xml_text)
        candidates = []
        # Loop through each <molecule> element in the response
        for molecule in root.findall(".//molecule"):
            chembl_id_el = molecule.find("molecule_chembl_id")
            pref_name_el = molecule.find("pref_name")
            chembl_id = chembl_id_el.text if chembl_id_el is not None else None
            pref_name = pref_name_el.text if pref_name_el is not None else ""
            
            # Calculate fuzzy matching score using RapidFuzz
            score = fuzz.ratio(query.lower(), pref_name.lower()) if pref_name else 0
            candidates.append((chembl_id, pref_name, score))
            print(f"Candidate: {chembl_id} ({pref_name}) with score {score}")
        
        if candidates:
            best_candidate = max(candidates, key=lambda x: x[2])
            print(f"\nCHEMBL best candidate for '{query}': {best_candidate[0]} ({best_candidate[1]}) with score {best_candidate[2]}")
            if best_candidate[2] >= score_threshold:
                return "Chemical", "Chembl", best_candidate[0]
            else:
                print(f"Best candidate score ({best_candidate[2]}) is below threshold ({score_threshold}).")
                return None
        else:
            print(f"No CHEMBL candidates found for '{query}'.")
            return None
    else:
        print(f"CHEMBL API request failed for '{query}' with status {response.status_code}.")
        return None

def query_disease_ontology_fuzzy(entity, threshold=0.55):
    """
    Query Disease Ontology (DO) for approximate string matching.
    Returns (label_type, namespace, ontology_id) tuple.
    """
    
    encoded_term = urllib.parse.quote(entity)

    # Construct the API URL
    url = f"https://www.ebi.ac.uk/ols/api/search?q={encoded_term}&ontology=doid"
    #url = f"https://www.ebi.ac.uk/ols/api/ontologies/doid/terms?q={entity_cleaned.lower()}"
    response = requests.get(url)
    #print(response.text)

    if response.status_code == 200:
        results = response.json()
        #print(results)
        terms = results.get("response", {}).get("docs", [])
        
        if terms:
            for term in terms:
                #print(term)
                # Decode the label
                raw_label = term.get("label", "")
                decoded_label = bytes(raw_label, "utf-8").decode("unicode_escape")  # Decoding Unicode
                label_lower = decoded_label.lower()

                # Exact match or fuzzy match
                if entity.lower() in label_lower or \
                   get_close_matches(entity.lower(), [label_lower], n=1, cutoff=threshold):
                    ontology_id = term.get("obo_id") or term.get("short_form")
                    #print(ontology_id)
                    # Save the decoded label instead of raw Unicode
                    return "Disease", "DO", ontology_id#, decoded_label
    return None, None, None


def query_hgnc_fuzzy(entity):
    """
    Enhanced query function with RefSeq and HGNC support.
    Returns (label_type, namespace, id) tuple.
    """
    entity = entity.strip().upper()
    headers = {"Accept": "application/json"}
    
    # Handle LOC identifiers with RefSeq check
    if entity.startswith('LOC'):
        try:
            # Use NCBI E-utilities to verify the LOC identifier
            esearch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&term={entity}&format=json"
            response = requests.get(esearch_url)
            
            if response.status_code == 200:
                results = response.json()
                if results.get('esearchresult', {}).get('count', '0') != '0':
                    # Get detailed info about the gene
                    gene_id = results['esearchresult']['idlist'][0]
                    esummary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&id={gene_id}&format=json"
                    detail_response = requests.get(esummary_url)
                    
                    if detail_response.status_code == 200:
                        details = detail_response.json()
                        gene_info = details['result'][gene_id]
                        # Verify it's a real gene entry
                        if gene_info.get('status') == 'live':
                            return "Gene", "RefSeq", f"LOC_{gene_id}"
        except Exception as e:
            print(f"Error in RefSeq query: {str(e)}")
            # Continue to HGNC check if RefSeq check fails
    
    # Try exact symbol match in HGNC
    try:
        url = f"https://rest.genenames.org/search/symbol/{entity}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            results = response.json()
            if results.get("response", {}).get("numFound", 0) > 0:
                doc = results["response"]["docs"][0]
                hgnc_id = doc.get("hgnc_id")
                
                # Check if it has RefSeq ID
                refseq_id = doc.get("refseq_accession")
                if refseq_id:
                    return "Gene", "RefSeq", refseq_id[0]  # Return first RefSeq ID if available
                return "Gene", "HGNC", hgnc_id
    except Exception as e:
        print(f"Error in HGNC symbol query: {str(e)}")
    
    # Try alias search in HGNC
    try:
        url = f"https://rest.genenames.org/search/alias_symbol/{entity}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            results = response.json()
            if results.get("response", {}).get("numFound", 0) > 0:
                doc = results["response"]["docs"][0]
                hgnc_id = doc.get("hgnc_id")
                
                # Check if it has RefSeq ID
                refseq_id = doc.get("refseq_accession")
                if refseq_id:
                    return "Gene", "RefSeq", refseq_id[0]
                return "Gene", "HGNC", hgnc_id
    except Exception as e:
        print(f"Error in HGNC alias query: {str(e)}")
    
    # If the entity matches LOC pattern but wasn't found in RefSeq
    if entity.startswith('LOC'):
        return "Gene", "RefSeq", entity  # Still return as RefSeq gene but without verification
    
    return None, None, None

def query_gene_ontology_fuzzy(entity, threshold=0.55):
    """
    Query Gene Ontology (GO) for approximate string matching.
    Returns (label_type, namespace, go_id) tuple.
    """
    url = f"https://www.ebi.ac.uk/ols/api/ontologies/go/terms?q={entity.lower()}"
    response = requests.get(url)

    if response.status_code == 200:
        results = response.json()
        terms = results.get("_embedded", {}).get("terms", [])

        if terms:
            for term in terms:
                label = term.get("label", "").lower()
                if entity.lower() in label or \
                   get_close_matches(entity.lower(), [label], n=1, cutoff=threshold):
                    ontology_id = term.get("obo_id") or term.get("short_form")
                    return "Biological_Process", "GO", ontology_id

    return None, None, None
def query_mammalian_phenotype_fuzzy(entity, threshold=0.55):
    """
    Query Mammalian Phenotype Ontology (MP) for phenotype terms.
    Returns (label_type, namespace, mp_id) tuple.
    """
    url = f"https://www.ebi.ac.uk/ols/api/ontologies/mp/terms?q={entity.lower()}"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json()
        terms = results.get("_embedded", {}).get("terms", [])
        
        if terms:
            for term in terms:
                label = term.get("label", "").lower()
                if entity.lower() in label or \
                   get_close_matches(entity.lower(), [label], n=1, cutoff=threshold):
                    ontology_id = term.get("obo_id") or term.get("short_form")
                    return "Phenotype", "MP", ontology_id
    
    return None, None, None

import urllib.parse
import requests

def query_kegg_pathway(entity):
    """
    Query KEGG for pathway information.
    Uses KEGG REST API to find pathways matching the query.
    Returns a tuple (label, namespace, pathway_id) if found.
    """
    encoded = urllib.parse.quote(entity)
    url = f"http://rest.kegg.jp/find/pathway/{encoded}"
    response = requests.get(url)
    if response.status_code == 200:
        lines = response.text.strip().splitlines()
        if lines:
            # The KEGG API returns lines like: 
            # pathway:map00010  Glycolysis / Gluconeogenesis - Homo sapiens (human)
            first_line = lines[0]
            pathway_id = first_line.split("\t")[0].split(":")[1]  # e.g., "map00010"
            return "Pathway", "KEGG", pathway_id
    return None, None, None

def query_reactome_pathway(entity):
    """
    Query Reactome for pathway information.
    Uses the Reactome Content Service search endpoint for pathways.
    Returns a tuple (label, namespace, reactome_id) if found.
    """
    encoded = urllib.parse.quote(entity)
    url = f"https://reactome.org/ContentService/search/query?query={encoded}&species=Homo+sapiens&types=Pathway"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        if results:
            best = results[0]
            reactome_id = best.get("stId")
            return "Pathway", "Reactome", reactome_id
    return None, None, None

def query_wikipathways(entity):
    """
    Query WikiPathways for pathway information.
    Uses the WikiPathways webservice to find pathways matching the query.
    Returns a tuple (label, namespace, wp_id) if found.
    """
    encoded = urllib.parse.quote(entity)
    url = f"https://webservice.wikipathways.org/findPathwaysByText?query={encoded}&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        pathways = data.get("pathways", [])
        if pathways:
            best = pathways[0]
            wp_id = best.get("id")
            return "Pathway", "WikiPathways", wp_id
    return None, None, None

def query_pathway_databases(entity):
    """
    Try querying KEGG, Reactome, and WikiPathways in order.
    Returns the first match found.
    """
    for query_fn in [query_kegg_pathway, query_reactome_pathway, query_wikipathways]:
        label, namespace, pathway_id = query_fn(entity)
        if label and pathway_id:
            return label, namespace, pathway_id
    return None, None, None

from rapidfuzz import fuzz

from rapidfuzz import fuzz

def query_mesh_nih(entity, fuzzy_threshold=60):
    """
    Query NIH MESH API for terms and classify based on MeSH tree numbers,
    qualifiers, custom keyword overrides, and fuzzy matching.
    Returns (label_type, namespace, mesh_id) tuple.
    """
    import requests
    import urllib.parse

    search_term = entity.strip().replace(" ", "+").lower()
    base_url = "https://id.nlm.nih.gov/mesh"
    
    try:
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        
        # Try an exact match first
        exact_url = f"{base_url}/lookup/descriptor?label={search_term}&match=exact&limit=5"
        response = requests.get(exact_url, headers=headers)
        candidates = []
        
        if response.status_code == 200:
            data = response.json()
            if data:
                # Use fuzzy matching on the candidates from the exact query
                for candidate in data:
                    candidate_label = candidate.get('label', '')
                    score = fuzz.ratio(entity.lower(), candidate_label.lower())
                    candidates.append((candidate, score))
        
        # If no candidates from exact match, try a partial ("contains") query
        if not candidates:
            partial_url = f"{base_url}/lookup/descriptor?label={search_term}&match=contains&limit=5"
            response = requests.get(partial_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                for candidate in data:
                    candidate_label = candidate.get('label', '')
                    score = fuzz.ratio(entity.lower(), candidate_label.lower())
                    candidates.append((candidate, score))
        
        if candidates:
            # Select the candidate with the highest fuzzy score
            best_candidate, best_score = max(candidates, key=lambda x: x[1])
            if best_score < fuzzy_threshold:
                print(f"Best fuzzy match score {best_score} is below threshold {fuzzy_threshold}.")
                return None, None, None
            
            # Process the best candidate
            mesh_id = best_candidate.get('resource', '').split('/')[-1]
            detail_url = f"{base_url}/lookup/details?descriptor={mesh_id}"
            detail_response = requests.get(detail_url, headers=headers)
            
            if detail_response.status_code == 200:
                details = detail_response.json()
                tree_numbers = details.get('treeNumbers', [])
                qualifiers = details.get('qualifiers', [])
                qualifier_labels = [q.get("label", "").lower() for q in qualifiers]
                term_lower = entity.lower()
                print(qualifier_labels)
                #print(details)
                # CUSTOM OVERRIDE for COVID-related terms
                if any(x in term_lower for x in ["sars-cov", "covid", "coronavirus"]):
                    return "Disease", "MESH", f"MESH:{mesh_id}"
                
                # --- Tree Number Based Classification ---
                label_type_tree = None
                if tree_numbers:
                    deepest_tree = max(tree_numbers, key=lambda tn: tn.count('.'))
                    mesh_category_mapping = {
                        'A': "Anatomical",
                        'B': "Organism",
                        'C': "Disease",
                        'D': "Chemical",
                        'E': "Analytical",
                        'F': "Psychiatry",
                        'G': "Biological_Process",
                        'H': "Disciplines",
                        'I': "Anthropology",
                        'J': "Technology",
                        'K': "Humanities",
                        'L': "Information",
                        'M': "Named_Groups",
                        'N': "Health_Care",
                        'V': "Publication_Characteristics",
                        'Z': "Geographical"
                    }
                    label_type_tree = mesh_category_mapping.get(deepest_tree[0], "Unknown")
                    # Refine tree mapping if term implies disease
                    if label_type_tree == "Chemical" and any(word in term_lower for word in ["disease", "syndrome", "disorder", "infection", "inflammatory"]):
                        label_type_tree = "Disease"
                
                # --- Qualifier-Based Classification ---
                # Build a mapping of qualifier keywords to classes
                qualifier_mapping = {
                    "chemistry": "Chemical",
                    "pharmacology": "Chemical",
                    "drug effects": "Chemical",
                    "enzymology": "Chemical",
                    "pathology": "Disease",
                    "disease": "Disease",
                    "disorder": "Disease",
                    "syndrome": "Disease",
                    "epidemiology": "Disease",
                    "physiology": "Biological_Process",
                    "metabolism": "Biological_Process",
                    "pathway": "Biological_Process",
                    "process": "Biological_Process",
                    "signaling": "Biological_Process",
                }
                mapping_count = {}
                for q in qualifier_labels:
                    for key, cat in qualifier_mapping.items():
                        if key in q:
                            mapping_count[cat] = mapping_count.get(cat, 0) + 1
                qualifier_category = None
                if mapping_count:
                    qualifier_category = max(mapping_count.items(), key=lambda x: x[1])[0]
                
                # --- Decide Final Classification ---
                if qualifier_category:
                    # If tree mapping exists, and qualifier mapping suggests Disease (with disease keywords present), override to Disease
                    if label_type_tree:
                        if qualifier_category == "Disease" and any(word in term_lower for word in ["disease", "syndrome", "disorder", "infection", "inflammatory"]):
                            return "Disease", "MESH", f"MESH:{mesh_id}"
                        # If both approaches agree, return that classification
                        if qualifier_category == label_type_tree:
                            return label_type_tree, "MESH", f"MESH:{mesh_id}"
                        # Otherwise, prioritize the qualifier-based result
                        return qualifier_category, "MESH", f"MESH:{mesh_id}"
                    else:
                        return qualifier_category, "MESH", f"MESH:{mesh_id}"
                
                # Fallback: if no qualifier mapping is available, use the tree-based classification if present
                if label_type_tree:
                    return label_type_tree, "MESH", f"MESH:{mesh_id}"
                
                # Final fallback if nothing else matches
                return "Disease", "MESH", f"MESH:{mesh_id}"
    
    except Exception as e:
        print(f"Error querying NIH MESH API: {str(e)}")
    
    return None, None, None



def query_protein_ontology_fuzzy(entity, threshold=0.55):
    """
    Query Protein Ontology (PRO) for protein terms.
    Returns (label_type, namespace, pro_id) tuple.
    """
    url = f"https://www.ebi.ac.uk/ols/api/ontologies/pr/terms?q={entity.lower()}"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json()
        terms = results.get("_embedded", {}).get("terms", [])
        
        if terms:
            for term in terms:
                label = term.get("label", "").lower()
                synonyms = term.get("synonyms", [])
                
                # Check label and synonyms for matches
                if (fuzz.ratio(entity.lower(), label) >= threshold * 100 or 
                    any(fuzz.ratio(entity.lower(), syn.lower()) >= threshold * 100 for syn in synonyms)):
                    ontology_id = term.get("obo_id") or term.get("short_form")
                    return "Protein", "PR", ontology_id
    
    # Try UniProt as fallback
    url = f"https://rest.uniprot.org/uniprotkb/search?query={entity.lower()}&format=json"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json()
        if results.get("results"):
            uniprot_id = results["results"][0]["primaryAccession"]
            return "Protein", "UniProt", uniprot_id
            
    return None, None, None

def query_ols_fuzzy(entity, threshold=0.5):
    """
    Enhanced OLS query function with better handling of compound terms and reduced threshold.
    Returns (label_type, namespace, ontology_id) tuple.
    """
    # Clean the search term
    search_term = entity.lower().strip()
    
    # First try Disease Ontology
    label, namespace, ontology_id = query_disease_ontology_fuzzy(entity, threshold)
    if label:
        return label, namespace, ontology_id

    # Then try Gene Ontology
    label, namespace, ontology_id = query_gene_ontology_fuzzy(entity, threshold)
    if label:
        return label, namespace, ontology_id

    # Try direct OLS query
    url = f"https://www.ebi.ac.uk/ols/api/search?q={search_term}&local=true"
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            results = response.json()
            docs = results.get("response", {}).get("docs", [])
            
            if docs:
                # Sort by score and exact matches first
                scored_docs = []
                for doc in docs:
                    label = doc.get("label", "").lower()
                    ontology = doc.get("ontology_prefix", "")
                    
                    # Calculate match score
                    exact_match = search_term == label
                    contains_match = search_term in label or label in search_term
                    fuzzy_score = fuzz.ratio(search_term, label)
                    
                    # Boost score for relevant ontologies
                    ontology_boost = 1.2 if ontology in {"GO", "MESH", "DOID", "PR", "CHEBI"} else 1.0
                    
                    final_score = (fuzzy_score * ontology_boost) + (50 if exact_match else 0) + (25 if contains_match else 0)
                    
                    scored_docs.append((final_score, doc))
                
                # Sort by score
                scored_docs.sort(reverse=True, key=lambda x: x[0])
                
                # Process the best match
                if scored_docs:
                    best_score, best_doc = scored_docs[0]
                    
                    if best_score >= threshold * 100:
                        ontology = best_doc.get("ontology_prefix", "")
                        ontology_id = best_doc.get("obo_id") or best_doc.get("short_form")
                        
                        # Map ontology to label type
                        if ontology in {"DOID", "MESH", "MESHD"}:
                            # Check if it's inflammation-related
                            if "inflammation" in search_term or "itis" in search_term:
                                return "Biological_Process", ontology, ontology_id
                            return "Disease", ontology, ontology_id
                        elif ontology == "GO":
                            return "Biological_Process", ontology, ontology_id
                        elif ontology == "PR":
                            return "Protein", ontology, ontology_id
                        elif ontology == "CHEBI":
                            return "Chemical", ontology, ontology_id
                        
                        # Additional checks for biological processes
                        label = best_doc.get("label", "").lower()
                        if any(term in label for term in ["process", "regulation", "pathway", "inflammation"]):
                            return "Biological_Process", ontology, ontology_id
    
    except Exception as e:
        print(f"Error querying OLS: {str(e)}")
    
    # Process-specific fallback for inflammation terms
    if "inflammation" in search_term or "itis" in search_term:
        return "Biological_Process", "GO", None
    
    return None, None, None

def determine_entity_label(entity):
    """
    Determine the entity label and obtain an ontology ID using external queries.
    
    This version prioritizes gene-specific queries if the entity looks gene-like,
    then falls back to broad ontology queries (OLS, MESH), and finally uses
    specialized queries for phenotypes, pathways, proteins, chemicals, etc.
    
    Returns:
        tuple: (label_type, namespace, ontology_id)
    """
    # Normalize the input
    entity_lower = clean_entity_text(entity)
    entity_gene = entity.strip().upper()
    # entity_lower = entity_clean.lower()
    #check basic rules:

    cell_keywords = {
    "cell", "cells", "cyte", "cytes", "blast", "blasts",
    "fibroblast", "fibroblasts",
    "macrophage", "macrophages",
    "neuron", "neurons", "neuroblast", "neuroblasts",
    "astrocyte", "astrocytes",
    "oligodendrocyte", "oligodendrocytes",
    "lymphocyte", "lymphocytes",
    "monocyte", "monocytes",
    "t cell", "t-cell", "t lymphocyte", "b cell", "b-cell", "b lymphocyte",
    "nk cell", "nk-cell", "natural killer cell",
    "dendritic cell", "dendritic cells",
    "stem cell", "stem cells", "pluripotent", "multipotent", "totipotent",
    "progenitor", "hematopoietic", "epithelial", "mesenchymal",
    "microglia", "microglial",
    "schwann cell", "schwann cells",
    "retinal ganglion", "ganglion cell", "rods", "cones",
    "osteoblast", "osteoblasts", "osteocyte", "osteocytes",
    "chondrocyte", "chondrocytes",
    "hepatocyte", "hepatocytes",
    "myocyte", "myocytes",
    "keratinocyte", "keratinocytes",
    "pancreatic beta cell", "beta cell", "alpha cell", "islet cell",
    "germ cell", "germ cells", "oocyte", "oocytes", "spermatocyte", "spermatogonia",
    "erythrocyte", "erythrocytes", "red blood cell", "rbc", "white blood cell", "wbc",
    "thrombocyte", "megakaryocyte"}

    anatomy_keywords = {
    "brain", "neuron", "neurons", "olfactory", "bulb", "epithelium", "barrier",
    "plate", "tissue", "organ", "ventricle", "cortex", "hippocampus",
    "amygdala", "hypothalamus", "pituitary", "thalamus", "spinal", "nervous system",
    "axon", "dendrite", "glomerulus", "glomeruli", "ganglion", "blood-brain barrier",
    "bone", "skin", "muscle", "kidney", "lung", "heart", "spleen", "liver", "cranium",    "brain", "neuron", "bulb", "barrier", "epithelium", "plate", "ventricle",
    "cortex", "ganglion", "glomerulus", "spinal", "hippocampus", "thalamus",
    "amygdala", "pituitary", "hypothalamus", "bone", "organ", "tissue", "nerve",
    "cranium", "membrane", "tract", "artery", "capillary", "astrocyte", "oligodendrocyte"
}
    protein_keywords = {
        "protein", "receptor", "enzyme", "cytokine", "interleukin", "chemokine",
        "ligand", "kinase", "channel", "ion channel", "transporter", "antibody",
        "peptide", "hormone receptor", "growth factor", "transcription factor",
        "mrna binding protein", "rna binding protein", "binding protein"
    }
    biological_process_keywords = {
        # --- Viral Entry & Transport ---
        "viral entry", "receptor binding", "membrane fusion", "endocytosis",
        "retrograde transport", "anterograde transport", "axonal transport",
        "trans-synaptic spread", "dynein motors", "sars-cov-2 entry", "neuroinvasion",

        # --- Blood-Brain Barrier (BBB) ---
        "bbb breakdown", "vascular leakage", "endothelial dysfunction", "permeability increase",
        "microvascular damage", "damaged blood vessels",

        # --- Inflammation / Immune ---
        "cytokine storm", "inflammatory response", "inflammatory factor release",
        "neuroinflammation", "t cell activation", "b cell activation", "microglial activation",
        "mast cell activation", "il-6 signaling", "tnf-alpha signaling", "chemokine release",
        "ros production", "oxidative stress", "nitric oxide release",

        # --- Coagulation / Microthrombi ---
        "thrombus formation", "coagulation cascade", "platelet aggregation",
        "microclots", "microvascular thrombosis", "endotheliitis", "prothrombotic state",

        # --- Cellular Dysfunction ---
        "mitochondrial dysfunction", "impaired atp synthesis", "mptp opening",
        "calcium overload", "ca2+ overload", "cyt c release", "free radical formation",

        # --- Protein Aggregation ---
        "alpha-synuclein aggregation", "amyloid fibril formation", "tau hyperphosphorylation",
        "protein misfolding", "ubiquitin-proteasome dysfunction", "autophagy disruption",

        # --- Synaptic & Neuronal Damage ---
        "synaptic dysfunction", "demyelination", "neuronal loss", "neuronal apoptosis",
        "excitotoxicity", "neurodegeneration", "axonal degeneration", "trans-synaptic degeneration",
        "astroglial response", "astrogliosis", "neuronal inflammation",

        # --- Cell Death ---
        "apoptosis", "pyroptosis", "necrosis", "autophagic cell death", "programmed cell death",

        # --- Stress & Dysregulation ---
        "hpa axis activation", "cortisol dysregulation", "stress response", "epigenetic modifications",

        # --- Misc Pathogenic Mechanisms ---
        "endoplasmic reticulum stress", "genomic rna replication", "viral shedding",
        "ribosomal hijacking", "translation inhibition", "pathogenic molecule release",
        "oe structural damage", "chronic infection", "increased neurological manifestation"
    }


    if any(kw in entity_lower for kw in cell_keywords):
        return "Cell", "CL", None

    # === Check for Anatomical Structures ===
    for kw in anatomy_keywords:
        if kw in entity_lower.replace("_", " "):  # also match "BLOOD_BRAIN_BARRIER"
            return "Anatomical_Structure", "UBERON", None

    if any(kw in entity_lower for kw in protein_keywords):
        return "Protein", "PR", None
        
    if any(kw in entity_lower for kw in biological_process_keywords):
        return "Biological Process", "GO", None
    # === 1. Gene-Specific Pre-check ===
    # If the entity is all uppercase and short (e.g. "NOS2"), assume it is a gene.
    if entity_gene.isupper() and len(entity_gene) <= 6:
        label, namespace, ontology_id = query_hgnc_fuzzy(entity_gene)
        if label:
            return sanitize_label(label), sanitize_label(namespace), ontology_id

    # === 2. Broad Queries ===
    # Try a broad OLS query with a higher threshold (e.g., 0.6) for increased precision.
    label, namespace, ontology_id = query_ols_fuzzy(entity_lower, threshold=0.6)
    if label:
        return sanitize_label(label), sanitize_label(namespace), ontology_id

    # Next, try a broad MESH query.
    label, namespace, ontology_id = query_mesh_nih(entity_lower)
    if label:
        return sanitize_label(label), sanitize_label(namespace), ontology_id

    # === 3. Specialized Queries ===
    # (a) Phenotype: keywords like "phenotype", "clinical", "abnormal", etc.
    if any(kw in entity_lower for kw in ["phenotype", "clinical", "abnormal", "trait", "symptom"]):
        label, namespace, ontology_id = query_hpo_fuzzy(entity_lower)
        if label:
            return sanitize_label(label), sanitize_label(namespace), ontology_id
        label, namespace, ontology_id = query_mammalian_phenotype_fuzzy(entity_lower)
        if label:
            return sanitize_label(label), sanitize_label(namespace), ontology_id
        return "Phenotype", "HP", None

    # (b) Pathway / Biological Process: keywords like "process", "activation", "pathway", etc.
    if any(kw in entity_lower for kw in ["activation", "process", "pathway", "signaling", "regulation", "metabolism"]):
        label, namespace, ontology_id = query_pathway_databases(entity_lower)
        if label and ontology_id:
            return sanitize_label(label), sanitize_label(namespace), ontology_id
        label, namespace, ontology_id = query_gene_ontology_fuzzy(entity_lower)
        if label:
            return sanitize_label(label), sanitize_label(namespace), ontology_id
        return "Biological_Process", "GO", None

    # (c) Protein: keywords like "cytokine", "interleukin", "chemokine", "protein", etc.
    if any(kw in entity_lower for kw in ["cytokine", "interleukin", "chemokine", "protein"]):
        label, namespace, ontology_id = query_protein_ontology_fuzzy(entity_lower)
        if label:
            return sanitize_label(label), sanitize_label(namespace), ontology_id
        return "Protein", "PR", None

    # (d) Chemical: keywords like "acid", "amine", "hormone", "neurotransmitter", etc.
    if any(kw in entity_lower for kw in ["acid", "amine", "hormone", "neurotransmitter", "melanin", "lipid", "compound", "chemical", "drug", "anti-inflammatory"]):
        chebi_result = query_chembl_compound(entity_lower)
        if chebi_result:
            return chebi_result
        return "Chemical", "Chembl", None

    # (e) Additional Specialized Types
    if any(kw in entity_lower for kw in ["cell", "cyte"]):
        return "Cell", "CL", None
    if any(kw in entity_lower for kw in ["inheritance", "genetic pattern", "familial"]):
        return "Inheritance", "GENO", None

    # === 4. Final Fallback ===
    return "Unknown", "Unknown", None

def determine_entity_label(entity):
    """
    Determine the entity label and obtain an ontology ID using external queries.
    Returns: tuple: (label_type, namespace, ontology_id)
    """
    entity = entity.strip()

    if any(sep in entity for sep in [';', ',']):
        parts = re.split(r'[;,]+', entity)
        labels = [determine_entity_label(part.strip()) for part in parts if part.strip()]
        from collections import Counter
        label_counts = Counter([lbl for lbl, _, _ in labels if lbl != "Unknown"])
        if label_counts:
            best_label = label_counts.most_common(1)[0][0]
            for lbl, ns, oid in labels:
                if lbl == best_label:
                    return sanitize_label(lbl), sanitize_label(ns), oid
        return "Unknown", "Unknown", None

    entity_lower = clean_entity_text(entity)
    entity_gene = entity.upper()

    interaction_keywords = list({
        "binding", "interaction", "complex", "association", "receptor-ligand",
        "co-localization", "direct interaction", "protein binding"
    })

    response_keywords = list({
        "response", "production", "generation", "secretion", "activation",
        "degranulation", "amplification", "expression", "modulation",
        "presentation", "induction", "elevation"
    })

    non_cell_process_keywords = list({
        "activation", "migration", "differentiation", "proliferation",
        "response", "infiltration", "signaling", "secretion", "maturation",
        "apoptosis", "physiology", "function", "recruitment", "regulation",
        "adhesion", "chemotaxis", "invasion", "homing", "priming", "stress"
    })

    anatomy_exclusion_keywords = list({
        "injury", "impairment", "abnormality", "dysfunction", "symptom",
        "fog", "condition", "syndrome", "lesion", "damage", "degeneration",
        "atrophy", "demyelination", "infection", "encephalopathy",
        "encephalitis", "stroke", "neuropathy"
    })

    phenotype_keywords = list({
        "phenotype", "clinical", "abnormal", "trait", "symptom",
        "appearance", "presentation", "manifestation", "morphology",
        "disability", "impairment", "deficit", "dysfunction", "weakness",
        "atrophy", "hypertrophy", "dystrophy", "degeneration", "delay",
        "malformation", "dysmorphism", "disfigurement", "disturbance",
        "palsy", "ataxia", "spasticity", "hyperactivity", "seizure",
        "deformity", "irregularity", "anomaly", "instability", "demyelination",
        "loss", "retardation", "microcephaly", "macrocephaly", "cognitive impairment",
        "gliosis", "microgliosis", "astrocytosis", "nerve degeneration"
    })

    process_keywords = list({
        "activation", "process", "pathway", "signaling", "regulation",
        "metabolism", "biosynthesis", "catabolism", "transport",
        "localization", "phosphorylation", "inflammation", "degradation",
        "response", "transduction", "repair", "replication", "division",
        "translation", "transcription", "uptake", "release", "homeostasis",
        "interferon", "production", "binding", "interaction", "modulation",
        "clearance", "processing", "response to", "induction", "cascade",
        "biosynthetic", "maturation", "stress", "phagocytosis", "toxicity",
        "cellular stress", "reticulophagy"
    })

    chemical_keywords = list({
        "acid", "amine", "hormone", "neurotransmitter", "melanin", "lipid",
        "compound", "chemical", "drug", "anti-inflammatory", "inhibitor",
        "agonist", "antagonist", "toxin", "metabolite", "cofactor", "substrate",
        "reactant", "effector", "small molecule", "steroid", "alkaloid",
        "opioid", "antibiotic", "vitamin", "mineral", "metal", "element",
        "species", "pollutant", "emitter", "contaminant", "additive", "autoantibody"
    })

    cell_keywords = list({
        "cell", "cells", "cyte", "cytes", "blast", "blasts", "fibroblast", "fibroblasts",
        "macrophage", "macrophages", "neuron", "neurons", "neuroblast", "neuroblasts",
        "astrocyte", "astrocytes", "oligodendrocyte", "oligodendrocytes", "lymphocyte",
        "lymphocytes", "monocyte", "monocytes", "t cell", "b cell", "nk cell",
        "natural killer cell", "dendritic cell", "stem cell", "pluripotent",
        "multipotent", "totipotent", "progenitor", "hematopoietic", "epithelial",
        "mesenchymal", "microglia", "schwann cell", "retinal ganglion", "rods",
        "cones", "osteoblast", "osteocyte", "chondrocyte", "hepatocyte",
        "myocyte", "keratinocyte", "beta cell", "alpha cell", "islet cell",
        "germ cell", "oocyte", "spermatocyte", "erythrocyte", "rbc", "wbc",
        "thrombocyte", "megakaryocyte"
    })

    anatomy_keywords = list({
        "brain", "olfactory", "bulb", "epithelium", "barrier", "plate", "tissue",
        "organ", "ventricle", "cortex", "hippocampus", "amygdala", "hypothalamus",
        "pituitary", "thalamus", "spinal", "nervous system", "axon", "dendrite",
        "glomerulus", "ganglion", "blood-brain barrier", "bone", "skin", "muscle",
        "kidney", "lung", "heart", "spleen", "liver", "cranium", "nerve", "membrane",
        "tract", "artery", "capillary", "astrocyte", "oligodendrocyte"
    })

    protein_keywords = list({
        "protein", "receptor", "enzyme", "cytokine", "interleukin", "chemokine",
        "ligand", "kinase", "channel", "ion channel", "transporter", "antibody",
        "peptide", "hormone receptor", "growth factor", "transcription factor",
        "mrna binding protein", "rna binding protein", "binding protein"
    })

    components = re.split(r'[\s,_\-]+', entity_lower)
    is_cell = any(c in cell_keywords for c in components)
    is_process = any(c in non_cell_process_keywords for c in components)

    if is_cell and is_process:
        return "Cell_Phenotype", "Custom", None

    skip_cell_labeling = any(word in entity_lower for word in non_cell_process_keywords)
    skip_anatomy_labeling = any(word in entity_lower for word in anatomy_exclusion_keywords)

    if not skip_cell_labeling and any(kw in entity_lower for kw in cell_keywords):
        return "Cell", "CL", None

    if any(kw in entity_lower for kw in phenotype_keywords):
        return "Phenotype", "HP", None

    if any(kw in entity_lower for kw in process_keywords + interaction_keywords + response_keywords):
        return "Biological_Process", "GO", None

    if not skip_anatomy_labeling and any(kw in entity_lower.replace("_", " ") for kw in anatomy_keywords):
        if not any(kw in entity_lower for kw in process_keywords + non_cell_process_keywords):
            return "Anatomical_Structure", "UBERON", None

    if any(kw in entity_lower for kw in protein_keywords):
        gene_candidate = query_hgnc_fuzzy(entity_gene)
        if gene_candidate:
            if "receptor" in entity_lower or "protein" in entity_lower:
                return "Protein", "PR", None
            else:
                return "Gene", gene_candidate[1], gene_candidate[2]
        return "Protein", "PR", None

    if entity_gene.isupper() and len(entity_gene) <= 6:
        label, namespace, ontology_id = query_hgnc_fuzzy(entity_gene)
        if label:
            return sanitize_label(label), sanitize_label(namespace), ontology_id

    label, namespace, ontology_id = query_ols_fuzzy(entity_lower, threshold=0.6)
    if label:
        return sanitize_label(label), sanitize_label(namespace), ontology_id

    label, namespace, ontology_id = query_mesh_nih(entity_lower)
    if label:
        return sanitize_label(label), sanitize_label(namespace), ontology_id

    if any(kw in entity_lower for kw in chemical_keywords):
        chebi_result = query_chembl_compound(entity_lower)
        if chebi_result:
            return chebi_result
        return "Chemical", "Chembl", None

    if any(kw in entity_lower for kw in ["inheritance", "genetic pattern", "familial"]):
        return "Inheritance", "GENO", None

    return "Unknown", "Unknown", None


# ------------------------------------------------------------
# Neo4j Uploader Class
# ------------------------------------------------------------
class Neo4jUploader:
    """
    Handles uploading of semantic triples into a Neo4j graph database using the official Neo4j driver.
    
    Methods:
        upload_triples(triples): Uploads a DataFrame of triples into Neo4j.
        close(): Closes the connection to the Neo4j driver.
    """

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()

    def upload_triples(self, triples: pd.DataFrame):
        with self.driver.session() as session:
            for idx, row in triples.iterrows():
                logger.info(f"Processing row {idx + 1}...")
                session.write_transaction(self.create_relationship, row)

    @staticmethod
    def create_relationship(tx, row):
        start_entity = row['Subject'].strip().upper()
        end_entity = row['Object'].strip().upper()
        rel_type = sanitize_label(row['Predicate'])
        image_url = str(row.get('URL', 'Not Available'))
        mechanism = row.get('Pathophysiological Process', 'Unknown')
        source = row.get('source', 'Unknown')
        cell = row.get('Cell')
        anatomy = row.get('Anatomy')
        meta_data = [cell, anatomy] if cell is not None and anatomy is not None else "None"

        # Optional fields
        pmid = int(row.get('PMID'))
        title = row.get('Title')
        evidence = row.get('Evidence')

        s_lbl, s_ns, s_oid = determine_entity_label(start_entity)
        e_lbl, e_ns, e_oid = determine_entity_label(end_entity)

        if not s_lbl or not e_lbl:
            raise ValueError(f"Could not determine label: {start_entity} -> {end_entity}")

        s_lbl = sanitize_label(s_lbl)
        e_lbl = sanitize_label(e_lbl)

        # Merge start node
        tx.run(f"""
            MERGE (a:{s_lbl} {{name: $start}})
            ON CREATE SET a.namespace = $sns, a.ontology_id = $soid, a.created_at = timestamp()
            ON MATCH SET a.last_seen = timestamp(),
                        a.ontology_id = CASE WHEN $soid <> 'UNKNOWN' THEN $soid ELSE a.ontology_id END
        """, start=start_entity, sns=s_ns or "UNKNOWN", soid=s_oid or "UNKNOWN")

        # Merge end node
        tx.run(f"""
            MERGE (b:{e_lbl} {{name: $end}})
            ON CREATE SET b.namespace = $ens, b.ontology_id = $eoid, b.created_at = timestamp()
            ON MATCH SET b.last_seen = timestamp(),
                        b.ontology_id = CASE WHEN $eoid <> 'UNKNOWN' THEN $eoid ELSE b.ontology_id END
        """, end=end_entity, ens=e_ns or "UNKNOWN", eoid=e_oid or "UNKNOWN")

        # Merge relationship — include optional fields only if available
        tx.run(f"""
            MATCH (a:{s_lbl} {{name: $start}})
            MATCH (b:{e_lbl} {{name: $end}})
            MERGE (a)-[r:{rel_type.upper()}]->(b)
            ON CREATE SET r.image_url = $image_url,
                        r.mechanism = $mech,
                        r.source = $src,
                        r.meta_data = $md,
                        r.pmid = CASE WHEN $pmid IS NOT NULL THEN $pmid ELSE NULL END,
                        r.title = CASE WHEN $title IS NOT NULL THEN $title ELSE NULL END,
                        r.evidence = CASE WHEN $evidence IS NOT NULL THEN $evidence ELSE NULL END,
                        r.created_at = timestamp()
        """, start=start_entity, end=end_entity,
            image_url=image_url, mech=mechanism, src=source, md=meta_data,
            pmid=pmid, title=title, evidence=evidence)
        
        logger.info(f"  Uploaded: {start_entity} ({s_lbl}) -> {end_entity} ({e_lbl})")
        time.sleep(0.5)



# ──────────────────────────────────────────────────────────────────────────────
#  Helper to load & tag each CSV, with delimiter sniffing
# ──────────────────────────────────────────────────────────────────────────────

def load_and_tag(file_path: str, source_tag: str) -> pd.DataFrame:
    """
    Load a CSV file containing semantic triples, infer delimiter, and tag rows with the source type.
    Ensures required columns exist and fills missing metadata.
    
    Args:
        file_path (str): Path to the CSV file.
        source_tag (str): Label for the data source (e.g., "CBM", "GPT").
    
    Returns:
        pd.DataFrame: Parsed and annotated dataframe of triples.
    """

    # sniff delimiter from a sample
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = f.read(2048)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[',',';','\t','|'])
            sep = dialect.delimiter
        except csv.Error:
            sep = ','

    df = pd.read_csv(file_path, sep=sep, encoding="utf-8", engine='python')
    df.fillna("Unknown", inplace=True)

    # Check for core required columns (excluding URL which might be missing)
    core_required = ["Subject", "Object", "Predicate", "Pathophysiological Process"]
    missing = [c for c in core_required if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path} missing required columns: {missing}")

    # Add URL column if missing (with default value)
    if "URL" not in df.columns:
        df["URL"] = "Not Available"
        logger.info(f"Added missing URL column to {file_path} with default value 'Not Available'")

    df["source"] = source_tag

    # Add Cell and Anatomy columns if missing
    for col in ("Cell", "Anatomy"):
        if col not in df.columns:
            df[col] = None

    # Optional columns: PMID, Title, Evidence
    for col in ("PMID", "Title", "Evidence"):
        if col not in df.columns:
            df[col] = None


    return df


# ──────────────────────────────────────────────────────────────────────────────
#  Main: load both CBM & GPT files, concatenate, upload
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload full-text GPT triples to Neo4j.")
    parser.add_argument(
        "--input",
        default="data/gold_standard_comparison/Triples_Full_Text_GPT_for_comp_cleaned.csv",
        help="Path to full-text triples file (CSV or XLSX). "
             "Defaults to data/gold_standard_comparison/Triples_Full_Text_GPT_for_comp_cleaned.csv"
    )
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j Bolt URI (default: bolt://localhost:7687)")
    parser.add_argument("--user", default="neo4j", help="Neo4j username (default: neo4j)")
    parser.add_argument("--password", help="Neo4j password (if omitted, you will be prompted)")
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"--input not found: {os.path.abspath(args.input)}")

    # Load data
    fulltext_df = load_and_tag(args.input, "GPT-fulltext")

    # Keep only required + optional columns
    base_cols = [
        "Subject", "Object", "Predicate", "URL",
        "Pathophysiological Process", "source", "Cell", "Anatomy"
    ]
    optional_cols = [c for c in ["PMID", "Title", "Evidence"] if c in fulltext_df.columns]
    triples = (
        fulltext_df[base_cols + optional_cols]
        .dropna(subset=["Subject", "Object", "Predicate"])
        .query("Subject != '' and Object != '' and Predicate != ''")
    )

    # Password: CLI flag has priority; otherwise prompt
    neo4j_password = args.password if args.password is not None else getpass("Enter Neo4j password: ")

    # Upload to Neo4j
    uploader = Neo4jUploader(uri=args.uri, user=args.user, password=neo4j_password)
    try:
        uploader.upload_triples(triples)
        logger.info("All data uploaded successfully.")
    finally:
        uploader.close()

if __name__ == "__main__":
    main()

# python src/neo4j_upload_fulltext.py --input data/gold_standard_comparison/Triples_Full_Text_GPT_for_comp_cleaned.csv --password YOUR_Neo4j_PASSWORD