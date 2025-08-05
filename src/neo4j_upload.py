# neo4j_upload.py

"""
Neo4j Triple Uploader Script
Authors: Negin Babaiha, Elizaveta Popova
Institution: University of Bonn, Fraunhofer SCAI
Date: 05/2025

Description:
    This script uploads biomedical knowledge triples into a Neo4j graph database.
    Each triple is structured as (subject–predicate–object), enriched with metadata
    (source, image URL, pathophysiological process), and semantically labeled using
    ontology-based entity recognition (e.g., GO, DOID, MESH, HGNC).

    The subject and object nodes are classified using lexical cues, external ontology APIs,
    and heuristic rules. The resulting labeled nodes and relationships are merged into
    a Neo4j database using Cypher queries.

Key functionalities:
    1. Loads two CSV files: CBM gold standard triples and GPT-generated triples.
    2. Determines entity types (e.g., Gene, Disease, Process) via string patterns and ontology lookups.
    3. Uploads entities and relationships into Neo4j, with proper labels and metadata.
    4. Supports flexible CSV parsing (auto delimiter detection).
    5. Handles optional metadata columns ("Cell", "Anatomy") gracefully.

Inputs:
    --cbm: Path to gold standard triples (CSV)
    --gpt: Path to GPT-generated triples (CSV)

Output:
    - Biomedical triples inserted into Neo4j (bolt://localhost:7687)
    - Console logs indicating upload status and classification per triple

Usage:
    python src/neo4j_upload.py --cbm data/gold_standard_comparison/Triples_CBM_Gold_Standard_cleaned.csv --gpt data/gold_standard_comparison/Triples_GPT_for_comparison.csv

Requirements:
    - pandas
    - neo4j
    - requests
    - rapidfuzz
    - argparse
    - compatible ontology APIs (OLS, MeSH, ChEMBL, UniProt)
"""

from __future__ import annotations
from operator import delitem
import sys
import pandas as pd
import requests
import argparse
from getpass import getpass
from rapidfuzz import fuzz, process
from neo4j import GraphDatabase
import os
import re
from difflib import get_close_matches
import urllib
import urllib.parse
import logging
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import csv
import xml.etree.ElementTree as ET
from rapidfuzz import fuzz

def sanitize_label(label):
    """
    Converts a label into a Neo4j-safe format.

    Replaces disallowed characters with underscores and trims whitespace.
    Ensures compatibility with Cypher node/relationship identifiers.

    Args:
        label (str): Input label string.

    Returns:
        str: Sanitized label suitable for Cypher queries.
    """
    label_str = str(label).strip()
    # Check if the label contains only allowed characters (letters, digits, and underscores)
    if re.fullmatch(r'\w+', label_str):
        return label_str
    # Otherwise, replace problematic characters
    return label_str.replace("-", "_").replace(" ", "_").replace(".", "_")


def clean_entity_text(entity: str) -> str:
    """
    Cleans and normalizes biomedical entity strings by lowercasing and removing special characters.

    Args:
        entity (str): Raw entity string.

    Returns:
        str: Cleaned, normalized text.
    """
    entity = entity.strip().lower()
    # Replace common separators with space or underscore
    entity = entity.replace("-", " ").replace("_", " ").replace("+", " plus ")
    # Remove or normalize any remaining special characters
    entity = re.sub(r"[^\w\s]", "", entity)  # keeps letters, numbers, spaces
    entity = re.sub(r"\s+", " ", entity)     # collapse multiple spaces
    return entity.strip()

def determine_entity_type(entity):
    """
    Determines if an entity is a likely gene or falls under another biomedical category
    using rule-based checks and pattern matching.

    Args:
        entity (str): Entity name.

    Returns:
        tuple: (bool, str or None) indicating if it's a gene and its label.
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
    """
    Infers entity type by examining content metadata like label, description, and semantic types.

    Args:
        result (dict): Metadata from ontology search result.

    Returns:
        str: Inferred type (e.g., Disease, Phenotype).
    """
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
    Queries the Human Phenotype Ontology using fuzzy string matching.

    Args:
        entity (str): Entity to search.
        threshold (float): Minimum fuzzy match threshold.

    Returns:
        tuple: (label_type, namespace, ontology_id)
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
    Performs fuzzy matching on Disease Ontology (DOID) terms using OLS API.

    Args:
        entity (str): Disease-like term.
        threshold (float): Match threshold.

    Returns:
        tuple: (label_type, namespace, ontology_id)
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
    Attempts to classify a term as a gene using HGNC and RefSeq APIs.

    Args:
        entity (str): Gene name or alias.

    Returns:
        tuple or None: (label, namespace, ontology_id)
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
    Fuzzy queries the Gene Ontology (GO) for biological processes.

    Args:
        entity (str): Entity to classify.
        threshold (float): Match threshold.

    Returns:
        tuple or None: (label, namespace, ontology_id)
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
    Queries Mammalian Phenotype Ontology (MP) for a term using fuzzy matching.

    Args:
        entity (str): Term to search.
        threshold (float): Match threshold.

    Returns:
        tuple: (label_type, namespace, ontology_id)
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

def query_kegg_pathway(entity):
    """
    Queries KEGG for biological pathways related to an entity.

    Args:
        entity (str): Pathway-related term.

    Returns:
        tuple or None: (label, namespace, ontology_id)
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
    Searches Reactome for pathway matches based on text queries.

    Args:
        entity (str): Entity name.

    Returns:
        tuple or None: (label, namespace, pathway_id)
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
    Queries WikiPathways to retrieve pathway identifiers.

    Args:
        entity (str): Search query.

    Returns:
        tuple or None: (label, namespace, pathway_id)
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
    Searches across KEGG, Reactome, and WikiPathways in sequence.

    Args:
        entity (str): Entity to classify.

    Returns:
        tuple or None: (label, namespace, ontology_id)
    """
    for query_fn in [query_kegg_pathway, query_reactome_pathway, query_wikipathways]:
        label, namespace, pathway_id = query_fn(entity)
        if label and pathway_id:
            return label, namespace, pathway_id
    return None, None, None

def query_mesh_nih(entity, fuzzy_threshold=60):
    """
    Queries NIH MESH API and classifies term based on tree and qualifier metadata.

    Args:
        entity (str): MeSH term to search.
        fuzzy_threshold (int): Fuzzy match score threshold.

    Returns:
        tuple: (label_type, namespace, mesh_id)
    """

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
    Queries Protein Ontology or UniProt to identify protein-related terms.

    Args:
        entity (str): Protein name or synonym.
        threshold (float): Fuzzy matching threshold.

    Returns:
        tuple or None: (label, namespace, ontology_id)
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
    Generic fuzzy ontology lookup using EBI OLS for broad classification.

    Args:
        entity (str): Term to look up.
        threshold (float): Match threshold.

    Returns:
        tuple: (label_type, namespace, ontology_id)
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
    Determines the semantic label, namespace, and ontology ID for a biomedical entity.

    This function combines string heuristics, ontology queries (e.g., HGNC, GO, MeSH),
    and domain-specific keyword checks to classify an input term into categories such as:
    Gene, Protein, Disease, Phenotype, Pathway, Cell, Anatomical_Structure, etc.

    It also supports compound terms separated by commas or semicolons.

    Args:
        entity (str): The biomedical entity string to classify.

    Returns:
        tuple: (label_type: str, namespace: str, ontology_id: str or None)
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
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()

    def upload_triples(self, triples: pd.DataFrame):
        with self.driver.session() as session:
            for idx, row in triples.iterrows():
                logger.info(f"Processing row {idx + 1}...")
                session.execute_write(self.create_relationship, row)  # Use session.write_transaction(self.create_relationship, row) for Neo4j < 5.x

    @staticmethod
    def create_relationship(tx, row):
        start_entity = row['Subject'].strip().upper()
        end_entity   = row['Object'].strip().upper()
        rel_type     = sanitize_label(row['Predicate'])
        image_url         = str(row['URL'])
        mechanism    = row['Pathophysiological Process']
        source       = row['source']
        cell         = row.get('Cell')
        anatomy      = row.get('Anatomy')
        meta_data    = [cell, anatomy] if cell is not None and anatomy is not None else "None"

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

        # Merge relationship
        tx.run(f"""
            MATCH (a:{s_lbl} {{name: $start}})
            MATCH (b:{e_lbl} {{name: $end}})
            MERGE (a)-[r:{rel_type.upper()} {{image_url: $image_url, mechanism: $mech, source: $src, meta_data: $md}}]->(b)
            ON CREATE SET r.created_at = timestamp()
        """, start=start_entity, end=end_entity,
             rel=rel_type, image_url=image_url, mech=mechanism,
             src=source, md=meta_data)

        logger.info(f"  Uploaded: {start_entity} ({s_lbl}) -> {end_entity} ({e_lbl})")
        time.sleep(0.5)


# ──────────────────────────────────────────────────────────────────────────────
#  Helper to load & tag each CSV, with delimiter sniffing
# ──────────────────────────────────────────────────────────────────────────────

def load_and_tag(file_path: str, source_tag: str) -> pd.DataFrame:
    """
    Loads a CSV file containing triples and applies a source tag and standard columns.

    Args:
        file_path (str): Path to CSV file.
        source_tag (str): Label for the data source (e.g., 'CBM', 'GPT').

    Returns:
        pd.DataFrame: Cleaned and standardized dataframe.
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

    required = ["Subject", "Object", "Predicate", "URL", "Pathophysiological Process"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path} missing required columns: {missing}")

    df["source"] = source_tag

    for col in ("Cell", "Anatomy"):
        if col not in df.columns:
            df[col] = None

    return df


# ──────────────────────────────────────────────────────────────────────────────
#  Main: load both CBM & GPT files, concatenate, upload
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload curated and GPT triples to Neo4j.")
    parser.add_argument("--cbm", required=True, help="Path to CBM gold standard CSV file")
    parser.add_argument("--gpt", required=True, help="Path to GPT-predicted triples CSV file")
    args = parser.parse_args()

    cbm_fp = args.cbm
    gpt_fp = args.gpt

    cbm_df = load_and_tag(cbm_fp, "CBM")
    gpt_df = load_and_tag(gpt_fp, "GPT")

    all_df = pd.concat([cbm_df, gpt_df], ignore_index=True)

    cols = [
        "Subject", "Object", "Predicate", "URL",
        "Pathophysiological Process", "source", "Cell", "Anatomy"
    ]
    triples = (
        all_df[cols]
        .dropna(subset=["Subject", "Object", "Predicate"])
        .query("Subject != '' and Object != '' and Predicate != ''")
    )

    neo4j_password = getpass("Enter Neo4j password: ")

    uploader = Neo4jUploader(
        uri="bolt://localhost:7687",
        user="neo4j",
        password=neo4j_password
    )
    try:
        uploader.upload_triples(triples)
        logger.info("All data uploaded successfully.")
    finally:
        uploader.close()


if __name__ == "__main__":
    main()
