
# Identifying Mechanistic Links Between COVID-19 and Neurodegenerative Diseases

**Authors**: Elizaveta Popova, Negin Sadat Babaiha  
**Institutions**: University of Bonn, Fraunhofer SCAI  
**Contact**: elizaveta.popova@uni-bonn.de, negin.babaiha@scai.fraunhofer.de  

---

## Abstract

This project presents a computational framework to identify mechanistic connections between **COVID-19** and **neurodegenerative diseases** by extracting and analyzing **semantic triples** from biomedical figures. The pipeline combines **LLM-based triple extraction**, **BioBERT-based semantic comparison**, and **MeSH-informed categorization** to map biological processes depicted in literature into structured and interpretable knowledge.

---

## Project Overview

### Goals
- Extract biological mechanisms from graphical abstracts
- Compare GPT-generated triples to a manually curated gold standard
- Classify processes into domain-relevant pathophysiological categories

### Key Techniques
- GPT-4o (multimodal) for image-to-text triple extraction
- BioBERT for semantic embedding and similarity
- Sentence-BERT + MeSH ontology for mechanistic classification

---

## Methodology Overview

### Step 1: Image Extraction and Filtering
- Automated Google Images scraping
- Relevance filtering via GPT-4o
- Output: `Relevant_URLs_only_GPT_4o.xlsx`

### Step 2: Triple Extraction from Biomedical Images
- GPT-4o queried with standardized prompts
- Output format: subject | predicate | object
- Output: `Triples_Final_All_Relevant.csv`

### Step 3: Triple Evaluation (Gold Standard Comparison)
- Uses BioBERT embeddings to compare GPT and CBM triples
- Applies both semantic and lexical similarity thresholds
- MeSH normalization improves robustness
- Outputs: match statistics, cosine similarity distribution

### Step 4: Mechanism Categorization
- Uses Sentence-BERT embeddings and MeSH keyword expansion
- Categories:
  - Viral Entry and Neuroinvasion
  - Immune and Inflammatory Response
  - Neurodegenerative Mechanisms
  - Vascular Effects
  - Psychological and Neurological Symptoms
  - Systemic Cross-Organ Effects
- Fallback: GPT-4o assigns categories for ambiguous cases

---

## Repository Structure

```
SCAI_CODE/
│
├── data/
│   ├── gold_standard_comparison/        ← Curated and predicted triples
│   ├── images_CBM/                      ← Images used for CBM manual triple extraction
│   ├── MeSh_data/                       ← MeSH XML & category/synonym outputs
│   ├── URL_relevance_analysis/          ← URL relevance check results (GPT-4o and manual)
│   └── triples_output/                  ← Extracted and categorized triples
│
├── notebooks/
│   └── MeSH_Keyword_Extraction.ipynb    ← Keyword extraction & entity normalization
│
├── src/
│   ├── Triple_Extraction_GPT4o.py
│   ├── Gold_Standard_Comparison_BioBERT.py
│   ├── Triples_Categorization.py
│   ├── GPT4o_uncategorized_handling.py
│   ├── Image_Enrichment_Analysis.py
│   ├── URLs_Relevance_Check.py
│   └── Hyperparameters_and_Prompt_Assessment.py
```

---

## Outputs

| File | Description |
|------|-------------|
| **TRIPLE EXTRACTION** |
| `Triples_Final_All_Relevant.csv/xlsx` | All semantic triples extracted from the full image pool using GPT-4o |
| `Triples_Final_comparison_with_CBM.csv/xlsx` | GPT triples for subset of images annotated by CBM |
| `Triples_GPT_for_comparison.xlsx` | GPT triples for CBM subset, with image-level mapping |
| `Triples_GPT_for_comparison_SubjObj_Categorized.xlsx/csv` | Same triples, with MeSH-based subject/object category labels |
| **GOLD STANDARD (CBM MANUAL ANNOTATION)** |
| `Triples_CBM_Gold_Standard.xlsx` | Manually curated CBM triples |
| `Triples_CBM_Gold_Standard_SubjObj_Categorized.xlsx/csv` | CBM triples with subject/object category labels |
| `Data_CBM.xlsx` | Image-level metadata for CBM-annotated subset |
| **CATEGORIZATION & NORMALIZATION** |
| `Triples_Final_All_Relevant_Categorized.xlsx/csv` | Categorized GPT triples via BERT + MeSH keywords |
| `Triples_Final_All_Relevant_Categorized_GPT4o.xlsx/csv` | Final categorized triples with GPT fallback |
| `mesh_category_terms.json` | MeSH category → keyword dictionary |
| `mesh_triples_synonyms.json` | Triple term → normalized MeSH descriptor & synonyms |
| **EVALUATION & RELEVANCE** |
| `Comparison_GPT_Manual_Relevance.xlsx` | Manual evaluation of GPT-extracted triples |
| `Relevant_URLs_only_GPT_4o.xlsx` | Final image set considered relevant by GPT-4o |
| `Supplementary_material_Table_1.xlsx` | Results from GPT prompt & hyperparameter tuning |

---

## How to Run (Pipeline)

```bash
# Step 1: Extract biomedical image links
python src/Image_Enrichment_Analysis.py   --query "Covid-19 and Neurodegeneration"   --main 100 --similar 100 --output_raw Enrichment_Search_URLs   --output_clean Enrichment_Cleaned --outdir ./data

# Step 2: Relevance check (GPT-based)
python src/URLs_Relevance_Check.py --input data/Enrichment_Search_URLs.xlsx --api_key YOUR_API_KEY

# Step 3: Extract semantic triples
python python src/Triple_Extraction_GPT4o.py --input data/Final_Relevant_URLs.xlsx --output_dir ./data/triples_output --api_key YOUR_API_KEY

# Step 4: Compare to gold standard using BioBERT
python src/Gold_Standard_Comparison_BioBERT.py   --gold data/gold_standard_comparison/Triples_CBM_Gold_Standard.xlsx   --eval data/gold_standard_comparison/Triples_GPT_for_comparison.xlsx

# Step 5: Classify pathophysiological processes (BERT + MeSH)
python src/Triples_Categorization.py   --input triples_output/Triples_Final_All_Relevant.csv   --output triples_output/Triples_Final_All_Relevant_Categorized   --mode pp

# Step 6: Resolve uncategorized entries with GPT-4o
python src/GPT4o_uncategorized_handling.py --input data/triples_output/Triples_Final_All_Relevant_Categorized.xlsx --output data/triples_output/Triples_Final_All_Relevant_Categorized_GPT4o --api_key YOUR_API_KEY
```

---

## MeSH Integration

This project uses:
- `desc2025.xml`: official MeSH descriptor file from NLM
- Keyword-based category matching (`mesh_category_terms.json`)
- Entity-level synonym normalization (`mesh_triples_synonyms.json`)

> [Download desc2025.xml](https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2025.xml)  
Place in: `/data/MeSh_data/desc2025.xml`

---

## Requirements

```bash
pip install -r requirements.txt
```

Key packages:
- `openai`
- `torch`
- `pandas`, `numpy`
- `sentence-transformers`
- `transformers`
- `openpyxl`, `matplotlib`, `imagehash`, `selenium`

---

## GPT API Usage
This project relies heavily on **OpenAI GPT-4o** for multimodal processing of biomedical images. You must have valid API access to use the scripts for:
- Relevance classification
- Triple extraction

Ensure that you:
1. Have an OpenAI account with GPT-4 API access.
2. Store your API key as an environment variable:
```bash
export OPENAI_API_KEY=sk-...
```
3. Or provide it as a command-line argument when running scripts.

⚠️ **Note**: Due to API call limits and costs, full pipeline execution may require batching or quota management.

---

## Citation

> Please cite this work as:


---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

- **Elizaveta Popova**  
  University of Bonn  
  elizaveta.popova@uni-bonn.de  

- **Negin Sadat Babaiha**  
  Fraunhofer SCAI  
  negin.babaiha@scai.fraunhofer.de  

---

## Funding

This research was supported by the 
Grant ID: 

---

## Reproducibility Checklist

| Requirement | Status |
|------------|--------|
| Open-source code | Available |
| Data files (intermediate and gold standard) | Included |
| Prompt templates | In scripts |
| Environment dependencies | `requirements.txt` |
| External model APIs | GPT-4o via OpenAI API (key required) |

---
