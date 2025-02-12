# Image-based Information Extraction using LLMs for Biomedical Research

## Overview
This project develops a novel methodology for extracting structured knowledge from biomedical images using Large Language Models (LLMs). We focus specifically on understanding the relationship between COVID-19 and neurodegeneration through analysis of scientific figures and graphical abstracts. The project leverages GPT-4o for image analysis and semantic triple extraction, creating a comprehensive knowledge graph of biomedical relationships.

![Workflow](workflow.svg)

## Key Features
- Automated collection of relevant biomedical images using Google Image Search
- Multi-stage filtering process using GPT-4o for relevance assessment
- Semantic triple extraction (subject-predicate-object) from images using GPT-4V and GPT-4o
- Knowledge graph construction for COVID-19 and neurodegeneration relationships
- High accuracy and precision in image classification and information extraction

## Project Structure
```
image-based-information-extraction-LLM/
├── src/
│   ├── data_collection/           # Scripts for Google Image Search automation
│   ├── image_processing/          # Image validation and preprocessing
│   ├── triple_extraction/         # GPT-4 based triple extraction
│   └── knowledge_graph/           # Knowledge graph construction
├── notebooks/
│   ├── Triple_Extraction_GPT4o.ipynb  # Main implementation notebook
│   └── analysis/                  # Additional analysis notebooks
├── data/
│   ├── raw/                      # Collected image URLs
│   ├── processed/                # Filtered and validated images
│   └── results/                  # Extracted triples and knowledge graphs
└── docs/                         # Documentation and methodology details
```

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API access (for GPT-4V and GPT-4o)
- Required Python packages

```bash
pip install -r requirements.txt
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/NeginBabaiha/image-based-information-extraction-LLM.git
cd image-based-information-extraction-LLM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

## Usage

### Image Collection
```python
from src.data_collection import ImageCollector

collector = ImageCollector()
urls = collector.search("Covid-19 and Neurodegeneration")
```

### Triple Extraction
```python
from src.triple_extraction import TripleExtractor

extractor = TripleExtractor()
triples = extractor.process_image(image_url)
```

## Methodology

Our workflow consists of several key stages:

1. **Data Collection**: Automated collection of 6,319 image URLs using Google Image Search
2. **URL Processing**: Validation and accessibility checking (3,614 valid URLs)
3. **Relevance Assessment**: 
   - First run: 626 relevant URLs
   - Second run: 567 refined URLs
   - Manual verification: 289 final images
4. **Triple Extraction**: Using GPT-4o for semantic relationship extraction
5. **Knowledge Graph Construction**: Building structured representations of biomedical relationships


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Negin Babaiha, Elizaveta Popova**
- Email:(negin.babaiha@scai.fraunhofer.de, elizaveta.popova@uni-bonn.de)

## Acknowledgments

Special thanks to OpenAI for providing access to GPT-4V and GPT-4o models, which were instrumental in this research.
