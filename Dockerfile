# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p data/triples_output \
    data/enrichment_data \
    data/URL_relevance_analysis \
    data/gold_standard_comparison \
    data/prompt_engineering \
    data/MeSh_data \
    data/neo4j_queries \
    data/CBM_data

# Set Python path to include src directory
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command (opens bash for interactive use)
CMD ["/bin/bash"]

# Alternative: Uncomment below to run a specific script by default
# CMD ["python", "src/Triple_Extraction_GPT4o.py", "--help"]