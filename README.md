# 10-Kwery

10-Kwery is a Retrieval-Augmented Generation (RAG) system for querying SEC 10-K filings using natural language. It downloads filings from the SEC EDGAR database, processes and indexes them into a hybrid retrieval system, and generates citation-grounded answers using a Large Language Model (LLM).

The project demonstrates an end-to-end financial document retrieval pipeline consisting of data ingestion, preprocessing, indexing, hybrid retrieval, answer generation, and evaluation.


## Features

- Download SEC 10-K filings from EDGAR
- HTML parsing and section-aware document chunking
- Token-based chunking with overlap using **tiktoken**
- Dense semantic retrieval with **BAAI/bge-small-en-v1.5**
- Keyword retrieval using **BM25**
- Hybrid retrieval via **Reciprocal Rank Fusion (RRF)**
- Cross-encoder reranking with **cross-encoder/ms-marco-MiniLM-L-6-v2**
- Citation-grounded answer generation using **Groq**
- Evaluation using **MRR**, **Hit@1**, **Hit@5**, citation validity, and keyword coverage


# Project Structure

```text
10Kwery/
├── README.md
├── requirements.txt
├── src/
│   ├── ingest.py
│   ├── chunk.py
│   ├── index.py
│   ├── retrieve.py
│   ├── generate.py
│   └── evaluate.py
└── eval/
    └── qa_pairs.json
```


# Architecture

```
SEC EDGAR
    │
    ▼
Download Filings
    │
    ▼
HTML Parsing
    │
    ▼
Section-aware Chunking
    │
    ▼
Embedding Generation
    │
    ▼
ChromaDB Index
    │
    │
User Query
    │
    ▼
Embedding Generation
    │
    ├──────────────┐
    │              │
    ▼              ▼
Vector Search    BM25 Search
    │              │
    └──────┬───────┘
           │
           ▼
Reciprocal Rank Fusion
           │
           ▼
Cross-Encoder Reranker
           │
           ▼
Top Ranked Chunks
           │
           ▼
        Groq LLM
           │
           ▼
Citation-grounded Answer
```


# Models

| Component | Model |
|----------|-------|
| Embedding | **BAAI/bge-small-en-v1.5** |
| Reranker | **cross-encoder/ms-marco-MiniLM-L-6-v2** |
| LLM | **Llama 3.3 70B Versatile (Groq)** |


# Technologies

- Python
- BeautifulSoup
- tiktoken
- ChromaDB
- Sentence Transformers
- CrossEncoder
- BM25
- Groq API
- NumPy
- Requests
- tqdm


# Installation

Clone the repository.

```bash
git clone https://github.com/<username>/10Kwery.git
cd 10Kwery
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
SEC_USER_AGENT=Your Name your.email@example.com
```

`SEC_USER_AGENT` is required by the SEC EDGAR API and should identify you with your name and email address.

Example:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
SEC_USER_AGENT=John Doe john.doe@example.com
```


# Usage

### 1. Download SEC filings

```bash
python src/ingest.py
```

### 2. Chunk the filings

```bash
python src/chunk.py
```

### 3. Build the retrieval index

```bash
python src/index.py
```

This step:

- Generates embeddings using **BAAI/bge-small-en-v1.5**
- Builds a **ChromaDB** vector database
- Builds a **BM25** keyword index


### 4. Query the system

```bash
python src/generate.py "How did Apple's R&D spending change year over year?"
```

Example output:

```text
ANSWER
------------------------------------------------------------
Apple's research and development expense increased from ...
[1][2]

SOURCES
------------------------------------------------------------
[1] AAPL 10-K filed 2024-11-01
[2] AAPL 10-K filed 2023-11-03
```


### 5. Evaluate retrieval performance

```bash
python src/evaluate.py
```

The evaluation compares:

- Vector-only retrieval
- Hybrid retrieval (Vector + BM25 + RRF + Cross-Encoder)

Metrics reported include:

- Mean Reciprocal Rank (MRR)
- Hit@1
- Hit@5
- Citation Validity
- Expected Keyword Coverage


# Retrieval Pipeline

1. Encode the user query using **BAAI/bge-small-en-v1.5**
2. Retrieve candidates from **ChromaDB**
3. Retrieve candidates using **BM25**
4. Merge both rankings using **Reciprocal Rank Fusion (RRF)**
5. Rerank candidates with **cross-encoder/ms-marco-MiniLM-L-6-v2**
6. Pass the highest-ranked chunks to the LLM
7. Generate a citation-grounded response


# Evaluation Metrics

| Metric | Description |
|---------|-------------|
| MRR | Measures how highly the correct document is ranked. |
| Hit@1 | Percentage of queries where the correct company is ranked first. |
| Hit@5 | Percentage of queries where the correct company appears within the top five results. |
| Citation Validity | Verifies that generated citations correspond to retrieved chunks. |
| Keyword Coverage | Measures how well generated answers cover the expected concepts. |


# Future Improvements

- Metadata filtering
- Streaming responses
- Web interface
- Multi-turn conversations
- Support for additional SEC filing types (10-Q, 8-K)


# License

This project is intended for educational and research purposes.