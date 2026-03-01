# Swiggy Annual Report RAG System

A RAG-based Q&A system built on top of Swiggy's FY 2023-24 Annual Report.

**Document source:** https://ir.swiggy.com/annual-reports

## How it works

PDF is parsed and split into chunks → embedded using sentence-transformers → stored in FAISS. On a query, relevant chunks are retrieved and passed to Mistral-7B on HuggingFace to generate a grounded answer.

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN="your-huggingface-token"
streamlit run app.py
```

## Tech used
- PyMuPDF — PDF parsing
- sentence-transformers — embeddings  
- FAISS — vector search
- Mistral-7B via HuggingFace — answer generation
- Streamlit — web interface
