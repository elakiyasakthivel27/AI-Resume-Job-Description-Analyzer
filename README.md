# AI Resume & Job Description Analyzer

A RAG-based Generative AI application that analyzes a candidate's resume against a job description and generates relevant insights using semantic retrieval and a Large Language Model.

## Overview

The application allows users to upload a PDF resume and provide a job description. It processes both inputs, generates embeddings, stores them in a vector database, retrieves relevant information, and uses an LLM to generate a resume-to-job analysis.

## Architecture

```text
Resume & Job Description
        ↓
Text Extraction & Chunking
        ↓
Sentence Transformer Embeddings
        ↓
ChromaDB Vector Store
        ↓
Semantic Retrieval
        ↓
Groq LLM
        ↓
Resume–Job Analysis
```

## Features

- Upload and process PDF resumes
- Analyze resumes against job descriptions
- Generate embeddings using Sentence Transformers
- Store embeddings in ChromaDB
- Perform semantic similarity retrieval
- Separate resume and job-description content using metadata
- Generate AI-powered analysis using Groq LLM
- Interactive user interface built with Streamlit

## Technologies Used

- Python
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- Hugging Face Embeddings
- Groq LLM
- PyPDF
- RAG (Retrieval-Augmented Generation)

## How to Run

1. Clone the repository.

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project folder and add your Groq API key:

```text
GROQ_API_KEY=your_api_key_here
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

## Security

API keys and environment variables are excluded from the repository using `.gitignore`.

The `.env` file containing API credentials should never be committed to a public repository.

## Author

**Elakiya S**  
Full Stack + AI Developer

**GitHub:** @elakiyasakthivel27  
**LinkedIn:** linkedin.com/in/elakiyasakthivel27
