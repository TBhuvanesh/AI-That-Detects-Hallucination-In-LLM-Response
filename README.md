# Semantic Entropy LLM Hallucination Detector

An AI-powered framework for detecting hallucinations in Large Language Model (LLM) responses using **Semantic Entropy**, **Hybrid Retrieval**, and **Natural Language Inference (NLI)**.

This project focuses on improving trust and reliability in LLM-generated outputs by identifying contradictions, unsupported claims, and semantic inconsistencies.

---

## Overview

Large Language Models often generate fluent but factually incorrect content, known as **hallucinations**.

This project introduces a hybrid verification pipeline that:

* Retrieves relevant knowledge from external sources
* Measures semantic uncertainty
* Validates generated responses against retrieved evidence
* Detects contradictions and unsupported claims

The goal is to improve factual correctness and reliability in AI systems.

---

## Features

* Semantic entropy-based hallucination detection
* Hybrid retrieval (BM25 + Dense Retrieval)
* FAISS vector similarity search
* Wikipedia-based evidence retrieval
* NLI-based contradiction detection
* Confidence scoring mechanism
* Explainable rule-based verification
* Modular pipeline architecture

---

## Architecture

User Query
↓
LLM Response Generation
↓
Wikipedia Retrieval (Top 10 docs)
↓
Hybrid Retrieval (Best 3 docs)
↓
Sentence Embedding Generation
↓
FAISS Similarity Search
↓
NLI Verification (Entailment / Contradiction / Neutral)
↓
Semantic Entropy Scoring
↓
Final Hallucination Classification

---

## Tech Stack

### Backend

* Python

### NLP & AI

* SentenceTransformers
* HuggingFace Transformers
* BM25
* Natural Language Inference Models

### Retrieval

* FAISS
* Wikipedia API
* Dense Retrieval

### Utilities

* NumPy
* Pandas
* Scikit-learn

---

## Project Structure

```text
config/        → Configuration files
data/          → Sample datasets
models/        → Model loading scripts
pipeline/      → Main hallucination detection pipeline
utils/         → Utility functions
app.py         → Main application entry point
```

---

## Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd semantic-entropy-llm-hallucination-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

---

## Example Workflow

Input:

```text
What is the capital of Australia?
```

LLM Response:

```text
Sydney is the capital of Australia.
```

Retrieved Evidence:

```text
Canberra is the capital city of Australia.
```

Output:

```text
Hallucination Detected
Confidence Score: High
Reason: Contradiction found
```

---

## Research Contribution

This project is based on the research idea:

**Detecting Hallucinations in Large Language Models using Semantic Entropy**

Key contributions:

* Hybrid factual verification pipeline
* Semantic uncertainty measurement
* Retrieval-enhanced verification
* Interpretable hallucination scoring

---

## Future Improvements

* Multi-source retrieval (Web + Research Papers)
* Real-time API integration
* Multi-modal hallucination detection
* Advanced uncertainty calibration
* Dashboard visualization

---

## Applications

* AI chatbots
* Academic assistants
* Healthcare AI
* Legal AI systems
* Research assistants
* Content verification systems

---

## Author

**Bhuvanesh T**

B.Tech AIML | Full Stack Developer | AI Research Enthusiast

---

## License

This project is intended for research and educational purposes.
