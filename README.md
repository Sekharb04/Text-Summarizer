# AI Text Summarization System

A full-stack NLP application that generates concise and coherent summaries from long-form text using both **Extractive (TextRank)** and **Abstractive (T5 Transformer)** techniques.

The system enables model comparison, confidence interpretation, customizable summary length, and PDF export functionality within a clean web interface.

---

## Overview

Reading long documents is time-consuming and inefficient. This project automates document understanding by providing:

- Extractive summarization (sentence selection)
- Abstractive summarization (neural text generation)
- Confidence scoring with visual indicators
- User-controlled summary length
- Exportable PDF reports

The project is designed to be **explorative**, allowing users to compare summarization approaches and understand trade-offs between abstraction and factual preservation.

---

## Key Features

### 1. Abstractive Summarization (T5)

- Transformer-based model
- Rewrites content into human-like summaries
- Supports Short / Medium / Long output lengths

### 2. Extractive Summarization (TextRank)

- Graph-based ranking algorithm
- Selects most relevant original sentences
- Preserves exact wording

### 3. Model Comparison

- Side-by-side summary outputs
- Confidence percentage for each model
- Threshold-based trust labels:
  - High
  - Medium
  - Low

### 4. Confidence Visualization

- Cosine similarity-based lexical alignment
- Visual confidence bar
- Helps interpret abstraction vs extraction trade-offs

### 5. PDF Export

- Downloadable summaries
- Separate export per model
- Timestamped reports

### 6. Clean Web Interface

- Black minimal UI
- Centered readable layout
- Justified summary output
- Responsive input controls

---
