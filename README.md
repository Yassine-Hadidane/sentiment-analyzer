# Sentiment Analyzer
A simple NLP project that analyzes whether sentences are positive or negative using a pre-trained model.

## About
This was my first NLP project. Instead of building a model from scratch, 
I used a pre-trained model from "HuggingFace" which was already trained on millions of labeled sentences. 
The pipeline handles tokenization, runs the model, and returns a confidence score for each sentence.

## Tech Stack
- Python
- HuggingFace Transformers
- Matplotlib

## How to run

1. Install dependencies:
   pip install transformers torch matplotlib

2. Run the project:
   python sentiment.py

3. Enter sentences one by one, type "done" when finished.

## Output
![Chart](chart.png)

## What I learned
- How NLP pipelines work
- What tokenization means
- How to use pre-trained models with HuggingFace
- Data visualization with Matplotlib