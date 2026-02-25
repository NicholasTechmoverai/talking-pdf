# Load a pretrained NLP model

from transformers import pipeline

summarizer = pipeline("summarization")

summary = summarizer(chunks[0], max_length=150)

print(summary)