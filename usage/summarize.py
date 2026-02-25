from transformers import pipeline

summarizer = pipeline("summarization")

summary = summarizer(text[:1000])

print(summary)