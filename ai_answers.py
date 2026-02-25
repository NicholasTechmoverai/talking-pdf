import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# embedding model (for search)
embed_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)

# LLM model (for answering and summarizing)
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_length=512
)

# read PDF
doc = fitz.open("cv_file.pdf")

text = ""
for page in doc:
    text += page.get_text()


# split into meaningful chunks
def split_text(text, max_chars=800):

    paragraphs = text.split("\n")

    chunks = []
    current = ""

    for p in paragraphs:

        if len(current) + len(p) < max_chars:
            current += "\n" + p
        else:
            chunks.append(current)
            current = p

    if current:
        chunks.append(current)

    return chunks


chunks = split_text(text)

# embed chunks
embeddings = embed_model.encode(chunks)


# retrieve relevant chunks
def retrieve(query, k=3):

    query_embedding = embed_model.encode([query])

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(scores)[-k:][::-1]

    return "\n".join([chunks[i] for i in top_indices])


# chatbot function using LLM
def ask(question):

    context = retrieve(question)

    prompt = f"""
    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    result = llm(prompt)

    return result[0]["generated_text"]


# summarization function
def summarize():

    prompt = f"""
    Provide a clear professional summary of this document:

    {text[:3000]}
    """

    result = llm(prompt)

    return result[0]["generated_text"]


# chat loop
while True:

    q = input("\nAsk (or type summary): ")

    if q == "exit":
        break

    if q == "summary":
        print("\nSummary:\n", summarize())
    else:
        print("\nAnswer:\n", ask(q))