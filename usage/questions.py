import numpy as np

def ask_question(question, chunks, embeddings):
    question_embedding = model.encode(question)

    scores = np.dot(embeddings, question_embedding)

    best_index = np.argmax(scores)

    return chunks[best_index]


answer = ask_question(
    "What skills does this CV have?",
    chunks,
    embeddings
)

print(answer)