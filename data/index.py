from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

embedding = model.encode("Malaria causes fever")
print(embedding.shape)