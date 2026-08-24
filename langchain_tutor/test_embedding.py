from retrieval.vector_store import get_embeddings

embeddings = get_embeddings()

result = embeddings.embed_query("What is a Python variable?")

print("Embedding created successfully.")
print("Number of values:", len(result))
print("First 5 values:", result[:5])