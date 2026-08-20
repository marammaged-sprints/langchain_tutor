from langchain_tutor.retrieval.vector_store import get_vector_store

vector_store = get_vector_store()

print("Chroma vector store created successfully.")
print("Collection:", vector_store._collection.name)
print("Number of documents:", vector_store._collection.count())