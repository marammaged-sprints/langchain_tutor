from config import settings

print("Book:", settings.book_path)
print("Book exists:", settings.book_path.exists())

print("Embedding model:", settings.embedding_model)

print("API key loaded:", bool(settings.google_api_key))

print("Chroma directory:", settings.persist_directory)

print("Collection:", settings.collection_name)