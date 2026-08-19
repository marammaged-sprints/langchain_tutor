from ingestion.build_index import build_or_load_index
from retrieval.retriever import retrieve


vector_store, count = build_or_load_index()

print("Indexed:", count)

results = retrieve(
    vector_store,
    "What is a variable in Python?"
)

for result in results:
    print("\n---")
    print("Chunk ID:", result.chunk_id)
    print("Source:", result.source)
    print("Page:", result.page)
    print("Score:", result.score)
    print("Content:", result.content[:300])