from langchain_tutor.ingestion.build_index import build_or_load_index
vector_store, count = build_or_load_index()

print("Indexed chunks:", count)