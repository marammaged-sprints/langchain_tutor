from __future__ import annotations

from pathlib import Path   #to work with folders and file paths
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_tutor.config import settings  # project settings


def get_embeddings():
    api_key = settings.require_api_key()

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=api_key,
    )

def get_vector_store(persist_directory: Path | None=None):
    from langchain_chroma import Chroma  #just making Chroma available

    persist_directory = Path(persist_directory or settings.persist_directory)    #persist_directory is the folder where Chroma saves the database./ if thr user provided a folder use it and if not use the default folder from config.py
    persist_directory.mkdir(parents=True, exist_ok=True)  #Create the folder if it doesn't exist.

    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_directory),
        collection_metadata={"hnsw:space": "cosine"},
        )


"""get_vector_store() does not add any data to the database.

It only creates or opens the database.

The chunks are added later in build_index.py"""
