from __future__ import annotations

from pathlib import Path   #to work with folders and file paths
from langchain_tutor.config import settings #the project settings e.g. settings.persist_directory


def get_embeddings():  #creates an embedding model
    from langchain_google_genai import GoogleGenerativeAIEmbeddings   

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,   #model name from config.py
        google_api_key=settings.google_api_key or None,
    )


def get_vector_store(persist_directory: Path | None=None):
    from langchain_chroma import Chroma  #just making Chroma available

    persist_directory = Path(persist_directory or settings.persist_directory)    #persist_directory is the folder where Chroma saves the database./ if thr user provided a folder use it and if not use the default folder from config.py
    persist_directory.mkdir(parents=True, exist_ok=True)  #Create the folder if it doesn't exist.

    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),  #calls the embedding model as iot needs this model to convert text into embeddings.
        persist_directory=str(persist_directory), #tells chroma to save everything in the data/chroma_db
    )





"""get_vector_store() does not add any data to the database.

It only creates or opens the database.

The chunks are added later in build_index.py"""