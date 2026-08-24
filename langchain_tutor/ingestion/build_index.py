from __future__ import annotations
import shutil  #used to delete the existing Chroma database.
from pathlib import Path


from langchain_tutor.config import settings #imports the settings module from the confiq package, which contains configuration variables for the application.
from langchain_tutor.ingestion.load_book import load_book # reads pdf
from langchain_tutor.ingestion.split_documents import split_documents # split pages into smaller chunks
from langchain_tutor.retrieval.vector_store import get_vector_store #the function that creates the Chroma database



def index_exists(persist_directory: Path) -> bool:  #helper function that checks if a Chroma index already exists in the specified directory. 
    persist_directory = Path(persist_directory)
    if not persist_directory.exists():  #if there is no chroma/ no database
        return False
    return any(persist_directory.iterdir())  # Shows everything inside this folder.



"""Should I use the existing database, or should I build a new one?"""
def build_or_load_index(force_rebuild: bool= False):     # force rebuild ---> "Ignore the old database and build a new one."
    persist_dir= Path( settings.persist_directory) # location of the chroma database 
    if force_rebuild and persist_dir.exists():  #if the user wants to rebuild the database and the database already exists, delete it.
        shutil.rmtree(persist_dir)  #delete the existing Chroma database.

    pages = load_book(
        settings.book_path,
        source_name=settings.book_title,
    )

    # Split the PDF into chunks.
    chunks = split_documents(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    expected_chunk_count = len(chunks)    #if you have 612 chunks, we know the completed database should contain 612.

    if not force_rebuild and index_exists(persist_dir): #if the user doesn't want to rebuild the database and the database already exists, load it.
       vs = get_vector_store(persist_directory=persist_dir) # open chroma database
       try:
        chunk_count = vs._collection.count() # counts how many chunks are in the database 
       except Exception:
          chunk_count = None
       if chunk_count == expected_chunk_count: # not accepting non-empty Chroma database as complete.
             print("Existing complete index found.")
             return vs, chunk_count
             
       print(
                f"Incomplete index found: "
                f"{chunk_count} / {expected_chunk_count} chunks."
            )

       shutil.rmtree(persist_dir)
    
    persist_dir.mkdir(parents=True, exist_ok=True)  #creates a folder for the chroma database if it doesn't already exist.  / parents=True ---> create any necessary parent directories. / exist_ok=True ---> don't raise an error if the directory already exists.
    vs = get_vector_store(persist_directory = persist_dir)  #opens chroma
    print("Building Chroma index...")

    vs.add_documents(chunks)


    indexed_count = vs._collection.count()

    if indexed_count != expected_chunk_count:
        raise RuntimeError(
            f"Indexing incomplete: expected "
            f"{expected_chunk_count} chunks, "
            f"but Chroma contains {indexed_count}."
            )

    print(f"Indexed chunks: {indexed_count}")

    return vs, indexed_count

if __name__ == "__main__":
    build_or_load_index()