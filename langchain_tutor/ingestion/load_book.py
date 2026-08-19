
from __future__ import annotations #features from newer versions of Python, even if the code is run in an older version

from pathlib import Path
from typing import List

from langchain_core.documents import Document  # document--> page content and metadata(page number)


def load_book(book_path: Path, source_name: str | None = None) -> List[Document]: 
    
    book_path = Path(book_path)
    if not book_path.exists():
        raise FileNotFoundError(
            f"Book PDF not found at '{book_path}'."
        )

    from langchain_community.document_loaders import PyPDFLoader  #opens PDF files and extracts text from them --->  Convert each page into a LangChain Document.

    loader = PyPDFLoader(str(book_path))
    raw_docs = loader.load() #opens the PDF file and extracts text from it, returning a list of Document objects, one for each page.

    display_name = source_name or book_path.stem  #Even if source_name isn't provided, the chatbot still knows which book the page came from.

    docs: List[Document] = []

    for d in raw_docs:   # Each d represents one page.
        page_0indexed = d.metadata.get("page", 0)  # if page exists use it otherwiae use 0 as default value. This is a safeguard in case the loader doesn't provide page numbers.

        docs.append(   # adds something to the list
            Document(
                page_content=d.page_content,
                metadata={
                    "source": display_name,
                    "page": page_0indexed + 1,
                    "file_path": str(book_path),
                },
            )
        )

    return docs
