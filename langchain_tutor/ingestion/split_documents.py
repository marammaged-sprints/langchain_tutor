from __future__ import annotations

import hashlib  #digital fingerprint of the content, used to create a unique chunk_id for each chunk of text
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter  #imports the tool that splits large pieces of text into smaller chunks.


def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter( #sets up the text splitter with the specified chunk size, overlap, and separators for splitting the text.
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)  #Split the documents into smaller chunks.

    out: List[Document] = []   #will store updated chunks with new metadata (chunk_id and chunk_index) for each chunk of text.
    for i, chunk in enumerate(chunks): #gives 2 values for each chunk: the index (i) and the chunk itself (chunk). This is useful for assigning a unique index to each chunk.
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", -1)
        digest = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()[:8]  #leaves the first 8 characters of the SHA-1 hash of the chunk's content. This is used to create a unique identifier for the chunk.
        chunk_id = f"{source}-p{page}-{digest}"    # e.g. think_python-p25-2a7c9f4b
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["chunk_index"] = i
        out.append(chunk)
    return out
