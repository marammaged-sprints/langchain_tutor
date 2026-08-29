from langchain_core.documents import Document

from scripts.check_chunks import find_suspicious_chunks


def test_find_suspicious_chunks_reports_possible_split_definition():
    chunks = [
        Document(page_content="Ordinary prose."),
        Document(page_content="Example:\ndef add(a,"),
        Document(page_content="def complete(a, b):"),
    ]

    suspicious = find_suspicious_chunks(chunks)

    assert [(index, chunk.page_content) for index, chunk in suspicious] == [
        (1, "Example:\ndef add(a,")
    ]
