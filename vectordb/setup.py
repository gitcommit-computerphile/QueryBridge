import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "knowledge_base"
SAMPLE_DOCS_PATH = str(Path(__file__).parent.parent / "data" / "sample_docs.txt")


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def load_sample_documents() -> list[Document]:
    with open(SAMPLE_DOCS_PATH, "r", encoding="utf-8") as f:
        raw = f.read()

    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    docs = []
    for block in blocks:
        lines = block.split("\n", 1)
        title = lines[0].replace("Title: ", "").strip() if lines else "Unknown"
        content = lines[1].strip() if len(lines) > 1 else block
        docs.append(Document(page_content=content, metadata={"title": title}))
    return docs


def get_vectorstore() -> Chroma:
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    # seed with sample docs if collection is empty
    if vectorstore._collection.count() == 0:
        print("[VectorDB] Collection empty — loading sample documents...")
        docs = load_sample_documents()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        vectorstore.add_documents(chunks)
        print(f"[VectorDB] Loaded {len(chunks)} chunks into ChromaDB.")
    else:
        print(f"[VectorDB] Collection has {vectorstore._collection.count()} chunks.")

    return vectorstore


def add_documents_to_vectorstore(texts: list[str], vectorstore: Chroma) -> int:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=t) for t in texts]
    chunks = splitter.split_documents(docs)
    vectorstore.add_documents(chunks)
    return len(chunks)
