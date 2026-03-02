"""
ingest.py - Document ingestion pipeline for the minimal RAG system.

Loads .txt files from docs/, chunks them, generates embeddings using
sentence-transformers, and stores the FAISS index and metadata to disk.
"""

import os
import json
import hashlib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DOCS_DIR = "docs"
INDEX_PATH = "index.faiss"
METADATA_PATH = "metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a long text into overlapping chunks of fixed character length.

    Args:
        text: The input text to chunk.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start += chunk_size - overlap

    return chunks


def load_documents(docs_dir: str) -> list[dict]:
    """
    Load all .txt files from the given directory.

    Args:
        docs_dir: Path to the directory containing .txt documents.

    Returns:
        A list of dicts with keys 'text' and 'source'.

    Raises:
        FileNotFoundError: If the docs directory does not exist.
        RuntimeError: If no .txt files are found in the directory.
    """
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Docs directory not found: '{docs_dir}'")

    txt_files = [f for f in os.listdir(docs_dir) if f.endswith(".txt")]
    if not txt_files:
        raise RuntimeError(f"No .txt files found in '{docs_dir}'. Add documents before ingesting.")

    documents = []
    for filename in sorted(txt_files):
        filepath = os.path.join(docs_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            print(f"  [WARN] Encoding error reading '{filename}'. Retrying with latin-1 fallback.")
            try:
                with open(filepath, "r", encoding="latin-1") as f:
                    text = f.read()
            except Exception as e:
                print(f"  [ERROR] Could not read '{filename}': {e}. Skipping.")
                continue
        except Exception as e:
            print(f"  [ERROR] Unexpected error reading '{filename}': {e}. Skipping.")
            continue

        text = text.strip()
        if not text:
            print(f"  [WARN] '{filename}' is empty. Skipping.")
            continue

        documents.append({"text": text, "source": filename})
        print(f"  Loaded: {filename} ({len(text)} chars)")

    if not documents:
        raise RuntimeError("All documents were empty or unreadable. Nothing to ingest.")

    return documents


def build_chunks(documents: list[dict]) -> list[dict]:
    """
    Chunk all documents and deduplicate chunks by content hash.

    Args:
        documents: List of dicts with 'text' and 'source' keys.

    Returns:
        A list of unique chunk dicts with 'text', 'source', and 'chunk_id' keys.
    """
    all_chunks = []
    seen_hashes = set()

    for doc in documents:
        chunks = chunk_text(doc["text"])
        print(f"  Chunked '{doc['source']}' into {len(chunks)} chunks.")
        for i, chunk in enumerate(chunks):
            content_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                print(f"    [WARN] Duplicate chunk #{i} from '{doc['source']}'. Skipping.")
                continue
            seen_hashes.add(content_hash)
            all_chunks.append({
                "chunk_id": len(all_chunks),
                "text": chunk,
                "source": doc["source"],
            })

    return all_chunks


def generate_embeddings(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """
    Generate sentence embeddings for all text chunks.

    Args:
        chunks: List of chunk dicts containing 'text'.
        model: A loaded SentenceTransformer model.

    Returns:
        A numpy array of shape (num_chunks, embedding_dim) with float32 dtype.
    """
    texts = [chunk["text"] for chunk in chunks]
    print(f"\nGenerating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a FAISS IndexFlatL2 index from the given embeddings.

    Args:
        embeddings: Float32 numpy array of shape (num_chunks, dim).

    Returns:
        A FAISS index populated with all embeddings.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"  FAISS index built: {index.ntotal} vectors, dimension={dim}")
    return index


def save_artifacts(index: faiss.IndexFlatL2, chunks: list[dict]) -> None:
    """
    Persist the FAISS index and metadata JSON to disk.

    Args:
        index: The populated FAISS index.
        chunks: The list of chunk dicts to serialize as metadata.
    """
    faiss.write_index(index, INDEX_PATH)
    print(f"  Saved FAISS index to '{INDEX_PATH}'")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  Saved metadata to '{METADATA_PATH}'")


def main() -> None:
    print("=" * 60)
    print("RAG Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load documents
    print("\n[1/4] Loading documents...")
    documents = load_documents(DOCS_DIR)
    print(f"  Loaded {len(documents)} document(s).")

    # Step 2: Chunk documents
    print("\n[2/4] Chunking documents...")
    chunks = build_chunks(documents)
    if not chunks:
        raise RuntimeError("No chunks were produced. Check your documents and chunk settings.")
    print(f"  Total unique chunks: {len(chunks)}")

    # Step 3: Load model and generate embeddings
    print(f"\n[3/4] Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = generate_embeddings(chunks, model)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Step 4: Build FAISS index and save
    print("\n[4/4] Building FAISS index and saving artifacts...")
    index = build_faiss_index(embeddings)
    save_artifacts(index, chunks)

    print("\n" + "=" * 60)
    print("Ingestion complete.")
    print(f"  Documents : {len(documents)}")
    print(f"  Chunks    : {len(chunks)}")
    print(f"  Index     : {INDEX_PATH}")
    print(f"  Metadata  : {METADATA_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
