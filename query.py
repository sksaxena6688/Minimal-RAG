"""
query.py - Query interface for the minimal RAG system.

Loads the FAISS index and metadata, embeds the user's question,
retrieves the top-k most relevant chunks, and generates a grounded
answer via the OpenAI chat API (temperature=0).
"""

import os
import sys
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI, AuthenticationError, APIConnectionError, APIStatusError


INDEX_PATH = "index.faiss"
METADATA_PATH = "metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0


def load_index(index_path: str) -> faiss.IndexFlatL2:
    """
    Load the FAISS index from disk.

    Args:
        index_path: Path to the .faiss file.

    Returns:
        The loaded FAISS index.

    Raises:
        FileNotFoundError: If the index file does not exist.
        RuntimeError: If the index is empty.
    """
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at '{index_path}'. "
            "Please run 'python ingest.py' first."
        )
    index = faiss.read_index(index_path)
    if index.ntotal == 0:
        raise RuntimeError("FAISS index is empty. Re-run ingestion with valid documents.")
    return index


def load_metadata(metadata_path: str) -> list[dict]:
    """
    Load chunk metadata from disk.

    Args:
        metadata_path: Path to the metadata JSON file.

    Returns:
        A list of chunk dicts with 'text', 'source', and 'chunk_id'.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
        ValueError: If the metadata is malformed or empty.
    """
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found at '{metadata_path}'. "
            "Please run 'python ingest.py' first."
        )
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, list) or len(metadata) == 0:
        raise ValueError("Metadata is empty or malformed. Re-run ingestion.")

    return metadata


def embed_query(question: str, model: SentenceTransformer) -> np.ndarray:
    """
    Embed a user query into a float32 numpy vector.

    Args:
        question: The user's question string.
        model: A loaded SentenceTransformer model.

    Returns:
        A 2D float32 numpy array of shape (1, embedding_dim).
    """
    embedding = model.encode([question], convert_to_numpy=True)
    return embedding.astype(np.float32)


def retrieve_chunks(
    query_embedding: np.ndarray,
    index: faiss.IndexFlatL2,
    metadata: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Search the FAISS index and return the top-k matching chunks.

    Args:
        query_embedding: Float32 array of shape (1, dim).
        index: The loaded FAISS index.
        metadata: List of all chunk dicts.
        top_k: Number of results to retrieve.

    Returns:
        A list of chunk dicts for the top-k nearest neighbours.

    Raises:
        RuntimeError: If no results are returned.
    """
    actual_k = min(top_k, index.ntotal)
    distances, indices = index.search(query_embedding, actual_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        chunk = metadata[idx].copy()
        chunk["distance"] = float(distances[0][rank])
        results.append(chunk)

    if not results:
        raise RuntimeError("No results returned from FAISS search.")

    return results


def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build the LLM prompt combining retrieved context with the question.

    Args:
        question: The user's question.
        chunks: Retrieved chunk dicts containing 'text'.

    Returns:
        A formatted prompt string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[Chunk {i} | Source: {chunk['source']}]\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "information provided in the context below. Do not use any outside knowledge. "
        "If the answer cannot be found in the context, respond with exactly: "
        "\"I don't know.\"\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )
    return prompt


def generate_answer(prompt: str, api_key: str) -> str:
    """
    Call the OpenAI chat API to generate an answer from the prompt.

    Args:
        prompt: The full prompt including context and question.
        api_key: A valid OpenAI API key.

    Returns:
        The model's answer as a string.

    Raises:
        AuthenticationError: If the API key is invalid.
        APIConnectionError: If the OpenAI API is unreachable.
        APIStatusError: For other API-side errors.
    """
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=512,
    )
    answer = response.choices[0].message.content
    return answer.strip() if answer else "I don't know."


def print_retrieved_chunks(chunks: list[dict]) -> None:
    """Pretty-print the retrieved chunks with source and distance info."""
    print("\n" + "=" * 60)
    print(f"Retrieved {len(chunks)} chunk(s):")
    print("=" * 60)
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n[Chunk {i}]")
        print(f"  Source   : {chunk['source']}")
        print(f"  Distance : {chunk['distance']:.4f}")
        print(f"  Text     : {chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}")
    print("=" * 60)


def main() -> None:
    # --- Validate API key ---
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print(
            "[ERROR] OPENAI_API_KEY environment variable is not set.\n"
            "  Set it with: set OPENAI_API_KEY=sk-...   (Windows)\n"
            "               export OPENAI_API_KEY=sk-... (Linux/macOS)"
        )
        sys.exit(1)

    # --- Load FAISS index and metadata ---
    print("Loading FAISS index and metadata...")
    try:
        index = load_index(INDEX_PATH)
        metadata = load_metadata(METADATA_PATH)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"  Index loaded: {index.ntotal} vectors")
    print(f"  Metadata loaded: {len(metadata)} chunks")

    # --- Load embedding model ---
    print(f"\nLoading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    # --- Query loop ---
    print("\nRAG Query Interface (type 'exit' or 'quit' to stop)\n")
    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            print("  Please enter a non-empty question.")
            continue

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        # Embed question
        query_embedding = embed_query(question, model)

        # Retrieve chunks
        try:
            chunks = retrieve_chunks(query_embedding, index, metadata, top_k=TOP_K)
        except RuntimeError as e:
            print(f"  [ERROR] Retrieval failed: {e}")
            continue

        # Print retrieved chunks
        print_retrieved_chunks(chunks)

        # Build prompt and generate answer
        prompt = build_prompt(question, chunks)
        print("\nGenerating answer...")
        try:
            answer = generate_answer(prompt, api_key)
        except AuthenticationError:
            print(
                "[ERROR] Invalid OpenAI API key. "
                "Check your OPENAI_API_KEY environment variable."
            )
            sys.exit(1)
        except APIConnectionError:
            print("[ERROR] Could not connect to OpenAI API. Check your internet connection.")
            continue
        except APIStatusError as e:
            print(f"[ERROR] OpenAI API error: {e.status_code} - {e.message}")
            continue

        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
