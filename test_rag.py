"""
test_rag.py - Self-test script for the minimal RAG system.

Automatically runs ingestion, executes a sample query, and verifies:
  - The FAISS index was created on disk.
  - Exactly TOP_K chunks are retrieved.
  - The generated answer is non-empty.

Prints "ALL TESTS PASSED" on success, or a detailed failure message.
"""

import os
import sys
import json
import subprocess
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


INDEX_PATH = "index.faiss"
METADATA_PATH = "metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
TEST_QUESTION = "What is Retrieval-Augmented Generation?"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _fail(reason: str) -> None:
    """Print a failure message and exit with a non-zero status code."""
    print(f"\n[FAIL] {reason}")
    sys.exit(1)


def _pass(message: str) -> None:
    """Print a passing assertion message."""
    print(f"  [PASS] {message}")


# --------------------------------------------------------------------------- #
# Step 1 – Run ingestion                                                        #
# --------------------------------------------------------------------------- #

def run_ingestion() -> None:
    """
    Invoke ingest.py as a subprocess to simulate a real pipeline run.
    Verifies that the subprocess exits cleanly (return code 0).
    """
    print("\n[Test 1] Running ingestion pipeline (python ingest.py)...")
    result = subprocess.run(
        [sys.executable, "ingest.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _fail(
            f"Ingestion failed with return code {result.returncode}.\n"
            f"  STDOUT: {result.stdout.strip()}\n"
            f"  STDERR: {result.stderr.strip()}"
        )
    _pass("Ingestion completed successfully (exit code 0).")


# --------------------------------------------------------------------------- #
# Step 2 – Verify index and metadata exist on disk                             #
# --------------------------------------------------------------------------- #

def verify_artifacts_exist() -> None:
    """Check that both index.faiss and metadata.json were created."""
    print("\n[Test 2] Verifying artifacts exist on disk...")

    if not os.path.isfile(INDEX_PATH):
        _fail(f"FAISS index file '{INDEX_PATH}' was not created by ingestion.")
    _pass(f"'{INDEX_PATH}' exists ({os.path.getsize(INDEX_PATH):,} bytes).")

    if not os.path.isfile(METADATA_PATH):
        _fail(f"Metadata file '{METADATA_PATH}' was not created by ingestion.")
    _pass(f"'{METADATA_PATH}' exists ({os.path.getsize(METADATA_PATH):,} bytes).")


# --------------------------------------------------------------------------- #
# Step 3 – Verify index is loadable and non-empty                              #
# --------------------------------------------------------------------------- #

def verify_index_loadable() -> faiss.IndexFlatL2:
    """Load the FAISS index and confirm it contains vectors."""
    print("\n[Test 3] Loading and validating FAISS index...")

    try:
        index = faiss.read_index(INDEX_PATH)
    except Exception as e:
        _fail(f"Could not load FAISS index: {e}")

    if index.ntotal == 0:
        _fail("FAISS index is empty — no vectors were stored during ingestion.")
    _pass(f"FAISS index loaded: {index.ntotal} vectors, dimension={index.d}.")
    return index


# --------------------------------------------------------------------------- #
# Step 4 – Verify metadata is loadable and consistent                          #
# --------------------------------------------------------------------------- #

def verify_metadata_loadable(index: faiss.IndexFlatL2) -> list[dict]:
    """Load metadata and check it is consistent with the FAISS index."""
    print("\n[Test 4] Loading and validating metadata...")

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        _fail(f"Could not load metadata JSON: {e}")

    if not isinstance(metadata, list) or len(metadata) == 0:
        _fail("Metadata is empty or not a list.")

    if len(metadata) != index.ntotal:
        _fail(
            f"Metadata length ({len(metadata)}) does not match "
            f"FAISS index size ({index.ntotal})."
        )

    required_keys = {"chunk_id", "text", "source"}
    for i, chunk in enumerate(metadata):
        missing = required_keys - chunk.keys()
        if missing:
            _fail(f"Chunk #{i} is missing required keys: {missing}")

    _pass(f"Metadata loaded: {len(metadata)} chunks, all keys valid.")
    return metadata


# --------------------------------------------------------------------------- #
# Step 5 – Retrieve TOP_K chunks for the test question                         #
# --------------------------------------------------------------------------- #

def verify_retrieval(
    index: faiss.IndexFlatL2, metadata: list[dict]
) -> list[dict]:
    """Embed the test question and verify that TOP_K chunks are retrieved."""
    print(f"\n[Test 5] Retrieving top-{TOP_K} chunks for: '{TEST_QUESTION}'...")

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([TEST_QUESTION], convert_to_numpy=True).astype(np.float32)

    actual_k = min(TOP_K, index.ntotal)
    distances, indices = index.search(query_embedding, actual_k)

    chunks = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            _fail(f"FAISS returned an invalid index (-1) at rank {rank}.")
        chunks.append(metadata[idx])

    if len(chunks) != TOP_K:
        _fail(
            f"Expected {TOP_K} chunks to be retrieved, but got {len(chunks)}. "
            "Check that enough documents were ingested."
        )

    _pass(f"{len(chunks)} chunks retrieved as expected.")

    for i, chunk in enumerate(chunks, start=1):
        print(f"    Chunk {i}: source='{chunk['source']}', "
              f"text_preview='{chunk['text'][:80].strip()}...'")

    return chunks


# --------------------------------------------------------------------------- #
# Step 6 – Verify answer is non-empty (LLM call is optional)                   #
# --------------------------------------------------------------------------- #

def verify_answer(chunks: list[dict]) -> None:
    """
    Verify that a non-empty answer can be produced from the retrieved context.

    If OPENAI_API_KEY is set, call the live API and check the response.
    If not set, skip the live call and validate that the context is non-empty,
    which is sufficient to confirm the retrieval pipeline is healthy.
    """
    print("\n[Test 6] Verifying answer generation...")

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if not api_key:
        print(
            "  [SKIP] OPENAI_API_KEY not set. Skipping live LLM call.\n"
            "         Context retrieval validated — pipeline is healthy."
        )
        combined_context = " ".join(c["text"] for c in chunks)
        if not combined_context.strip():
            _fail("Retrieved chunks contain no text content.")
        _pass("Retrieved context is non-empty (LLM step skipped — no API key).")
        return

    # Live LLM call
    from openai import OpenAI, AuthenticationError, APIConnectionError, APIStatusError

    context_parts = [
        f"[Chunk {i} | Source: {c['source']}]\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    ]
    context = "\n\n".join(context_parts)
    prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "information provided in the context below. Do not use any outside knowledge. "
        "If the answer cannot be found in the context, respond with exactly: "
        "\"I don't know.\"\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {TEST_QUESTION}\n\n"
        "ANSWER:"
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )
        answer = (response.choices[0].message.content or "").strip()
    except AuthenticationError:
        _fail("OpenAI authentication failed. Check your OPENAI_API_KEY.")
    except APIConnectionError:
        _fail("Could not connect to OpenAI API. Check your internet connection.")
    except APIStatusError as e:
        _fail(f"OpenAI API error: {e.status_code} - {e.message}")

    if not answer:
        _fail("LLM returned an empty answer.")

    if answer.lower() == "i don't know.":
        _fail(
            "LLM returned 'I don't know.' for the test question. "
            "The retrieved context may not contain relevant information."
        )

    _pass(f"Non-empty answer received ({len(answer)} chars).")
    print(f"\n  Answer preview: {answer[:300]}{'...' if len(answer) > 300 else ''}")


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    print("=" * 60)
    print("RAG Self-Test Suite")
    print("=" * 60)

    run_ingestion()
    verify_artifacts_exist()
    index = verify_index_loadable()
    metadata = verify_metadata_loadable(index)
    chunks = verify_retrieval(index, metadata)
    verify_answer(chunks)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
