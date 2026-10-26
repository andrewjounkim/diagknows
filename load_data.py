"""Load Symptom2Disease.csv into the DiagKnows ChromaDB collection.

Each CSV row (one patient's symptom description) becomes one item:
the "text" column is the document, and the "label" column is stored
as metadata "condition", along with "source": "symptom2disease".

Safe to re-run: IDs are stable ("s2d-0001" ... "s2d-1200", by row
position), and loading is skipped if the dataset is already loaded.
main.py calls load_dataset() automatically on startup.
"""

import csv
import sys
import time
from pathlib import Path

import chromadb

STORE_PATH = "./chroma_store"
COLLECTION_NAME = "symptoms"
CSV_PATH = Path("./data/Symptom2Disease.csv")
SOURCE = "symptom2disease"
FIRST_CHUNK = 50
BATCH_SIZE = 100
MODEL_CACHE = Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"


def read_rows(path: Path = CSV_PATH) -> list[dict]:
    if not path.exists():
        sys.exit(f"error: {path} not found. Download Symptom2Disease.csv from Kaggle into ./data/")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"label", "text"} - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"error: {path} is missing expected column(s): {sorted(missing)}")
        # The unnamed first column restarts at 0 every 300 rows, so it
        # isn't unique; IDs come from row position instead.
        return [
            {"text": row["text"].strip(), "label": row["label"].strip()}
            for row in reader
            if row["text"].strip()
        ]


def add_batch(collection, rows: list[dict], start: int) -> None:
    collection.upsert(
        ids=[f"s2d-{start + i + 1:04d}" for i in range(len(rows))],
        documents=[row["text"] for row in rows],
        metadatas=[{"condition": row["label"], "source": SOURCE} for row in rows],
    )


def count_loaded(collection) -> int:
    # Count only dataset rows; user-added cases share the collection.
    return len(collection.get(where={"source": SOURCE}, include=[])["ids"])


def load_dataset(collection) -> None:
    rows = read_rows()
    total = len(rows)

    existing = count_loaded(collection)
    if existing >= total:
        print(f"Dataset already loaded ({existing} rows in '{collection.name}'); skipping.")
        return
    if existing:
        # A previous run was interrupted. Upserting with stable IDs fills the gaps without duplicates.
        print(f"Dataset partially loaded ({existing}/{total} rows); resuming.")

    if not MODEL_CACHE.exists():
        print("Note: the first run downloads Chroma's embedding model (all-MiniLM-L6-v2, ~80 MB).")
        print("      It's cached in ~/.cache/chroma/ afterwards, so later runs skip this.\n")

    print(f"Loading {total} rows from {CSV_PATH} into '{collection.name}'...")

    t0 = time.perf_counter()
    add_batch(collection, rows[:FIRST_CHUNK], start=0)
    print(f"  first {FIRST_CHUNK} rows: {time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    for start in range(FIRST_CHUNK, total, BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        add_batch(collection, batch, start=start)
        done = start + len(batch)
        print(f"  {done:>5}/{total} ({done / total:.0%})  {time.perf_counter() - t1:.1f}s")

    print(f"Done. '{collection.name}' now has {collection.count()} items "
          f"({time.perf_counter() - t0:.1f}s total).")


def main() -> None:
    client = chromadb.PersistentClient(path=STORE_PATH)
    load_dataset(client.get_or_create_collection(name=COLLECTION_NAME))


if __name__ == "__main__":
    main()
