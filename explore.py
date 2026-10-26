"""Run a few test queries against the symptoms collection and show the results.

For each query, prints the full raw object returned by collection.query,
then a compact table of condition + distance for the top results.

Run load_data.py first.
"""

import sys
from pprint import pprint

import chromadb

STORE_PATH = "./chroma_store"
COLLECTION_NAME = "symptoms"
N_RESULTS = 10

QUERIES = [
    "it feels like I'm peeing fire",
    "my eyes and skin have turned the color of a banana",
    "the stock market closed higher on Tuesday",  # nonsense control
]


def main() -> None:
    client = chromadb.PersistentClient(path=STORE_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        sys.exit(f"error: collection '{COLLECTION_NAME}' not found. Run load_data.py first.")

    space = collection.configuration["hnsw"]["space"]
    print(f"Collection '{COLLECTION_NAME}': {collection.count()} items, distance space: {space}")

    for query in QUERIES:
        results = collection.query(
            query_texts=[query],
            n_results=N_RESULTS,
            include=["documents", "metadatas", "distances"],
        )

        print("\n" + "=" * 80)
        print(f"QUERY: {query!r}")
        print("=" * 80)
        print("\n--- raw result ---")
        pprint(results, width=120)

        # Every field is a list with one inner list per query text; we sent one, so take [0].
        print(f"\n--- top {N_RESULTS} ---")
        print(f"{'rank':>4}  {'distance':>8}  {'id':<9}  condition")
        for rank, (id_, meta, dist) in enumerate(
            zip(results["ids"][0], results["metadatas"][0], results["distances"][0]), start=1
        ):
            print(f"{rank:>4}  {dist:>8.4f}  {id_:<9}  {meta['condition']}")


if __name__ == "__main__":
    main()
