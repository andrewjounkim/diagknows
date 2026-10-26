"""Smoke test: confirm a ChromaDB collection persists across process restarts.

Run it twice. The first run should report 0 records before writing;
the second run should report 10 before writing, proving the data survived.
"""

import chromadb

STORE_PATH = "./chroma_store"
COLLECTION_NAME = "smoke_test_symptoms"

SAMPLES = [
    ("Runny nose, sneezing, scratchy throat, mild tiredness for a couple of days.", "Common cold"),
    ("Sudden high fever, body aches, chills, and feeling completely wiped out.", "Influenza"),
    ("Throbbing pain on one side of my head, sensitive to light, feel nauseous.", "Migraine"),
    ("Burning feeling in my chest after meals, worse when I lie down.", "Acid reflux"),
    ("Itchy watery eyes and sneezing every spring when the pollen is out.", "Seasonal allergies"),
    ("Constantly thirsty, peeing a lot, and losing weight without trying.", "Type 2 diabetes"),
    ("Stomach cramps, diarrhea, and vomiting a few hours after eating out.", "Food poisoning"),
    ("Burning when I pee and feel like I need to go all the time.", "Urinary tract infection"),
    ("Tight band of pressure around my forehead after long days at the screen.", "Tension headache"),
    ("Racing heart, constant worry, can't sit still, trouble falling asleep.", "Anxiety"),
]


def main() -> None:
    client = chromadb.PersistentClient(path=STORE_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print(f"chromadb {chromadb.__version__}")
    print(f"count before write: {collection.count()}")

    # upsert with stable ids, so re-running doesn't create duplicates
    collection.upsert(
        ids=[f"sample-{i}" for i in range(len(SAMPLES))],
        documents=[text for text, _ in SAMPLES],
        metadatas=[{"condition": condition} for _, condition in SAMPLES],
    )

    print(f"count after write:  {collection.count()}")


if __name__ == "__main__":
    main()
