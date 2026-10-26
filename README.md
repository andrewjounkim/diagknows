# DiagKnows

A command-line symptom explorer built on a vector database.

> **DiagKnows is an educational tool, not a medical diagnosis.** Always consult a qualified healthcare professional. In an emergency, call 911 or your local emergency number.

<!--
  The five sections below are written by the author. Everything under
  "AI-generated documentation" further down was drafted with Claude Code
  and checked against the code.
-->

## What it does and why

> DiagKnows is a command-line app where you describe how you feel in plain English, and it tells you which conditions your description most resembles, which kind of doctor to see, and shows a real patient description that's similar to yours. I chose it because people describe the same symptom in endless different ways ("peeing fire" vs. "burning when I urinate"), so a keyword search would miss most matches. Searching by meaning is exactly what a vector database is for.

## What's stored in the database

> The data comes from the Symptom2Disease dataset on Kaggle: 1,200 short, first-person descriptions of symptoms, each labeled with one of 24 conditions (50 per condition). Each description is one item. I didn't need to split anything, because every description is between 14 and 78 tokens, well under the embedding model's 256-token limit. Each item stores the description as the document (that's the text that gets embedded and searched), plus two metadata fields: condition (the label) and source ("symptom2disease" for the dataset, "user" for cases added in the app). Dataset items have IDs like s2d-0001; user cases get a random uuid-based ID so they can never collide. There's also a second, small collection with 20 hand-written emergency descriptions, used only to check whether a query sounds like an emergency.

## How to add and search

> To search, choose option 1 and type how you feel in plain English. The app finds the 10 most similar patient descriptions and groups them by condition. Each match counts as a vote for its condition, and closer matches count more. It then shows the top conditions, each with a match strength (Strong / Possible / Weak), the kind of doctor to see, and one similar description. If nothing is close enough, it says "No confident match" instead of guessing. If the description sounds like an emergency, it tells you to seek emergency care instead of ranking anything. To add a case, choose option 2, type a description and a condition name (an existing one or a new one), and it's searchable immediately. Option 3 shows how many items are stored, the count per condition, and how many were added by users.

## Two queries a keyword search would miss

> Query 1: breakouts all over my forehead and chin, whiteheads everywhere
DiagKnows returns Acne as a strong match. One of the descriptions it retrieves is s2d-0598: "When I woke up this morning, I saw a terrible rash on my skin. Blackheads and pus-filled pimples are all over the place. Additionally, my skin has been changing a lot." A keyword search would miss this, because the item never says "breakouts", "forehead", "chin" or "whiteheads". It says "blackheads" and "pimples" instead, and a simple keyword search over the dataset actually ranked Hypertension first.

> Query 2: I look sallow and yellowish, my wee is dark like cola, and my belly aches
DiagKnows returns Jaundice (its only match). The top retrieved item is s2d-0813: "I've had a scratchy stomach and been throwing up. Aside from losing weight, I am also really exhausted. I have a high fever and yellow skin. Besides having stomach ache, my pee is black." A keyword search would miss it, because the query says "sallow", "yellowish", "wee", "cola" and "belly" while the item says "yellow skin", "pee", "black" and "stomach". They mean the same thing with different words, and the keyword search ranked Pneumonia first.

## How I used AI

> I used Claude Code in VS Code (model: Claude Opus 5.5) to build the project step by step. I decided the idea, the stack (Python + ChromaDB), what counts as an item and its metadata, the ranking approach, the emergency check, the doctor suggestions, and that every threshold had to come from real distances instead of guesses. Claude wrote most of the code, ran the tests, and drafted the technical README sections at the bottom. Things it got wrong that were caught along the way:

> The first ranking let several far-away matches outvote one very close match. My test case ranked 3rd until the weighting was fixed.
> It labeled a weak match "high confidence".
> It made a folder that conflicted with the main module.
> It guessed the dataset author's full name from a Kaggle username.
> I also ran the app myself, found the output hard to read, and had it redesigned. The full prompt history is in prompt_log.md.

> Sources: Symptom2Disease dataset (Kaggle, uploaded by niyarrbarman); ChromaDB; the all-MiniLM-L6-v2 embedding model (Sentence Transformers).

---

## AI-generated documentation

*The sections below were drafted by Claude (via Claude Code) from the code in this repo and reviewed by the author.*

### Requirements

- Python 3 (developed and tested on Python 3.14.3, macOS)
- Python packages, pinned in `requirements.txt`:
  - `chromadb==1.5.9`: the vector database
  - `python-dotenv==1.2.4`: reads the optional `.env` file
  - `openai==3.38.0`: only used for the optional AI explanations

### Installation

```bash
git clone <this-repo-url> DiagKnows
cd DiagKnows
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Getting the dataset

DiagKnows uses the **Symptom2Disease** dataset from Kaggle: https://www.kaggle.com/datasets/niyarrbarman/symptom2disease

The dataset isn't included in this repo (`data/` is gitignored), so you need to download it. Either:

**Option A, in the browser:** open the Kaggle page, click **Download**, unzip the archive, create a `data/` folder in the project root, and put `Symptom2Disease.csv` in it.

**Option B, from the command line:**

```bash
mkdir -p data
curl -L -o /tmp/symptom2disease.zip \
  https://www.kaggle.com/api/v1/datasets/download/niyarrbarman/symptom2disease
unzip -o /tmp/symptom2disease.zip Symptom2Disease.csv -d data/
```

The file must end up at exactly `data/Symptom2Disease.csv`. The loader expects these columns:

| Column | Contents |
|---|---|
| *(unnamed)* | Row index. It restarts at 0 every 300 rows, so it isn't used. |
| `label` | The condition name |
| `text` | One patient's symptom description, in plain English |

The file has 1,200 rows covering 24 conditions, 50 descriptions each. If the file is missing or lacks the `label` or `text` column, the loader exits with an error saying so.

### Running

```bash
python main.py
```

This opens a menu:

```
[1] Check symptoms   [2] Add a case   [3] Stats   [4] Quit
```

Results are shown in plain language: each condition gets a match strength (Strong / Possible / Weak), the kind of doctor to see, and one similar case from the database. To also see the underlying numbers (neighbor counts, score share and distances), run:

```bash
python main.py --details
```

Other scripts in the repo:

| Command | What it does |
|---|---|
| `python load_data.py` | Loads the dataset into the database without starting the app (`main.py` does this automatically). |
| `python explore.py` | Runs three fixed test queries and prints Chroma's full raw query results plus a top-10 condition/distance table. Run the loader or `main.py` first. |
| `python smoke_test.py` | Setup check: writes 10 sample items to a separate `smoke_test_symptoms` collection and prints the count before and after, to confirm data persists between runs. |

### How the database is built on first run

On startup, `main.py` calls `ensure_database()` in `diagknows.py`, which:

1. **Loads the dataset into the `symptoms` collection** if it isn't fully loaded yet (via `load_dataset()` in `load_data.py`):
   - It counts only items with `source: "symptom2disease"`, so user-added cases don't affect the check. If all CSV rows are already present, loading is skipped.
   - Each CSV row becomes one item: the `text` is the document, and the metadata is `{"condition": <label>, "source": "symptom2disease"}`.
   - IDs are stable and based on row position (`s2d-0001` … `s2d-1200`). Items are written with `upsert`, so an interrupted load can simply be re-run without creating duplicates.
   - The first 50 rows are added and timed on their own, then the rest go in batches of 100 with progress printed. A full load took about 14 seconds in testing.
2. **Builds the `red_flags` collection**: a small set of hand-written emergency symptom descriptions defined in `RED_FLAGS` in `diagknows.py`. If the collection's size doesn't match the list, it's deleted and rebuilt from the list.

Cases added through the menu go into the same `symptoms` collection, with metadata `{"condition": <what the user typed>, "source": "user"}` and a uuid-based ID (`user-<uuid4>`). They're added with `add()`, not `upsert()`, after checking the ID isn't already used.

Everything is stored on disk in `./chroma_store/` (gitignored). To rebuild from scratch, delete that folder and run `python main.py` again. **Deleting it also deletes any cases you've added.**

### Vector database and embedding model

- **Vector database:** [ChromaDB](https://www.trychroma.com/) 1.5.9, running locally and embedded in the Python process via `chromadb.PersistentClient(path="./chroma_store")`. No server is needed.
- **Collections:** `symptoms` (the dataset plus user-added cases) and `red_flags` (the emergency check). If you've run `smoke_test.py`, there's also a `smoke_test_symptoms` collection, which the app doesn't use.
- **Embedding model:** Chroma's default embedding function, **all-MiniLM-L6-v2**, run locally through ONNX Runtime. It produces 384-dimensional, unit-length vectors. Text is never sent anywhere to be embedded.
- **Input length:** Chroma truncates input to this model at 256 tokens. Dataset descriptions are 14–78 tokens, so each row is stored whole as one item without splitting. Very long user-added descriptions would be cut off after 256 tokens.
- **Distance metric:** Chroma's default `l2` space, which reports *squared* Euclidean distance. Because the vectors are unit length, `distance = 2 − 2 × cosine similarity`: 0 means identical meaning, about 2 means unrelated. The app's cutoffs are constants at the top of `diagknows.py`:
  - `MATCH_CUTOFF = 1.0`: if the closest description is this far or farther, the app reports "no confident match" instead of guessing.
  - `EMERGENCY_CUTOFF = 0.7`: a description within this distance of any red-flag entry triggers the emergency message instead of a ranking.

#### What downloads on first run

The first time anything is embedded, Chroma downloads the all-MiniLM-L6-v2 model (**about 79 MB**) to `~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/` and shows a progress bar. The loader prints a note about this if the model isn't cached yet. Later runs use the cached copy and work offline, except for the optional AI explanations.

### Optional: AI explanations (OpenAI API key)

If an OpenAI API key is set, then after a successful ranking the app asks OpenAI for a 3–4 sentence plain-English explanation of why those conditions came up. The prompt tells the model to use only the text provided. **Without a key, the app works fully.** The explanation is simply left out, and the app prints a one-line notice at startup.

Setup:

```bash
cp .env.example .env
```

Then edit `.env`:

```
OPENAI_API_KEY=sk-...
# Optional; defaults to gpt-5.4-mini
OPENAI_MODEL=
```

`.env` is gitignored, so never commit your key. The key can also be set as a regular environment variable instead of in `.env`.

**What's sent to OpenAI:** only when a key is set and a query produces a ranking, the app sends your symptom description, up to 6 of the retrieved patient descriptions, and the names of the ranked conditions. Nothing is sent for emergency or "no confident match" results, or when no key is set. If the API call fails for any reason, the app prints `(AI explanation unavailable: <error type>)` and shows the results without the explanation.

### Known limitations

- **The emergency check is a safety net with gaps.** In testing, at the 0.7 cutoff it caught only 6 of 10 emergency descriptions it hadn't seen before. Among the misses were a sudden stroke description and a suicidal statement. Never rely on it.
- **Multi-symptom queries blur.** The whole description becomes one embedding, so a query that mixes unrelated symptoms (e.g. sneezing plus stabbing chest pain) tends to match only one of them.
- **Similar conditions get confused.** With 50 examples per condition, close neighbors like Common Cold and allergy, or Dengue and Malaria, often split the vote.
- **Dataset quirks.** Labels have inconsistent capitalization (the app shows them capitalized), 47 rows repeat another row's text exactly, and a few labels look wrong (e.g. `s2d-1200` is labelled diabetes but describes sneezing and a cough).

### Sources

- Dataset: *Symptom2Disease*, uploaded to Kaggle by user `niyarrbarman`. https://www.kaggle.com/datasets/niyarrbarman/symptom2disease
- Vector database: ChromaDB. https://www.trychroma.com/
- Embedding model: *all-MiniLM-L6-v2* (Sentence Transformers). https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
