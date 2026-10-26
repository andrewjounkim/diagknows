# Prompt Log: DiagKnows (HW6, Vector Database App)

## Tools and models

- **Claude Code** (VS Code extension), model **Claude Opus 5.5** (`claude-opus-5-5`): used for the whole build, from project setup through the README draft.

> **[PLACEHOLDER: author]** A sentence or two on which model/tool you used for which parts, and why you chose it. Also add any other tools you used (e.g. ChatGPT, docs, Ed, office hours) and any work done outside this session.

## How this log was made

The prompts below are copied **verbatim** from the Claude Code session, in order. Typos are left as typed. The short "What happened" notes under each prompt were drafted by Claude from the session and reviewed by me.

---

## 1. Project setup and smoke test

**Prompt:**

```
I'm building a Python command-line app called DiagKnows: a symptom explorer that takes a plain-English description of how someone feels and suggests which conditions it most resembles, using a vector database.

Stack: Python + ChromaDB, running locally. First, just set up the project:
- Create a folder structure and a requirements.txt
- Check which chromadb version is installed (pip show chromadb) and use the CURRENT API for that version. Use chromadb.PersistentClient(path="./chroma_store"). Do not use chromadb.Client(Settings(...)) or collection.persist(); those are old APIs.
- Write a tiny smoke_test.py that creates a persistent collection, adds ~10 hand-written symptom descriptions with a "condition" metadata field, and prints collection.count().

Don't build the app yet. I want to run the smoke test twice and confirm the count survives a restart.
```

**What happened:** chromadb wasn't installed, so Claude created a project virtualenv and installed it (version 1.5.9). The smoke test prints the count before and after writing. Run 1 showed 0 then 10; run 2 showed 10 before writing, so the data survived the restart. The first run also downloaded the ~79 MB embedding model.

## 2. Load the real dataset

**Prompt:**

```
Now load the real dataset. I downloaded Symptom2Disease.csv from Kaggle into ./data/. Open it and tell me the actual column names and row count before writing the loader. Don't assume them.

Write load_data.py that:
- Treats each row (one patient's symptom description) as one item: the description is the document, the condition label goes in metadata as "condition", plus "source": "symptom2disease"
- Uses stable unique IDs like "s2d-0001" so re-running doesn't create duplicates (check the collection first and skip loading if it's already populated)
- Adds in batches of ~100, not one call per row
- Loads the first 50 rows first and prints how long it took, then continues with the rest while printing progress
- Warns that the embedding model downloads on first run

Add data/ and chroma_store/ to .gitignore.
```

**What happened:** the CSV wasn't actually in `data/` yet, so Claude stopped instead of guessing the columns, and only updated `.gitignore`.

**Prompt:**

```
can you download symptom2disease from kaggle for me/
```

**What happened:** Claude downloaded it from Kaggle's public download URL (no API key needed) and inspected it.
- **Columns:** an unnamed index, `label` and `text`
- **Size:** 1,200 rows, 24 conditions × 50 rows each
- **Quirks:** the unnamed index restarts every 300 rows, so it can't be used as an ID; 47 rows repeat another row's text; and some labels look wrong.

Then it wrote `load_data.py`. The first 50 rows took 0.75s, the full load about 14s, and a second run skipped loading.

**Prompt (pasted again by mistake):** the same "Now load the real dataset…" prompt as above. Claude confirmed nothing had changed and re-ran the loader, which skipped as expected.

## 3. Look at raw results and pick a distance cutoff

**Prompt:**

```
Write explore.py that runs a few queries and prints the FULL raw result object from collection.query (ids, documents, metadatas, distances), then a cleaner view showing condition + distance for the top 10.

Test these queries:
1. "it feels like I'm peeing fire"
2. "my eyes and skin have turned the color of a banana"
3. a nonsense query: "the stock market closed higher on Tuesday"

Explain the shape of the results (why it's a list of lists) and what the distance numbers mean for Chroma's default distance metric. Then help me pick a distance cutoff that separates real matches from the nonsense query, based on the actual numbers.
```

**What happened:**
- **The metric:** Claude checked the collection's settings and found Chroma's default `l2` (squared Euclidean) with unit-length embeddings, so distance = 2 − 2·cosine.
- **Your three queries:**
  - "peeing fire": urinary tract infection at 0.69 ✓
  - **banana: failed.** Jaundice only appeared at rank 10, with a best distance of 1.13.
  - stock market: 1.64
- **Choosing the cutoff:** three points weren't enough, so Claude ran 28 more test queries. Real symptoms scored 0.37–0.79 and off-topic queries 1.29 or more, which led to **cutoff 1.0**.

## 4. Core logic: ranking, uncertainty, emergency check, doctors, adding cases

**Prompt:**

```
Now build the core logic in diagknows.py (separate from the CLI):

1. rank_conditions(query): retrieve the top 10 matches, group them by condition, and score each condition using both how many neighbors voted for it and how close they were (closer = more weight). Return the top 3 conditions with a confidence label and 1-2 example matching descriptions each.
2. Uncertainty: if the best distance is past the cutoff we picked, return "no confident match" instead of guessing.
3. Emergency check: create a SEPARATE small persistent collection of ~15 hand-written red-flag symptom descriptions (e.g., crushing chest pain spreading to the arm, sudden one-sided face drooping, worst headache of my life, can't breathe). If the query is very close to any of these, skip ranking and tell the user to seek emergency care. Pick that threshold from real distances too.
4. A dict mapping each of the dataset's conditions to which kind of doctor to see, included in the output.
5. add_case(description, condition): adds a new user case with a unique ID (uuid-based) and metadata "source": "user". It should be searchable immediately. Use add(), not upsert(), and make sure IDs can never collide.

Every result shown to the user should include a short disclaimer that this is an educational tool, not a medical diagnosis.
```

**What happened:**
- **No clean emergency threshold.** On the first try, every cutoff either missed real emergencies or flagged everyday queries.
- **Revision:** Claude made the generic red flags more specific (e.g. "high fever" had been matching Malaria) and added a second phrasing for the most critical ones, ending with 20 entries.
- **Fair test:** it checked the revision against a held-out set of emergency phrasings written beforehand.
- **Result: 0.7** flagged 0 of 14 everyday queries, but caught only **6 of 10** held-out emergencies. It missed a stroke description and "I've been thinking about ending it all". This is documented as a known limitation.

## 5. CLI menu

**Prompt:**

```
Wrap everything in main.py as a menu loop:
1. Describe my symptoms
2. Add a new case
3. Show database stats (total items, count per condition, how many user-added)
4. Quit

On startup, build the database automatically if it's empty. Add a requirements.txt. Then walk me through an end-to-end test: add a case, search for it with different wording, quit, relaunch, and confirm it's still there.
```

**What happened:**
- **The end-to-end test:** Claude added a "Gout" case, which isn't in the dataset, and searched for it with different wording.
- **Bug found:** the Gout case was the **closest** neighbor (0.73) but ranked **3rd**. Several distant diabetes neighbors outvoted it. Neighbors past the match cutoff now get zero weight, which fixed it, and accuracy on the labelled test queries stayed the same (8/15 at #1, 14/15 in the top 3).
- **Second bug:** a single neighbor at 0.96 was labelled "high confidence". "High" now also requires a close match.
- **Persistence confirmed:** after quitting and relaunching, a third wording found Gout first, at distance 0.40.
- **Conflict found:** an empty `diagknows/` package folder from step 1 would have blocked importing `diagknows.py`, so it was removed.

## 6. Optional OpenAI explanation

**Prompt:**

```
Add an optional feature: after ranking, send the user's description and the top retrieved patient descriptions to OpenAI to write a 3-4 sentence plain-English explanation of why those conditions came up, using ONLY the retrieved text. Read the API key from a .env file using python-dotenv, add .env to .gitignore, include a .env.example, and make the app still work fully if no key is set.
```

**What happened:** the installed `openai` package was version 3.38, newer than Claude's own knowledge. Claude checked the installed SDK's methods instead of writing calls from memory, and used the Responses API. It also raised the output-token limit, because reasoning models count their thinking tokens against it. The no-key path and a simulated failed call were both tested. **The live API call has not been tested.**

## 7. README draft

**Prompt:**

```
Draft the technical sections of README.md based on the code that actually exists in this repo: how to download Symptom2Disease from Kaggle and where to put it, how the database gets built on first run, which vector database and embedding model are used and what downloads on first run, install and run commands, and the API key setup if present. Read the code before writing. Don't describe features that aren't there.

Leave clearly marked placeholders for these sections, which I'll write myself: what the app does and why, what's stored, how to add/search, my two test queries, and how I used AI.
```

## 8. Trying it myself: readability

**Prompt:**

```
can i try running it
```

Then I ran it, tried one query, and pasted the terminal output with:

```
i just tried it

kind of hard to read the results not easy to use
```

<details><summary>My query and the old output (pasted terminal output, trimmed)</summary>

```
Describe how you feel: sneezing ill chest feels like knife stabbing it

Top matches for: "sneezing ill chest feels like knife stabbing it"

  1. Common Cold  [low confidence: 7/10 neighbors, 61% of score]
     See: Primary care physician
      - "I've been sneezing a lot and feeling really tired and sick. There's also a lot
      of gross stuff coming out of my nose and my throat feels really scratchy. And my
      neck feels swollen too." (distance 0.86)
      ...
```

</details>

**What happened:**
- **Readability fix:** Claude rebuilt the display with plain-language match strength (Strong / Possible / Weak with a ●●○ meter), one trimmed example per condition, capitalized names, a tip when every match is weak, and a compact menu. The technical numbers moved behind `--details`.
- **Matching gap:** my query also showed that mixing two symptoms in one sentence blurs the match. The chest pain part was ignored completely.

**Prompt:**

```
what other details can i add
```

**What happened:** Claude ran richer versions of my query. A detailed description of fever, cough and stabbing chest pain matched **Pneumonia (strong)**, and a detailed sneezing description matched **Common Cold (strong)**. It also found a classic asthma description that still came back as Pneumonia.

## 9. Checking against the assignment requirements

**Prompt:** (followed by the full HW6 assignment page pasted in, not reproduced here)

```
is this good enough for these requirements if not change things
```

**What happened:**
- **Gaps found:** the "earns its place" queries, this prompt log, the README rule that AI-written docs go under a labelled heading at the bottom, and getting the project onto GitHub.
- **Query search:** Claude tested about 30 candidate queries against both the vector search and a simple keyword-matching baseline, looking for ones where the vector search finds the right condition, keyword search fails, and the query shares no important words with the retrieved description.
- **README:** restructured, plus a check that every dataset description (14–78 tokens) fits under the model's 256-token limit.

**Prompt:**

```
commit to github do all the requirements including all the write ups (other than my own words but guide me so i can write it later myself), put it in my portfolio with photos and whatnot and js do everything other than the video and writeups i need to do
```

**What happened:** Claude pushed the repo to GitHub, made terminal screenshots from real app output, added a DiagKnows card to the Projects page of my portfolio site, and wrote a private guide (not committed) for the sections I write myself.

---

## Where the AI got things wrong (and how it was caught)

- **Ranking weights:** the first scoring let several far-away neighbors outvote one close match. Caught by the end-to-end test with the Gout case.
- **Confidence labels:** "high confidence" was shown for a match at distance 0.96. Caught by testing the elephant-on-chest query.
- **Leftover folder:** the empty `diagknows/` package from step 1 conflicted with `diagknows.py`.
- **Weak test query:** the "banana" query failed. I learned that playful comparisons don't land with this embedding model and dataset unless the query also names actual symptoms.
- **Dataset author:** a README draft gave the dataset author a full name guessed from a Kaggle username. It was changed to cite the username.

> **[PLACEHOLDER: author]** Add your own reflections: what you learned, what surprised you, decisions you made yourself, and time spent.
