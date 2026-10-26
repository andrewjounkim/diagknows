"""DiagKnows core logic: emergency check, condition ranking, and adding cases.

Distances: the collections use Chroma's default "l2" space, which returns
SQUARED Euclidean distance. The default embedding model (all-MiniLM-L6-v2)
produces unit-length vectors, so distance = 2 - 2 * cosine_similarity:
0 = identical meaning, 2 = unrelated, 4 = opposite. The cutoffs below were
picked from real query distances (see README).
"""

import uuid
from collections import defaultdict
from functools import lru_cache

import chromadb

from load_data import COLLECTION_NAME, SOURCE, STORE_PATH, load_dataset

EMERGENCY_COLLECTION_NAME = "red_flags"
N_NEIGHBORS = 10
TOP_CONDITIONS = 3
EXAMPLES_PER_CONDITION = 2

# Best match must be closer than this or we report "no confident match".
# Real symptom queries scored 0.37-0.79; off-topic queries scored 1.29+.
MATCH_CUTOFF = 1.0
# A neighbor's vote weight falls linearly from 1 at distance 0 to 0 at the
# match cutoff: closer neighbors count more, and a neighbor too far to be a
# match by itself gets no vote. (Zeroing at 1.5 let several weak neighbors
# outvote one strong match.)
ZERO_WEIGHT_DISTANCE = MATCH_CUTOFF
# A query this close to any red-flag description triggers the emergency message.
# At 0.7: 0/14 everyday test queries and 22/1200 dataset descriptions flagged,
# but only 6/10 held-out emergency paraphrases caught. It's a safety net with gaps.
EMERGENCY_CUTOFF = 0.7

DISCLAIMER = (
    "DiagKnows is an educational tool, not a medical diagnosis. "
    "Always consult a qualified healthcare professional."
)

EMERGENCY_MESSAGE = (
    "Your description resembles symptoms that can be a medical emergency. "
    "Call 911 (or your local emergency number) or go to the nearest emergency room now. "
    "If you are thinking about harming yourself, call or text 988 (US Suicide & Crisis Lifeline)."
)

# The most critical emergencies get two phrasings, since one wording alone
# missed common paraphrases during calibration.
RED_FLAGS = [
    "Crushing chest pain or heavy pressure, like something sitting on my chest, spreading to my arm, jaw, or back.",
    "Chest pain that started suddenly with sweating, nausea, and shortness of breath.",
    "Sudden face drooping on one side, slurred speech or can't get words out, weakness or numbness in one arm or leg.",
    "I suddenly can't move or feel one side of my body.",
    "I suddenly got the worst headache of my life, it hit like a thunderclap out of nowhere.",
    "I can't breathe, I'm gasping and choking for air and my lips are turning blue.",
    "I'm suffocating and can't get any air in at all, even sitting still.",
    "My throat is closing up and my tongue and lips are swelling after eating something or getting stung.",
    "I'm coughing up or vomiting large amounts of blood.",
    "Sudden unbearable abdominal pain, my belly is rigid and hard as a board.",
    "Someone had a seizure and they are not waking up or responding.",
    "Someone is unconscious or unresponsive and we can't wake them up.",
    "I am thinking about killing myself and want to end my life.",
    "I don't want to live anymore and I have a plan to hurt myself.",
    "Neck so stiff I can't bend it forward, with fever and a purple rash that doesn't fade when pressed.",
    "I suddenly lost vision in one or both eyes.",
    "Heavy bleeding that won't stop even when I press on it.",
    "I took way too many pills or swallowed something poisonous.",
    "Sudden severe tearing or ripping pain in my chest that goes straight through to my back.",
    "I fainted and my heart is racing or pounding irregularly, with chest pain.",
]

# Keys match the dataset's labels exactly (including its inconsistent capitalization).
DOCTOR_FOR_CONDITION = {
    "Acne": "Dermatologist",
    "Arthritis": "Rheumatologist",
    "Bronchial Asthma": "Pulmonologist",
    "Cervical spondylosis": "Orthopedist or spine specialist",
    "Chicken pox": "Primary care physician",
    "Common Cold": "Primary care physician",
    "Dengue": "Infectious disease specialist",
    "Dimorphic Hemorrhoids": "Gastroenterologist or colorectal surgeon",
    "Fungal infection": "Dermatologist",
    "Hypertension": "Primary care physician or cardiologist",
    "Impetigo": "Dermatologist",
    "Jaundice": "Hepatologist or gastroenterologist",
    "Malaria": "Infectious disease specialist",
    "Migraine": "Neurologist",
    "Pneumonia": "Pulmonologist",
    "Psoriasis": "Dermatologist",
    "Typhoid": "Infectious disease specialist",
    "Varicose Veins": "Vascular surgeon",
    "allergy": "Allergist / immunologist",
    "diabetes": "Endocrinologist",
    "drug reaction": "Primary care physician or allergist",
    "gastroesophageal reflux disease": "Gastroenterologist",
    "peptic ulcer disease": "Gastroenterologist",
    "urinary tract infection": "Primary care physician or urologist",
}
DEFAULT_DOCTOR = "Primary care physician"


@lru_cache(maxsize=1)
def get_client():
    return chromadb.PersistentClient(path=STORE_PATH)


def get_symptoms_collection():
    return get_client().get_or_create_collection(name=COLLECTION_NAME)


def get_emergency_collection():
    return get_client().get_or_create_collection(name=EMERGENCY_COLLECTION_NAME)


def ensure_database() -> None:
    """Build both collections if they're missing or incomplete."""
    load_dataset(get_symptoms_collection())

    red_flags = get_emergency_collection()
    if red_flags.count() != len(RED_FLAGS):
        # Rebuild from scratch so edits to RED_FLAGS never leave stale entries behind.
        get_client().delete_collection(EMERGENCY_COLLECTION_NAME)
        red_flags = get_emergency_collection()
        red_flags.add(
            ids=[f"rf-{i:02d}" for i in range(1, len(RED_FLAGS) + 1)],
            documents=RED_FLAGS,
        )


def doctor_for(condition: str) -> str:
    return DOCTOR_FOR_CONDITION.get(condition, DEFAULT_DOCTOR)


def check_emergency(query: str) -> dict:
    result = get_emergency_collection().query(
        query_texts=[query], n_results=1, include=["documents", "distances"]
    )
    distance = result["distances"][0][0]
    return {
        "is_emergency": distance <= EMERGENCY_CUTOFF,
        "distance": distance,
        "matched_red_flag": result["documents"][0][0],
    }


def confidence_label(share: float, best_distance: float) -> str:
    # Needs both agreement among neighbors (share) and a genuinely close match;
    # correct top matches in testing were mostly 0.37-0.69, weak ones 0.8+.
    if share >= 0.6 and best_distance <= 0.6:
        return "high"
    if share >= 0.3 and best_distance <= 0.8:
        return "medium"
    return "low"


def rank_conditions(query: str) -> dict:
    """Rank the conditions a symptom description most resembles.

    Returns a dict whose "status" is one of:
      "emergency" - query resembles a red-flag symptom; no ranking is done
      "no_match"  - nearest neighbor is beyond MATCH_CUTOFF
      "ok"        - "conditions" holds up to TOP_CONDITIONS ranked entries
    Every result includes "disclaimer".
    """
    query = query.strip()
    base = {"query": query, "disclaimer": DISCLAIMER}

    emergency = check_emergency(query)
    if emergency["is_emergency"]:
        return {**base, "status": "emergency", "message": EMERGENCY_MESSAGE, "emergency": emergency}

    results = get_symptoms_collection().query(
        query_texts=[query],
        n_results=N_NEIGHBORS,
        include=["documents", "metadatas", "distances"],
    )
    # One inner list per query text; we sent one.
    neighbors = [
        {"id": id_, "text": doc, "condition": meta["condition"], "source": meta["source"], "distance": dist}
        for id_, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
    best_distance = neighbors[0]["distance"] if neighbors else None

    if best_distance is None or best_distance >= MATCH_CUTOFF:
        return {
            **base,
            "status": "no_match",
            "message": "No confident match. Try describing your symptoms in more detail.",
            "best_distance": best_distance,
        }

    # Each neighbor within the cutoff votes for its condition; closer neighbors carry more weight.
    by_condition = defaultdict(list)
    for n in neighbors:
        if n["distance"] < ZERO_WEIGHT_DISTANCE:
            by_condition[n["condition"]].append(n)
    scores = {
        condition: sum(1 - n["distance"] / ZERO_WEIGHT_DISTANCE for n in group)
        for condition, group in by_condition.items()
    }
    total = sum(scores.values())

    ranked = sorted(scores, key=scores.get, reverse=True)[:TOP_CONDITIONS]
    conditions = []
    for condition in ranked:
        share = scores[condition] / total
        group = by_condition[condition]  # already sorted nearest-first
        conditions.append({
            "condition": condition,
            "score": round(scores[condition], 3),
            "share": round(share, 3),
            "confidence": confidence_label(share, group[0]["distance"]),
            "votes": len(group),
            "best_distance": round(group[0]["distance"], 3),
            "doctor": doctor_for(condition),
            "examples": [
                {"text": n["text"], "distance": round(n["distance"], 3), "source": n["source"]}
                for n in group[:EXAMPLES_PER_CONDITION]
            ],
        })

    return {**base, "status": "ok", "best_distance": best_distance, "conditions": conditions}


def known_conditions() -> list[str]:
    return sorted(DOCTOR_FOR_CONDITION, key=str.lower)


def add_case(description: str, condition: str) -> str:
    """Add a user-supplied case. Returns its new ID. Searchable immediately."""
    description, condition = description.strip(), condition.strip()
    if not description or not condition:
        raise ValueError("Both a description and a condition are required.")

    # Reuse the dataset's spelling if the condition matches one case-insensitively.
    canonical = {c.lower(): c for c in DOCTOR_FOR_CONDITION}
    condition = canonical.get(condition.lower(), condition)

    collection = get_symptoms_collection()
    # uuid4 is random 128-bit; still, confirm the ID is unused before add().
    while True:
        case_id = f"user-{uuid.uuid4()}"
        if not collection.get(ids=[case_id], include=[])["ids"]:
            break
    collection.add(
        ids=[case_id],
        documents=[description],
        metadatas=[{"condition": condition, "source": "user"}],
    )
    return case_id


def stats() -> dict:
    metadatas = get_symptoms_collection().get(include=["metadatas"])["metadatas"]
    per_condition = defaultdict(int)
    user_added = 0
    for meta in metadatas:
        per_condition[meta["condition"]] += 1
        user_added += meta.get("source") == "user"
    return {
        "total": len(metadatas),
        "user_added": user_added,
        "per_condition": dict(sorted(per_condition.items(), key=lambda kv: kv[0].lower())),
    }
