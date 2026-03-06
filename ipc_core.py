"""
ipc_core.py — Lazy-loading version of the IPC prediction engine.
Use get_models() to load models on demand (compatible with @st.cache_resource).
"""

import os
import json
import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# CONFIG
# -------------------------------
IPC_FILE = os.path.join(os.path.dirname(__file__), "merged.json")
MODEL_NAME = "all-mpnet-base-v2"
SIMILARITY_THRESHOLD = 0.35
HYBRID_WEIGHT_EMBEDDING = 0.6
HYBRID_WEIGHT_KEYWORD = 0.4


# -------------------------------
# TEXT PREPROCESSING
# -------------------------------
def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s§\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -------------------------------
# SEVERITY HELPERS
# -------------------------------
def estimate_crime_severity(text: str) -> float:
    if not text:
        return 0.5
    text_lower = text.lower()

    extreme_keywords = [
        'murder', 'kill', 'killed', 'stabbed to death', 'beheaded', 'hanged',
        'rape', 'raped', 'sexual assault', 'molested badly', 'gang rape',
        'terrorism', 'terrorist', 'bomb', 'explosive', 'blast', 'grenade',
        'shooting', 'firing', 'massacre', 'genocide',
        'kidnap', 'abduct', 'taken away', 'hostage', 'missing forcefully',
        'suicide', 'attempt suicide', 'jumped from', 'drank poison',
        'burn alive', 'acid attack'
    ]
    high_keywords = [
        'assault', 'attacked', 'beaten up', 'hit badly', 'punched',
        'hurt', 'injured', 'stab', 'knife attack', 'shot with gun',
        'rob', 'robbed', 'snatched bag', 'stole purse', 'chain snatching',
        'burglary', 'broke into house', 'looted', 'dacoity',
        'extort', 'demand money', 'blackmail', 'ransom',
        'threaten', 'life threat', 'warned to kill',
        'weapon', 'armed', 'gunpoint', 'knife point',
        'molested', 'touched inappropriately', 'forced himself',
        'arson', 'set fire', 'burnt house'
    ]
    medium_keywords = [
        'cheat', 'cheated', 'fraud', 'scam', 'took money', 'promised but ran away',
        'forgery', 'fake papers', 'counterfeit notes',
        'thief', 'pickpocket', 'wallet stolen', 'phone stolen',
        'steal', 'stole bike', 'cycle stolen', 'house theft',
        'embezzlement', 'took office money', 'bribery',
        'damage', 'broke window', 'damaged car', 'smashed property',
        'harass', 'harassed', 'eve teasing', 'catcalling',
        'defamation', 'spread rumors', 'insult publicly',
        'stalking', 'followed me', 'kept coming behind'
    ]
    low_keywords = [
        'trespass', 'entered property', 'jumped wall', 'came inside without permission',
        'nuisance', 'loud music', 'shouting on street',
        'disobey', 'not following rules', 'argued with officer',
        'disturb', 'disturbance', 'creating scene',
        'loitering', 'roaming suspiciously', 'wandering around',
        'simple hurt', 'slapped', 'pushed', 'minor injury',
        'public drinking', 'drunk on road', 'drunk fight',
        'noise complaint', 'neighbors fighting loudly',
        'misconduct', 'misbehaved', 'abused in public'
    ]

    extreme_count = sum(1 for w in extreme_keywords if w in text_lower)
    high_count    = sum(1 for w in high_keywords    if w in text_lower)
    medium_count  = sum(1 for w in medium_keywords  if w in text_lower)
    low_count     = sum(1 for w in low_keywords     if w in text_lower)

    total = extreme_count + high_count + medium_count + low_count
    if total == 0:
        return 0.5

    score = (extreme_count * 1.0 + high_count * 0.75 +
             medium_count * 0.5  + low_count  * 0.25) / total
    return min(score, 1.0)


def estimate_punishment_severity(punishment_text: str) -> float:
    if not punishment_text:
        return 0.5
    t = punishment_text.lower()
    if any(w in t for w in ['death', 'life imprisonment', 'imprisonment for life']):
        return 1.0
    if any(w in t for w in ['10 year', 'ten year', '14 year', 'fourteen year']):
        return 0.75
    if any(w in t for w in ['7 year', 'seven year', '5 year', 'five year', '3 year', 'three year']):
        return 0.5
    if any(w in t for w in ['1 year', 'one year', '6 month', 'fine only', 'month']):
        return 0.25
    return 0.5


# -------------------------------
# MODEL LOADING (lazy, cache-friendly)
# -------------------------------
_models = None   # cached in-process

def get_models():
    """Load and cache all models + data. Call once; subsequent calls are instant."""
    global _models
    if _models is not None:
        return _models

    # Load IPC data
    with open(IPC_FILE, "r", encoding="utf-8") as f:
        ipc_data = json.load(f)

    ipc_sections = list(ipc_data.keys())
    ipc_texts = [
        f"{ipc} {ipc_data[ipc]['offense']} {ipc_data[ipc]['description']} {ipc_data[ipc]['simple_words']}"
        for ipc in ipc_sections
    ]
    ipc_texts_processed = [preprocess_text(t) for t in ipc_texts]

    # Precompute punishment severity
    ipc_punishment_severity = {
        ipc: estimate_punishment_severity(ipc_data[ipc].get('punishment', ''))
        for ipc in ipc_sections
    }

    # Load sentence-transformer
    model = SentenceTransformer(MODEL_NAME)

    # Load or create embeddings
    base_dir = os.path.dirname(__file__)
    embed_file = os.path.join(base_dir, f"ipc_embeddings_{MODEL_NAME.replace('/', '_')}.pkl")
    if os.path.exists(embed_file):
        with open(embed_file, "rb") as f:
            ipc_sections, ipc_embeddings = pickle.load(f)
    else:
        ipc_embeddings = model.encode(ipc_texts_processed, convert_to_tensor=True)
        with open(embed_file, "wb") as f:
            pickle.dump((ipc_sections, ipc_embeddings), f)

    # Load or create TF-IDF
    tfidf_file = os.path.join(base_dir, "ipc_tfidf.pkl")
    if os.path.exists(tfidf_file):
        with open(tfidf_file, "rb") as f:
            tfidf_vectorizer, tfidf_matrix = pickle.load(f)
    else:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000, stop_words='english', ngram_range=(1, 2)
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(ipc_texts_processed)
        with open(tfidf_file, "wb") as f:
            pickle.dump((tfidf_vectorizer, tfidf_matrix), f)

    _models = {
        "ipc_data": ipc_data,
        "ipc_sections": ipc_sections,
        "ipc_embeddings": ipc_embeddings,
        "ipc_punishment_severity": ipc_punishment_severity,
        "model": model,
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
    }
    return _models


# -------------------------------
# HYBRID SEARCH
# -------------------------------
def find_ipc_sections(user_input: str, top_k: int = 5) -> list[dict]:
    """Main prediction function. Returns list of result dicts."""
    m = get_models()

    processed = preprocess_text(user_input)
    crime_severity = estimate_crime_severity(user_input)

    # Semantic scores
    query_embedding = m["model"].encode(processed, convert_to_tensor=True)
    semantic_scores = util.pytorch_cos_sim(query_embedding, m["ipc_embeddings"])[0].cpu().numpy()

    # Keyword scores
    query_tfidf = m["tfidf_vectorizer"].transform([processed])
    keyword_scores = cosine_similarity(query_tfidf, m["tfidf_matrix"]).flatten()

    # Normalize
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    semantic_scores = normalize(semantic_scores)
    keyword_scores  = normalize(keyword_scores)

    hybrid_scores = (HYBRID_WEIGHT_EMBEDDING * semantic_scores +
                     HYBRID_WEIGHT_KEYWORD   * keyword_scores)

    # Severity calibration
    calibrated = []
    for i, score in enumerate(hybrid_scores):
        ipc = m["ipc_sections"][i]
        punish_sev = m["ipc_punishment_severity"][ipc]
        severity_diff = abs(punish_sev - crime_severity)
        penalty = 1.0 - severity_diff * 0.3
        calibrated.append(score * penalty)
    calibrated = np.array(calibrated)

    top_indices = np.argsort(calibrated)[::-1][:top_k * 2]

    results = []
    for idx in top_indices:
        score = calibrated[idx]
        if score < SIMILARITY_THRESHOLD and len(results) >= top_k:
            continue
        ipc = m["ipc_sections"][idx]
        det = m["ipc_data"][ipc]
        punish_sev = m["ipc_punishment_severity"][ipc]
        results.append({
            "ipc": ipc,
            "offense": det.get("offense", "N/A"),
            "punishment": det.get("punishment", "N/A"),
            "cognizable": det.get("cognizable", "N/A"),
            "bailable": det.get("bailable", "N/A"),
            "court": det.get("court", "N/A"),
            "description": det.get("simple_words", det.get("description", "")),
            "score": float(score),
            "crime_severity": crime_severity,
            "punishment_severity": punish_sev,
            "severity_match": 1.0 - abs(crime_severity - punish_sev),
        })
        if len(results) >= top_k:
            break

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
