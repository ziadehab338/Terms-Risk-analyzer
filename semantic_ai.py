import numpy as np
from sentence_transformers import SentenceTransformer


semantic_model = None
dataset_embeddings = None
semantic_df = None


def load_semantic_model():
    global semantic_model

    if semantic_model is None:
        semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    return semantic_model


def build_semantic_index(df):
    global dataset_embeddings, semantic_df

    model = load_semantic_model()

    semantic_df = df.dropna(subset=["text", "binary_label", "risk_type"]).copy()
    semantic_df["text"] = semantic_df["text"].astype(str).str.strip()
    semantic_df = semantic_df[semantic_df["text"] != ""].reset_index(drop=True)

    texts = semantic_df["text"].tolist()

    dataset_embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return dataset_embeddings


def semantic_check(sentence, top_k=5, threshold=0.55):
    global dataset_embeddings, semantic_df

    if dataset_embeddings is None or semantic_df is None:
        raise ValueError("Semantic index is not built. Call build_semantic_index(df) first.")

    model = load_semantic_model()

    query_embedding = model.encode(
        [sentence],
        normalize_embeddings=True
    )[0]

    scores = np.dot(dataset_embeddings, query_embedding)

    top_indices = np.argsort(scores)[::-1][:top_k]

    matches = []
    for idx in top_indices:
        row = semantic_df.iloc[idx]

        matches.append({
            "matched_text": row["text"],
            "binary_label": row["binary_label"],
            "risk_type": row["risk_type"],
            "score": float(scores[idx])
        })

    risky_matches = [
        m for m in matches
        if str(m["binary_label"]).lower() == "risk"
    ]

    best_match = matches[0]

    semantic_risk = (
        len(risky_matches) > 0
        and risky_matches[0]["score"] >= threshold
    )

    if semantic_risk:
        final_type = risky_matches[0]["risk_type"]
        final_score = risky_matches[0]["score"]
        reason = (
            "AI semantic layer detected similar risky clauses from the trained dataset. "
            f"Closest risk meaning: {risky_matches[0]['matched_text']}"
        )
    else:
        final_type = "No Risk"
        final_score = best_match["score"]
        reason = "AI semantic layer did not find a strong risky semantic match."

    return {
        "semantic_risk": semantic_risk,
        "semantic_risk_type": final_type,
        "semantic_score": final_score,
        "semantic_reason": reason,
        "semantic_matches": matches
    }
