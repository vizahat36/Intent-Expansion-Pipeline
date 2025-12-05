import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ----------------------------------------------------------------------
# 1. LOAD JSON INPUT
# ----------------------------------------------------------------------

def load_inputs(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ----------------------------------------------------------------------
# 2. BASIC PREPROCESSING
# ----------------------------------------------------------------------

def preprocess_text(txt: str) -> str:
    if txt is None:
        return ""
    txt = txt.replace("\n", " ").strip()
    return txt


# ----------------------------------------------------------------------
# 3. EMBEDDING MODEL + CLUSTERING MODULE
# ----------------------------------------------------------------------

class EmbedCluster:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        print("Embedding messages...")
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    def auto_cluster(self, embeddings: np.ndarray, min_k=5, max_k=40):
        print("Running clustering sweep...")
        best_score = -1
        best_k = None
        best_labels = None

        max_k = min(max_k, embeddings.shape[0] // 2)

        for k in range(min_k, max_k + 1):
            try:
                clusterer = AgglomerativeClustering(n_clusters=k)
                labels = clusterer.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
            except Exception:
                continue

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        return best_labels, {"best_k": best_k, "silhouette": best_score}


# ----------------------------------------------------------------------
# 4. CLUSTER STATS (FOR FINDING SPLIT-WORTHY INTENTS)
# ----------------------------------------------------------------------

def compute_cluster_stats(labels, messages):
    clusters = {}

    for idx, cluster_id in enumerate(labels):
        clusters.setdefault(cluster_id, {"indexes": []})
        clusters[cluster_id]["indexes"].append(idx)

    for cid, info in clusters.items():
        info["size"] = len(info["indexes"])

    return clusters


# ----------------------------------------------------------------------
# 5. LLM STUB (ACTIVATED IN STEP 4)
# ----------------------------------------------------------------------

def call_llm_label_cluster(example_texts: List[str]):
    """
    Stub: In Step 4 we will integrate Gemini / OpenAI.
    For now, return a placeholder to complete Step 3.

    """
    return {
        "label": "candidate_intent",
        "id": "candidate_intent_id",
        "level": "secondary",
        "description": "Auto-generated placeholder intent.",
        "examples": example_texts[:5]
    }


# ----------------------------------------------------------------------
# 6. MAIN PIPELINE
# ----------------------------------------------------------------------

def run_pipeline(input_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading dataset...")
    data = load_inputs(input_path)

    messages = data["customer_messages"]
    processed = []

    # Use current message or fallback to entire history
    for m in messages:
        if "current_human_message" in m:
            txt = m["current_human_message"]
        else:
            txt = m.get("current_message", "")

        txt_clean = preprocess_text(txt)
        processed.append(txt_clean)

    # ------------ EMBEDDINGS ------------
    embedder = EmbedCluster()
    embeddings = embedder.embed_texts(processed)

    # ------------ CLUSTERING ------------
    labels, meta = embedder.auto_cluster(embeddings)

    print("BEST CLUSTER COUNT =", meta["best_k"])
    print("SILHOUETTE SCORE =", meta["silhouette"])

    clusters = compute_cluster_stats(labels, messages)

    # Save raw clusters for analysis before LLM step
    raw_output = []
    for cid, info in clusters.items():
        raw_output.append({
            "cluster_id": int(cid),
            "size": info["size"],
            "messages": [processed[i] for i in info["indexes"]]
        })

    with open(f"{output_dir}/cluster_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_output, f, indent=2, ensure_ascii=False)

    print("Cluster raw output saved at:", f"{output_dir}/cluster_raw.json")
    print("STEP 3 COMPLETED — Now ready for LLM labeling in STEP 4.")


if __name__ == "__main__":
    run_pipeline("inputs_for_assignment.json")
