"""
intent_expansion_pipeline_with_gemini.py
Final fixed version — JSON escaping + Gemini 3 API
"""

import os
import json
import time
import re
from typing import List, Dict, Any
import google.generativeai as genai

# --------------------------------
# CONFIG
# --------------------------------
# MODEL_NAME = "gemini-1.5-flash-latest"
MODEL_NAME = "gemini-2.5-flash"


MIN_CLUSTER_SIZE = 12
OUTPUT_DIR = "output"
CLUSTER_RAW = os.path.join(OUTPUT_DIR, "cluster_raw.json")
SUGGESTIONS_PATH = os.path.join(OUTPUT_DIR, "intent_suggestions.json")


# --------------------------------
# SETUP GEMINI
# --------------------------------
def configure_genai():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set! Use: $env:GOOGLE_API_KEY='your_key'")
    genai.configure(api_key=api_key)
    print("Gemini configured with model:", MODEL_NAME)


# --------------------------------
# PROPERLY ESCAPED PROMPT TEMPLATE
# --------------------------------
PROMPT_TEMPLATE = """
You are given a list of customer messages belonging to the same cluster.
Your task is to propose a new intent strictly as JSON in this schema:

{{
  "label": "<concise_snake_case_label>",
  "id": "<snake_case_id>",
  "level": "<primary|secondary>",
  "short_description": "<one sentence clear definition>",
  "when_to_use": "<rule for when the classifier should pick this intent>",
  "examples": ["ex1", "ex2", "ex3"],
  "confidence": <0.0_to_1.0>,
  "notes": "<optional>"
}}

Rules:
- MUST return pure JSON. No markdown.
- Use snake_case.
- level = secondary unless clearly new primary intent.
- confidence < 0.6 means ambiguity.

Cluster messages:
{examples}
"""


# --------------------------------
# JSON EXTRACTION
# --------------------------------
def extract_json_from_text(text: str) -> str:
    match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if not match:
        return text
    candidate = match.group(1)
    candidate = candidate.replace("'", '"')
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)
    return candidate


# --------------------------------
# GEMINI CALL
# --------------------------------
def label_cluster_with_gemini(messages: List[str]) -> Dict[str, Any]:
    examples_text = "\n".join([f"- {m}" for m in messages])
    prompt = PROMPT_TEMPLATE.format(examples=examples_text)

    model = genai.GenerativeModel(MODEL_NAME)

    try:
        response = model.generate_content(prompt)
        raw_text = response.text
    except Exception as e:
        return {
            "label": "gemini_call_failed",
            "id": "gemini_call_failed",
            "level": "secondary",
            "short_description": f"Gemini error: {str(e)}",
            "when_to_use": "",
            "examples": messages[:3],
            "confidence": 0.0,
            "notes": "Generation failed"
        }

    json_text = extract_json_from_text(raw_text)

    try:
        obj = json.loads(json_text)
    except:
        return {
            "label": "json_parse_failed",
            "id": "json_parse_failed",
            "level": "secondary",
            "short_description": "LLM output could not be parsed",
            "when_to_use": "",
            "examples": messages[:3],
            "confidence": 0.0,
            "notes": raw_text[:500]
        }

    obj.setdefault("confidence", 0.0)
    obj["confidence"] = float(obj["confidence"])

    return obj


# --------------------------------
# MAIN PIPELINE
# --------------------------------
def main():
    configure_genai()

    if not os.path.exists(CLUSTER_RAW):
        raise FileNotFoundError("cluster_raw.json not found — run embedding pipeline first.")

    with open(CLUSTER_RAW, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    suggestions = []

    for cluster in clusters:
        cid = cluster["cluster_id"]
        size = cluster["size"]
        messages = cluster["messages"]

        if size < MIN_CLUSTER_SIZE:
            suggestions.append({
                "cluster_id": cid,
                "size": size,
                "status": "skipped_small_cluster",
                "reason": f"size < {MIN_CLUSTER_SIZE}",
                "examples": messages[:3]
            })
            continue

        print(f"\nLabeling cluster {cid} (size={size})...")

        sample = messages[:12]
        llm_output = label_cluster_with_gemini(sample)

        status = "candidate_high_confidence" if llm_output["confidence"] >= 0.6 else "candidate_low_confidence"

        suggestions.append({
            "cluster_id": cid,
            "size": size,
            "llm_output": llm_output,
            "status": status,
            "examples": sample
        })

        time.sleep(0.4)

    # Save
    output = {"model": MODEL_NAME, "suggestions": suggestions}

    with open(SUGGESTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nDONE — Intent suggestions saved to:", SUGGESTIONS_PATH)


if __name__ == "__main__":
    main()
