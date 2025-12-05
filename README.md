# Intent Expansion Pipeline — Verifast Tech

**INTRODUCTION**

This assignment focuses on designing an Intent Expansion Pipeline that strengthens the accuracy, scalability, and reliability of a Conversational AI system. The work was completed for Verifast Tech, which aims to improve its customer-support automation by identifying hidden, missing, or split-worthy intents in real chat data.

My role in this assignment was to:

- Analyse the existing intent mapper and real user messages.
- Build a scalable and explainable workflow to discover new intents.
- Validate findings using LLM-assisted semantic clustering.
- Produce actionable, recommendation-ready outputs for the AI workflow team.

**PROBLEM STATEMENT**

Verifast Tech’s current NLU model handles a broad set of intents but struggles when users express more specific needs. Real chat messages revealed that:

- Several intents were not captured in the existing mapper.
- Some messages were incorrectly grouped under broad intents.
- The system lacked a repeatable, scalable method to discover new intents.

Therefore, the company required a systematic pipeline that can:

- Analyse unstructured user messages.
- Cluster them semantically.
- Detect missing or high-value intents.
- Validate them with interpretable, evidence-driven reasoning.

**PROJECT OVERVIEW**

The Intent Expansion Pipeline was designed to produce three major outcomes:

- Discovery of new intents — Identify clusters that represent meaningful user goals not present in the existing intent list.
- Interpretability & justification — Ensure every identified intent is backed by quantitative cluster scores and qualitative message patterns.
- A repeatable workflow — Create a pipeline that can be executed on future datasets with minimal manual intervention.

**WORKFLOW ARCHITECTURE DIAGRAM (TEXT DIAGRAM)**

<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/63dd7033-a96d-4df8-919b-be8bfd7c9e2c" />
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/10478898-b780-41ae-901f-23b95e56040f" />


                     ┌────────────────────────┐
                     │     Input Dataset       │
                     │  (200+ real messages)   │
                     └────────────┬────────────┘
                                  │
                                  ▼
                   ┌────────────────────────────┐
                   │   Preprocessing Layer       │
                   │  - Clean text               │
                   │  - Remove noise             │
                   │  - Extract “current message”│
                   └────────────┬───────────────┘
                                  │
                                  ▼
           ┌──────────────────────────────────────────┐
           │        Embedding Engine (SBERT)          │
           │   - Convert messages → vector embeddings │
           │   - Model: all-MiniLM-L6-v2              │
           └───────────────────┬──────────────────────┘
                               │
                               ▼
         ┌──────────────────────────────────────────────┐
         │        Clustering Module (Agglomerative)     │
         │   - Auto-select best K using silhouette      │
         │   - Groups semantically similar messages     │
         │   - Output: cluster_raw.json                 │
         └────────────────────────┬─────────────────────┘
                                  │
                                  ▼
      ┌─────────────────────────────────────────────────────┐
      │        LLM Intent Expansion Engine (Gemini 2.5)     │
      │   - Takes cluster samples (≤12 msgs)                │
      │   - Generates candidate intents                     │
      │   - Outputs JSON-only results                       │
      │   - Guardrails: confidence, parsing validation      │
      └─────────────────────────┬──────────────────────────┘
                                │
                                ▼
     ┌──────────────────────────────────────────────────────┐
     │              JSON Validation + Guardrails            │
     │   - Fix JSON formatting                              │
     │   - Mark low-confidence intents                      │
     │   - Flag ambiguous clusters                          │
     └───────────────────────────┬──────────────────────────┘
                                 │
                                 ▼
         ┌──────────────────────────────────────────────┐
         │                 Output Layer                 │
         │   intent_suggestions.json                    │
         │   (Final recommended intents)                │
         └──────────────────────────────────────────────┘

**FLOWCHART OF ENTIRE PIPELINE**

START
  │
  ▼
Load JSON Input  
  │
  ▼
Preprocess Messages (clean, normalize)
  │
  ▼
Generate Embeddings (SBERT)
  │
  ▼
Cluster Messages (Agglomerative)
  │
  ▼
For each cluster ≥ size threshold:
    ├─► Sample messages
    ├─► Send to Gemini 2.5 Flash
    ├─► Generate intent suggestion (JSON)
    └─► Validate + score confidence
  │
  ▼
Aggregate all suggestions
  │
  ▼
Write final output → `intent_suggestions.json`
  │
  ▼
END

<img width="1024" height="1536" alt="ChatGPT Image Dec 5, 2025, 08_25_29 PM" src="https://github.com/user-attachments/assets/70c71a01-30ef-46b6-939f-e1a6bfcd28aa" />


**SUMMARY OF FINDINGS**

Two high-value intents emerged from the clustering analysis:

Intent 1: Track Order
Cluster Size: 16
Confidence: High
Users consistently requested delivery status, tracking information, or package updates.

Intent 2: Requires Hindi Support
Cluster Size: 15
Confidence: 0.9
Users clearly expressed difficulty with English and asked to continue the conversation in Hindi.

**INTENT DETAILS**

Intent Name: Track Order
ID: `track_order`
Level: Primary
Description: Queries asking for shipment status, order tracking, or expected delivery time.
Representative samples:
“How to track order?”, “Where is my package?”, “Help me track order”

Justification:
Tightly grouped cluster, distinct from general order issues, requires its own top-level intent.

Intent Name: Requires Hindi Support
ID: `requires_hindi_support`
Level: Secondary
Description: Users requesting conversation in Hindi or expressing inability to understand English.
Representative samples:
“Hindi me batao please”, “Aap Hindi me bolo mujhe English samaj nahi aata”

Justification:
Strong semantic grouping, high LLM confidence, supports multilingual agent workflows.

**QUANTITATIVE EVIDENCE**

Best cluster count selected: 40
Average cluster size: ~5
Two clusters were significantly larger (≥12) and semantically tight.
High silhouette relevance ensures accurate separation of user intents.
The signals strongly justify elevating these as new system intents.

**EXECUTION EVIDENCE (SCREENSHOTS)**

Place the two screenshots from the project here. The files were saved in `output/` during execution. If you want the images embedded in the README, save them in `intent_expansion/output/` and update the filenames below.


<img width="872" height="499" alt="image" src="https://github.com/user-attachments/assets/e8cb188a-35c2-4e8e-a46d-ed1d89db88e2" />
<img width="880" height="665" alt="image" src="https://github.com/user-attachments/assets/e5ac3232-5727-466a-a934-a507def43318" />


**WHY THESE INTENTS IMPROVE THE SYSTEM**

Track Order

- Reduces misclassification under broad order categories.
- Allows direct routing to tracking APIs.
- Improves customer satisfaction by minimizing ambiguity.

Requires Hindi Support

- Enables multilingual routing logic.
- Reduces communication friction.
- Enhances personalization in NLU processing.

**GUARDRAILS & SCALABILITY**

Guardrails applied:

- Minimum cluster size threshold
- Enforced JSON-only LLM responses
- Confidence scoring for all intents
- Parser validation with fallback

Scalability achieved through separation of concerns:

- Embeddings handle large-scale semantic representation
- Clustering generalizes across thousands of messages
- LLM used only on representative samples to reduce cost and hallucination

**CONCLUSION**

The Intent Expansion Pipeline successfully surfaced two missing intents and provided robust justification for including them. The solution is scalable, explainable, cost-efficient, and aligned with Verifast Tech’s automation and customer experience objectives.

**APPROACH & LEARNING DOCUMENT**

APPROACH OVERVIEW
The assignment began with a clear understanding of the problem: Verifast Tech required a scalable and explainable method to uncover missing and split-worthy intents from real customer messages. My approach involved:

- Analysing raw chat data and the existing intent taxonomy.
- Designing a structured workflow capable of processing large volumes of messages.
- Applying semantic embeddings to capture meaning beyond keywords.
- Performing clustering to reveal natural conversational themes.
- Using LLM-based reasoning to assign accurate intent names with justification.
- Validating outputs through quantitative metrics and sample inspection.

TECHNICAL LEARNINGS

Throughout the pipeline development, several technical insights emerged:

- Semantic embeddings provide far superior grouping than keyword-based techniques.
- Clustering helps uncover structure even when user messages appear noisy or chaotic.
- LLMs perform best at intent naming when strict guardrails and JSON constraints are applied.
- Guardrails significantly reduce hallucination and maintain consistency in downstream parsing.
- Extracting structured JSON from an LLM requires careful prompt engineering and validation routines.

PROJECT LEARNINGS

Beyond the technical execution, this assignment highlighted several broader lessons:

- Intent taxonomies must evolve continuously based on real user behavior.
- Customers frequently shift topics mid-conversation, requiring flexible NLU logic.
- Intent granularity must be balanced to avoid overfitting while still capturing key distinctions.
- Enterprise-scale conversational AI systems need workflows that are reproducible, explainable, and resilient.
- A well-architected pipeline reduces operational burden and improves long-term maintainability.

---

============================================================

**AUTHOR & PROJECT COMPLETION**

Prepared By:
Mohammed Vijahath
B.Tech – Artificial Intelligence & Machine Learning
University Visvesvaraya College of Engineering (UVCE), Bangalore

Email: mohammedvijahath@gmail.com

This project was fully implemented, tested, and documented as part of the AI Workflow Analyst Assignment for Verifast Tech. It demonstrates end-to-end intent discovery using embeddings, clustering, LLM reasoning, guardrails, and JSON-validated outputs.

============================================================
