# Pydantic in RAG: Prompt Injection & Guardrails

A hands-on mini-project showing exactly where and why Pydantic adds safety to a RAG pipeline.

## Project Structure

```
pydantic_rag_guardrails/
├── README.md
├── models.py          # All Pydantic models (the core lesson)
├── guardrails.py      # Validation logic using those models
├── rag_pipeline.py    # Simulated RAG pipeline end-to-end
└── run_demo.py        # Run all scenarios: safe vs. injected inputs
```

## What You'll Learn

| File | Concept |
|------|---------|
| `models.py` | Define strict schemas for every RAG input/output |
| `guardrails.py` | Use validators to detect/block prompt injection |
| `rag_pipeline.py` | Wire models + guardrails into a full pipeline |
| `run_demo.py` | See real attack attempts get caught |

## Run It

```bash
pip install pydantic
python run_demo.py
```

============================================================
  KEY TAKEAWAYS
============================================================

  1. Pydantic models are your SCHEMA — define rules once, enforce everywhere.
  2. field_validator catches injection patterns before they reach the LLM.
  3. model_validator enforces cross-field rules (e.g., total token budget).
  4. Retrieved documents need the SAME validation as user inputs
     (indirect prompt injection is a real attack vector).
  5. Output guardrails matter too — validate what the LLM returns.
  6. ValidationError gives you structured errors, not fragile string parsing.
