"""
guardrails.py — Higher-level guardrail functions built on top of Pydantic models.

PATTERN:
  Each function attempts to construct a Pydantic model.
  If validation fails → ValidationError is raised with a clear reason.
  Callers catch ValidationError and handle it (log, reject, sanitise).

This separation keeps your models clean (schema rules) and your
guardrail functions readable (business rules).
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from models import (
    LLMResponse,
    PromptPayload,
    RetrievedDocument,
    UserQuery,
    UserRole,
)


# ---------------------------------------------------------------------------
# Result type — avoids exceptions propagating into your pipeline
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    passed: bool
    data: object | None        # the validated Pydantic model on success
    error: str | None = None   # human-readable reason on failure

    def __str__(self) -> str:
        if self.passed:
            return f"✅  PASSED — {type(self.data).__name__}"
        return f"❌  BLOCKED — {self.error}"


# ---------------------------------------------------------------------------
# 1. Validate incoming user query
# ---------------------------------------------------------------------------

def validate_user_query(
    question: str,
    user_id: str,
    role: str = "user",
) -> GuardrailResult:
    try:
        role_enum = UserRole(role)
    except ValueError:
        return GuardrailResult(
            passed=False,
            data=None,
            error=f"Unknown role '{role}'. Valid roles: {[r.value for r in UserRole]}",
        )

    try:
        query = UserQuery(question=question, user_id=user_id, role=role_enum)
        return GuardrailResult(passed=True, data=query)
    except ValidationError as e:
        reasons = "; ".join(err["msg"] for err in e.errors())
        return GuardrailResult(passed=False, data=None, error=reasons)


# ---------------------------------------------------------------------------
# 2. Validate documents returned from the vector store
# ---------------------------------------------------------------------------

def validate_retrieved_docs(raw_docs: list[dict]) -> list[GuardrailResult]:
    """
    Validates each retrieved doc. Returns a result per doc so the pipeline
    can filter out bad ones rather than crashing entirely.
    """
    results = []
    for doc in raw_docs:
        try:
            validated = RetrievedDocument(**doc)
            results.append(GuardrailResult(passed=True, data=validated))
        except ValidationError as e:
            reasons = "; ".join(err["msg"] for err in e.errors())
            results.append(GuardrailResult(passed=False, data=None, error=f"[doc_id={doc.get('doc_id', '?')}] {reasons}"))
    return results


# ---------------------------------------------------------------------------
# 3. Assemble and validate the final prompt payload
# ---------------------------------------------------------------------------

def build_prompt_payload(
    system_prompt: str,
    good_docs: list[RetrievedDocument],
    question: str,
    max_tokens: int = 512,
) -> GuardrailResult:
    try:
        payload = PromptPayload(
            system_prompt=system_prompt,
            context_chunks=[doc.content for doc in good_docs],
            user_question=question,
            max_tokens=max_tokens,
        )
        return GuardrailResult(passed=True, data=payload)
    except ValidationError as e:
        reasons = "; ".join(err["msg"] for err in e.errors())
        return GuardrailResult(passed=False, data=None, error=reasons)


# ---------------------------------------------------------------------------
# 4. Validate the LLM's response before returning to user
# ---------------------------------------------------------------------------

def validate_llm_response(raw: dict) -> GuardrailResult:
    try:
        response = LLMResponse(**raw)
        return GuardrailResult(passed=True, data=response)
    except ValidationError as e:
        reasons = "; ".join(err["msg"] for err in e.errors())
        return GuardrailResult(passed=False, data=None, error=reasons)
