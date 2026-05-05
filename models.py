"""
models.py — Pydantic schemas for every stage of a RAG pipeline.

WHY THIS MATTERS:
  Without Pydantic, inputs are raw strings — anything can slip through.
  With Pydantic, every field has a type, length limit, and custom validator.
  Injection attempts fail at the schema boundary, before touching your LLM.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# 1. USER QUERY — the first attack surface
# ---------------------------------------------------------------------------

class UserQuery(BaseModel):
    """
    Validates the raw user question before it enters the pipeline.

    Key guardrails:
      - max length stops token-flooding attacks
      - field_validator strips known injection patterns
      - role field is an Enum, so it cannot be overridden by free text
    """

    question: Annotated[str, Field(min_length=3, max_length=500)]
    user_id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{3,50}$")]
    role: "UserRole" = "UserRole.USER"  # resolved below after UserRole is defined

    @field_validator("question")
    @classmethod
    def no_injection_patterns(cls, v: str) -> str:
        """
        Block common prompt injection attempts.
        Real-world: use a classifier; this regex demo shows the concept.
        """
        injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions?",
            r"disregard\s+.{0,30}instructions?",
            r"you\s+are\s+now\s+",              # persona hijacking
            r"act\s+as\s+(if\s+you\s+are\s+)?",
            r"system\s*:\s*",                   # fake system prompts
            r"<\s*/?system\s*>",                # XML injection
            r"\bDAN\b",                         # jailbreak alias
            r"reveal\s+(your\s+)?(prompt|instructions?|system)",
        ]
        lowered = v.lower()
        for pattern in injection_patterns:
            if re.search(pattern, lowered, re.IGNORECASE):
                raise ValueError(
                    f"Prompt injection detected. Pattern matched: '{pattern}'"
                )
        return v

    @field_validator("question")
    @classmethod
    def no_excessive_special_chars(cls, v: str) -> str:
        """Catch attempts to escape context via special chars."""
        special_ratio = sum(1 for c in v if not c.isalnum() and c not in " .,?!'-") / len(v)
        if special_ratio > 0.3:
            raise ValueError("Too many special characters — possible obfuscation attempt.")
        return v


class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    READONLY = "readonly"


# Fix the forward reference
UserQuery.model_rebuild()


# ---------------------------------------------------------------------------
# 2. RETRIEVED DOCUMENT — validate what comes back from your vector DB
# ---------------------------------------------------------------------------

class RetrievedDocument(BaseModel):
    """
    Schema for a document chunk returned by your retriever.

    WHY: Malicious content can live in your knowledge base too.
    Validate the document *before* stuffing it into your prompt.
    """

    doc_id: str
    content: Annotated[str, Field(max_length=2000)]
    source_url: Annotated[str, Field(max_length=300)]
    similarity_score: Annotated[float, Field(ge=0.0, le=1.0)]
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def no_embedded_instructions(cls, v: str) -> str:
        """
        Detects if a retrieved doc itself contains injection-style text.
        This is the 'indirect prompt injection' vector — an attacker
        poisons your knowledge base to hijack your LLM.
        """
        suspicious = [
            r"ignore\s+previous",
            r"new\s+instruction",
            r"system\s*:",
            r"</?(system|prompt|instruction)>",
        ]
        for pattern in suspicious:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(
                    f"Document contains suspicious instruction-like content: '{pattern}'. "
                    "Possible indirect prompt injection in knowledge base."
                )
        return v

    @field_validator("source_url")
    @classmethod
    def safe_url(cls, v: str) -> str:
        if not v.startswith(("https://", "http://", "internal://")):
            raise ValueError("Document source URL must use a known scheme.")
        return v

    @field_validator("similarity_score")
    @classmethod
    def minimum_relevance(cls, v: float) -> float:
        if v < 0.6:
            raise ValueError(
                f"Similarity score {v:.2f} is below the 0.60 threshold. "
                "Document may be irrelevant and add hallucination risk."
            )
        return v


# ---------------------------------------------------------------------------
# 3. PROMPT PAYLOAD — what actually gets sent to the LLM
# ---------------------------------------------------------------------------

class PromptPayload(BaseModel):
    """
    The final assembled prompt before it reaches the LLM.

    WHY: One last checkpoint. Even if individual pieces passed,
    the *combined* payload could exceed limits or smuggle context.
    """

    system_prompt: Annotated[str, Field(max_length=1000)]
    context_chunks: list[str] = Field(max_length=5)   # max 5 retrieved chunks
    user_question: str
    max_tokens: Annotated[int, Field(ge=50, le=2048)] = 512

    @model_validator(mode="after")
    def total_token_budget(self) -> "PromptPayload":
        """
        Rough token estimate (1 token ≈ 4 chars) to prevent runaway costs
        and context-stuffing attacks.
        """
        total_chars = (
            len(self.system_prompt)
            + sum(len(c) for c in self.context_chunks)
            + len(self.user_question)
        )
        estimated_tokens = total_chars // 4
        if estimated_tokens + self.max_tokens > 4096:
            raise ValueError(
                f"Total estimated tokens ({estimated_tokens} input + {self.max_tokens} output) "
                "exceeds the 4096 budget. Reduce context or max_tokens."
            )
        return self


# ---------------------------------------------------------------------------
# 4. LLM RESPONSE — validate what comes *back* from the model
# ---------------------------------------------------------------------------

class LLMResponse(BaseModel):
    """
    Validates and sanitises the raw LLM output before returning to the user.

    WHY: Models can be manipulated mid-conversation to leak system prompts
    or output harmful content. Output guardrails are just as important as input.
    """

    answer: Annotated[str, Field(min_length=1, max_length=3000)]
    model_id: str
    tokens_used: Annotated[int, Field(ge=0)]
    finish_reason: str

    @field_validator("answer")
    @classmethod
    def no_system_prompt_leak(cls, v: str) -> str:
        """Block responses that appear to echo back the system prompt."""
        leak_signals = [
            r"you are an? (ai|assistant|language model)",
            r"my instructions? (say|state|tell)",
            r"as per (my|the) system prompt",
        ]
        for pattern in leak_signals:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(
                    "Response may contain system prompt content. Blocked for safety."
                )
        return v

    @field_validator("finish_reason")
    @classmethod
    def expected_finish(cls, v: str) -> str:
        allowed = {"stop", "length", "end_turn"}
        if v not in allowed:
            raise ValueError(f"Unexpected finish_reason '{v}'. Expected one of {allowed}.")
        return v
