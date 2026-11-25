"""
Stage 1: Input Processing Module.

Responsible for receiving raw user text, validating constraints,
cleaning unnecessary whitespace and artifacts, and calculating input statistics.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class InputProcessingResult:
    """Represents the output of Stage 1: Input Processing."""
    is_valid: bool
    cleaned_text: str
    original_text: str
    character_count: int
    word_count: int
    estimated_tokens: int
    validation_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization and logging."""
        return {
            "is_valid": self.is_valid,
            "cleaned_text": self.cleaned_text,
            "original_length": len(self.original_text),
            "cleaned_length": len(self.cleaned_text),
            "character_count": self.character_count,
            "word_count": self.word_count,
            "estimated_tokens": self.estimated_tokens,
            "validation_error": self.validation_error,
            "metadata": self.metadata,
        }


class InputProcessor:
    """
    Stage 1 Processor: Validates and sanitizes raw user inputs.
    """

    def __init__(self, min_length: int = 3, max_length: int = 8000) -> None:
        self.min_length = min_length
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize raw text:
        1. Normalize Unicode characters (NFKC).
        2. Remove non-printable control characters (except newline, tab).
        3. Normalize repeated whitespace within lines.
        4. Normalize excessive blank lines (max 2 consecutive newlines).
        5. Trim leading and trailing whitespace.
        """
        if not text:
            return ""

        # Normalize unicode characters
        normalized = unicodedata.normalize("NFKC", text)

        # Remove control characters except standard line breaks and tabs
        cleaned = "".join(
            ch for ch in normalized
            if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t"
        )

        # Standardize carriage returns
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse horizontal whitespace (spaces & tabs) per line while preserving line breaks
        cleaned_lines = []
        for line in cleaned.split("\n"):
            # Replace multiple horizontal spaces/tabs with a single space
            collapsed = re.sub(r"[ \t]+", " ", line).strip()
            cleaned_lines.append(collapsed)

        # Re-join lines and collapse more than 2 consecutive newlines to 2
        joined = "\n".join(cleaned_lines)
        joined = re.sub(r"\n{3,}", "\n\n", joined)

        return joined.strip()

    def validate(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Validate whether the text meets acceptable constraints.
        Returns (is_valid, error_message).
        """
        if text is None:
            return False, "Input cannot be None."

        stripped = text.strip()
        if not stripped:
            return False, "Input text cannot be empty or purely whitespace."

        if len(stripped) < self.min_length:
            return (
                False,
                f"Input is too short ({len(stripped)} chars). Minimum required is {self.min_length} characters.",
            )

        if len(stripped) > self.max_length:
            return (
                False,
                f"Input exceeds maximum length ({len(stripped)} chars). Maximum allowed is {self.max_length} characters.",
            )

        return True, None

    def process(self, raw_input: str) -> InputProcessingResult:
        """
        Execute Stage 1 processing:
        Validates, cleans, computes metrics, and prepares for Stage 2.
        """
        raw_text = raw_input or ""
        is_valid, error = self.validate(raw_text)

        if not is_valid:
            return InputProcessingResult(
                is_valid=False,
                cleaned_text="",
                original_text=raw_text,
                character_count=len(raw_text),
                word_count=0,
                estimated_tokens=0,
                validation_error=error,
            )

        cleaned = self.clean_text(raw_text)

        # Re-check after cleaning
        is_valid_cleaned, error_cleaned = self.validate(cleaned)
        if not is_valid_cleaned:
            return InputProcessingResult(
                is_valid=False,
                cleaned_text="",
                original_text=raw_text,
                character_count=len(cleaned),
                word_count=0,
                estimated_tokens=0,
                validation_error=error_cleaned,
            )

        words = cleaned.split()
        word_count = len(words)
        char_count = len(cleaned)
        # Standard heuristic: ~1 token per 4 characters or ~1.3 tokens per word
        estimated_tokens = max(1, int(word_count * 1.3))

        metadata = {
            "characters_removed": len(raw_text) - len(cleaned),
            "line_count": len(cleaned.splitlines()),
        }

        return InputProcessingResult(
            is_valid=True,
            cleaned_text=cleaned,
            original_text=raw_text,
            character_count=char_count,
            word_count=word_count,
            estimated_tokens=estimated_tokens,
            metadata=metadata,
        )
