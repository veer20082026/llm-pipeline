"""
Stage 4: Response Processing Module.

Receives raw LLM response, verifies response validity, cleans unnecessary
formatting artifacts, extracts useful insights, and computes presentation metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ResponseProcessingResult:
    """Represents the output of Stage 4: Response Processing."""
    is_valid: bool
    processed_response: str
    raw_response: str
    word_count: int
    character_count: int
    sentence_count: int
    estimated_reading_time_minutes: float
    code_blocks_count: int
    key_points: List[str]
    tokens_per_second: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization and logging."""
        return {
            "is_valid": self.is_valid,
            "processed_response": self.processed_response,
            "raw_response_length": len(self.raw_response),
            "processed_response_length": len(self.processed_response),
            "word_count": self.word_count,
            "character_count": self.character_count,
            "sentence_count": self.sentence_count,
            "estimated_reading_time_minutes": self.estimated_reading_time_minutes,
            "code_blocks_count": self.code_blocks_count,
            "key_points_count": len(self.key_points),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class ResponseProcessor:
    """
    Stage 4 Processor: Cleans raw LLM outputs, extracts structural metrics and takeaways.
    """

    def clean_response(self, text: str) -> str:
        """
        Sanitize and format response:
        - Strip extraneous leading/trailing whitespace.
        - Collapse excessive blank lines to a clean double newline.
        - Remove broken markdown or trailing whitespace on individual lines.
        """
        if not text:
            return ""

        # Normalize line endings
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        # Strip trailing spaces on each line
        cleaned_lines = [line.rstrip() for line in normalized.split("\n")]
        joined = "\n".join(cleaned_lines)

        # Normalize excessive blank lines (max 2 consecutive newlines)
        joined = re.sub(r"\n{3,}", "\n\n", joined)

        return joined.strip()

    def count_sentences(self, text: str) -> int:
        """Count approximate sentence boundaries."""
        if not text:
            return 0
        sentences = re.split(r"[.!?]+(?:\s+|$)", text.strip())
        return len([s for s in sentences if s.strip()])

    def count_code_blocks(self, text: str) -> int:
        """Count Markdown fenced code blocks (```)."""
        if not text:
            return 0
        # Count pairs of triple backticks
        matches = re.findall(r"```", text)
        return len(matches) // 2

    def extract_key_points(self, text: str, max_points: int = 5) -> List[str]:
        """
        Extract bullet points, numbered items, or prominent headings from the response.
        """
        if not text:
            return []

        points = []
        for line in text.splitlines():
            line_strip = line.strip()
            # Match bullets (- or * or •) or numbered lists (1. or 1))
            bullet_match = re.match(r"^(?:[\-\*•]|\d+[\.\)])\s+(.*)", line_strip)
            if bullet_match:
                point_text = bullet_match.group(1).strip()
                # Remove bold markers if wrapped
                clean_point = re.sub(r"^\*\*(.*?)\*\*", r"\1", point_text)
                if len(clean_point) > 5 and clean_point not in points:
                    points.append(clean_point)
                    if len(points) >= max_points:
                        break

        return points

    def process(
        self,
        raw_response: str,
        completion_tokens: int = 0,
        latency_seconds: float = 0.0,
    ) -> ResponseProcessingResult:
        """
        Execute Stage 4: Clean, validate, and compute analytics on LLM response.
        """
        raw = raw_response or ""

        # Handle empty response
        if not raw.strip():
            return ResponseProcessingResult(
                is_valid=False,
                processed_response="",
                raw_response=raw,
                word_count=0,
                character_count=0,
                sentence_count=0,
                estimated_reading_time_minutes=0.0,
                code_blocks_count=0,
                key_points=[],
                tokens_per_second=0.0,
                error_message="Received empty response content from LLM.",
            )

        cleaned = self.clean_response(raw)
        words = cleaned.split()
        word_count = len(words)
        char_count = len(cleaned)
        sentence_count = self.count_sentences(cleaned)
        code_blocks = self.count_code_blocks(cleaned)
        key_points = self.extract_key_points(cleaned)

        # Average reading speed: 200 words per minute
        reading_time = round(word_count / 200.0, 2)

        # Compute generation speed: completion tokens per second
        tps = 0.0
        if latency_seconds > 0 and completion_tokens > 0:
            tps = round(completion_tokens / latency_seconds, 2)

        return ResponseProcessingResult(
            is_valid=True,
            processed_response=cleaned,
            raw_response=raw,
            word_count=word_count,
            character_count=char_count,
            sentence_count=sentence_count,
            estimated_reading_time_minutes=reading_time,
            code_blocks_count=code_blocks,
            key_points=key_points,
            tokens_per_second=tps,
            metadata={
                "has_code": code_blocks > 0,
                "has_bullet_points": len(key_points) > 0,
            },
        )
