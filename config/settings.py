"""
Application Configuration and Settings.

Manages environment variables, default model settings, output style templates,
and pipeline stage definitions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Automatically load environment variables from .env file
# Looks in project root and current working directory
load_dotenv(override=False)


class StageStatus(str, Enum):
    """Execution status for each pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStage(str, Enum):
    """The 5 discrete stages of the LLM pipeline."""
    INPUT_PROCESSING = "Stage 1: Input Processing"
    PROMPT_ENGINEERING = "Stage 2: Prompt Engineering"
    LLM_INTEGRATION = "Stage 3: LLM Integration"
    RESPONSE_PROCESSING = "Stage 4: Response Processing"
    OUTPUT_GENERATION = "Stage 5: Output Generation"


# Pre-configured output styles with tailored system instructions
OUTPUT_STYLES: Dict[str, Dict[str, str]] = {
    "Concise & Direct": {
        "description": "Short, precise answers without fluff or conversational filler.",
        "instruction": "Be concise, direct, and fact-focused. Avoid conversational preamble, filler, or unnecessary repetition. Provide clear answers in minimal sentences.",
    },
    "Detailed & Analytical": {
        "description": "In-depth breakdown with reasoning, nuance, and structural analysis.",
        "instruction": "Provide a thorough, structured, and analytical response. Break down complex concepts into logical sections with clear explanations, trade-offs, and reasoning.",
    },
    "Bullet Points & Summary": {
        "description": "Structured lists with key takeaways highlighted for quick skimming.",
        "instruction": "Format the entire response using structured bullet points. Group related points under bold headings, and conclude with a concise summary takeaway.",
    },
    "Beginner-Friendly Explanation": {
        "description": "Simple language, intuitive analogies, and accessible explanations.",
        "instruction": "Explain concepts in simple, accessible terms as if teaching a curious beginner. Use real-world analogies, define technical jargon clearly, and maintain an encouraging tone.",
    },
    "Technical & Code-Oriented": {
        "description": "Developer-centric with code snippets, technical accuracy, and best practices.",
        "instruction": "Provide technical, implementation-focused guidance. Include clean, idiomatic code examples with explanations where relevant, and highlight production best practices and edge cases.",
    },
    "Creative & Engaging": {
        "description": "Vivid storytelling, engaging narrative, and expressive presentation.",
        "instruction": "Adopt an engaging, lively, and creative tone. Use expressive metaphors, captivating phrasing, and compelling narrative flow while delivering accurate substance.",
    },
}

# Curated list of popular OpenAI and OpenAI-compatible models
SUPPORTED_MODELS: List[str] = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-3.5-turbo",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "deepseek-chat",
    "deepseek-reasoner",
    "mistral-large-latest",
]

# Sample inputs for testing and interactive demonstrations
SAMPLE_INPUTS: Dict[str, str] = {
    "Explain Quantum Computing": (
        "What is quantum computing, how do qubits differ from classical bits, "
        "and what are its most promising real-world applications in the next decade?"
    ),
    "Summarize Meeting Notes": (
        "Team Sync Notes:\n"
        "- Alice completed the authentication refactor and migrated to OAuth2 tokens.\n"
        "- Bob reported database connection timeouts on the reporting replica during peak hours.\n"
        "- Charlie proposed adding Redis caching for user profile lookups; benchmarks show a 70% latency reduction.\n"
        "- Action Items: Bob to scale replica IOPS by Wednesday. Charlie to prepare caching PR for review. Alice to update API documentation."
    ),
    "Draft Product Launch Announcement": (
        "We are launching 'DataPulse', a real-time anomaly detection tool for microservices architectures. "
        "Key features include zero-config telemetry ingestion, AI-driven root-cause alerts, and Slack/PagerDuty integrations. "
        "Target audience is DevOps engineers and Site Reliability Engineers."
    ),
    "Debug Python Code": (
        "def compute_average(scores):\n"
        "    total = 0\n"
        "    for s in scores:\n"
        "        total += s\n"
        "    return total / len(scores)\n\n"
        "What happens when scores is empty or contains non-numeric values, and how should we make this robust?"
    ),
}


@dataclass
class AppConfig:
    """Application runtime configuration."""
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "").strip() or None
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    )
    min_input_length: int = 3
    max_input_length: int = 8000
    default_temperature: float = 0.7
    default_max_tokens: int = 1500
    request_timeout: int = 60

    def is_api_key_configured(self) -> bool:
        """Check if an API key has been supplied."""
        return bool(self.api_key and self.api_key.strip())


def get_config() -> AppConfig:
    """Retrieve the application configuration instance."""
    return AppConfig()
