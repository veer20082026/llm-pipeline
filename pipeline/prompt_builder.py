"""
Stage 2: Prompt Engineering Module.

Constructs structured, persona-aware, style-guided prompts formatted
for chat-based Large Language Models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.settings import OUTPUT_STYLES


@dataclass
class PromptEngineeringResult:
    """Represents the output of Stage 2: Prompt Engineering."""
    system_prompt: str
    user_prompt: str
    messages: List[Dict[str, str]]
    style_applied: str
    persona_applied: str
    total_prompt_length: int
    estimated_prompt_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization and logging."""
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "messages": self.messages,
            "style_applied": self.style_applied,
            "persona_applied": self.persona_applied,
            "total_prompt_length": self.total_prompt_length,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "metadata": self.metadata,
        }


# Standard AI Personas
PERSONAS: Dict[str, str] = {
    "General Assistant": "You are a helpful, knowledgeable, and honest AI assistant.",
    "Senior Software Engineer": "You are a senior software architect and tech lead with deep expertise in systems design, clean code, and production reliability.",
    "Academic Research Scientist": "You are an expert research scientist with rigorous analytical thinking, clear terminology, and evidence-based explanations.",
    "Executive Product Manager": "You are an experienced VP of Product focusing on user value, clear communication, strategic priorities, and actionable roadmaps.",
    "Patient Educator": "You are an inspiring, patient educator who excels at breaking down complex concepts using intuitive mental models.",
}


class PromptBuilder:
    """
    Stage 2 Builder: Compiles system prompts, user inputs, styles, and task context.
    """

    def __init__(self) -> None:
        self.available_styles = OUTPUT_STYLES
        self.available_personas = PERSONAS

    def build_system_prompt(
        self,
        style_name: str = "Concise & Direct",
        persona_name: str = "General Assistant",
        custom_system_instruction: Optional[str] = None,
    ) -> str:
        """
        Assemble the system instruction including persona, style, and guardrails.
        """
        # Persona
        persona = self.available_personas.get(
            persona_name,
            self.available_personas["General Assistant"]
        )

        # Style instruction
        style_entry = self.available_styles.get(
            style_name,
            self.available_styles["Concise & Direct"]
        )
        style_instruction = style_entry["instruction"]

        # Base system guidelines
        base_guidelines = (
            "Core Guidelines:\n"
            "- Strive for factual accuracy and intellectual honesty.\n"
            "- If you are uncertain about a fact, acknowledge the uncertainty.\n"
            "- Follow formatting requirements strictly."
        )

        parts = [
            f"Persona:\n{persona}",
            f"Style & Tone Instructions:\n{style_instruction}",
            base_guidelines,
        ]

        if custom_system_instruction and custom_system_instruction.strip():
            parts.append(f"Additional Instructions:\n{custom_system_instruction.strip()}")

        return "\n\n".join(parts)

    def build_user_prompt(self, cleaned_input: str, task_context: Optional[str] = None) -> str:
        """
        Format user input with structural delimiters to prevent prompt injection
        and ensure clear instruction boundaries.
        """
        if task_context and task_context.strip():
            return (
                f"Context / Task Goal:\n{task_context.strip()}\n\n"
                f"User Request / Input:\n```\n{cleaned_input}\n```\n\n"
                f"Please address the user's request according to the established instructions."
            )

        return (
            f"User Request / Input:\n```\n{cleaned_input}\n```\n\n"
            f"Please address the user's request thoroughly according to the established instructions."
        )

    def build(
        self,
        cleaned_input: str,
        style_name: str = "Concise & Direct",
        persona_name: str = "General Assistant",
        custom_system_instruction: Optional[str] = None,
        task_context: Optional[str] = None,
    ) -> PromptEngineeringResult:
        """
        Execute Stage 2: Prompt Engineering.
        Generates structured chat completion messages.
        """
        system_prompt = self.build_system_prompt(
            style_name=style_name,
            persona_name=persona_name,
            custom_system_instruction=custom_system_instruction,
        )

        user_prompt = self.build_user_prompt(
            cleaned_input=cleaned_input,
            task_context=task_context,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        total_length = len(system_prompt) + len(user_prompt)
        # Rough token estimate: ~1 token per 4 chars
        estimated_tokens = max(1, total_length // 4)

        metadata = {
            "style_description": self.available_styles.get(style_name, {}).get("description", ""),
            "message_count": len(messages),
        }

        return PromptEngineeringResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            messages=messages,
            style_applied=style_name,
            persona_applied=persona_name,
            total_prompt_length=total_length,
            estimated_prompt_tokens=estimated_tokens,
            metadata=metadata,
        )
