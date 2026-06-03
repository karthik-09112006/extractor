
"""Data models for the AI Skill Extraction Agent."""

import logging
import os
from typing import List, Literal

from google import genai
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field


SkillCategory = Literal["Technical", "Soft Skill", "Tool/Framework"]


class Skill(BaseModel):
    """Represents one standardized skill extracted from a job description."""

    name: str = Field(
        ...,
        description=(
            "Standardized/canonical skill name inferred from the job description. "
            "Use normalized naming. Examples: 'Python', 'Communication', 'TensorFlow'."
        ),
        min_length=1,
    )
    category: SkillCategory = Field(
        ...,
        description=(
            "High-level skill category. Must be one of: "
            "'Technical', 'Soft Skill', or 'Tool/Framework'. "
            "Examples: 'Technical', 'Soft Skill', 'Tool/Framework'."
        ),
    )
    context_snippet: str = Field(
        ...,
        description=(
            "Short verbatim quote from the job description indicating where this skill was found. "
            "Keep concise and faithful to source text. "
            "Examples: 'Experience with Python and SQL in production environments'; "
            "'Strong written and verbal communication skills'."
        ),
        min_length=1,
    )


class JobDescriptionExtraction(BaseModel):
    """Top-level extraction output for a job description."""

    skills: List[Skill] = Field(
        ...,
        description=(
            "List of extracted skills identified in the job description. "
            "Include each distinct skill once using standardized naming."
        ),
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Overall confidence of the extraction result from 0.0 to 1.0, "
            "where 1.0 indicates highest confidence. Example: 0.92."
        ),
    )
    primary_role_inferred: str = Field(
        ...,
        description=(
            "Best-guess primary role title inferred from the job description "
            "(e.g., 'Data Scientist', 'Frontend Developer', 'Product Manager'). "
            "Example: 'Machine Learning Engineer'."
        ),
        min_length=1,
    )


class SkillExtractorAgent:
    """Agent responsible for extracting normalized skills from job text."""

    def __init__(self, client: genai.Client | None = None) -> None:
        self.logger = logging.getLogger(__name__)
        api_key = os.getenv("GEMINI_API_KEY")

        if client is not None:
            self.client = client
            return

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set it in your environment."
            )
        self.client = genai.Client(api_key=api_key)

    async def extract_skills(self, job_text: str) -> JobDescriptionExtraction:
        """Extract structured skills from a raw job description text."""
        if not job_text or not job_text.strip():
            raise ValueError("job_text must be a non-empty string.")

        system_prompt = (
            "You are an expert technical recruiter and talent intelligence analyst. "
            "Extract skills from the provided job description with high precision. "
            "Normalize all technology names to canonical forms (for example, "
            "'Amazon Web Services' -> 'AWS', 'JS' -> 'JavaScript', "
            "'Postgres' -> 'PostgreSQL'). "
            "Separate core programming languages from tools/frameworks/platforms. "
            "Only include skills supported by the source text and avoid speculation. "
            "For each skill, capture a short, accurate verbatim context_snippet taken "
            "from the job description. "
            "Return a clean structured response that strictly matches the provided schema."
        )

        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\nJob description:\n{job_text}"}],
                    }
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": JobDescriptionExtraction,
                },
            )

            parsed = response.parsed
            if parsed is None:
                self.logger.error("Model returned no parsed payload.")
                raise ValueError("Model did not return a valid structured extraction.")

            self.logger.info(
                "Extracted %d skills with confidence %.3f",
                len(parsed.skills),
                parsed.confidence_score,
            )
            return parsed
        except genai_errors.APIError as exc:
            self.logger.exception("Gemini API error during skill extraction: %s", exc)
            raise RuntimeError("Gemini API call failed while extracting skills.") from exc
        except Exception as exc:
            self.logger.exception("Unexpected error during skill extraction: %s", exc)
            raise
