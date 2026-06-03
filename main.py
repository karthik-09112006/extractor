"""CLI entrypoint for testing the SkillExtractorAgent asynchronously."""

from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import os
from collections import defaultdict
from typing import Any

from src.agent import SkillExtractorAgent

SAMPLE_JOB_DESCRIPTION = """
Senior Full-Stack Developer (React + Node.js + AWS)

We're scaling a fast-moving product team and looking for a pragmatic full-stack engineer
who can own features end-to-end. You'll collaborate closely with product, design, and QA
to ship customer-facing improvements every sprint.

Requirements:
- 4+ years building modern web apps using React and Node.js/Express.
- Strong JavaScript/TypeScript fundamentals and hands-on REST API design.
- Experience deploying and operating apps on AWS (EC2, S3, Lambda preferred).
- Working knowledge of SQL/NoSQL databases (PostgreSQL, MongoDB).
- Comfortable with Git, CI/CD pipelines, and Agile/Scrum ceremonies.
- Great written and verbal communication; can explain technical ideas clearly.
- Team player with strong problem-solving mindset and ability to mentor juniors.

Nice to have:
- Familiarity with Docker, Kubernetes, and testing frameworks (Jest/Cypress).
- Exposure to observability tools and production monitoring.
""".strip()


def _colorize(text: str, color: str) -> str:
    """Return colored text when colorama is available."""
    if not color:
        return text
    return f"{color}{text}{Style.RESET_ALL}"


try:
    from colorama import Fore, Style, init

    init(autoreset=True)
except ImportError:
    class _NoColor:
        BLACK = ""
        BLUE = ""
        CYAN = ""
        GREEN = ""
        MAGENTA = ""
        RED = ""
        WHITE = ""
        YELLOW = ""
        RESET_ALL = ""

    Fore = _NoColor()
    Style = _NoColor()


def _print_header(title: str) -> None:
    print(_colorize(f"\n=== {title} ===", Fore.CYAN))


def _print_extraction(extraction: Any) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for skill in extraction.skills:
        grouped[skill.category].append(
            {"name": skill.name, "context_snippet": skill.context_snippet}
        )

    _print_header("Inferred Role")
    print(_colorize(extraction.primary_role_inferred, Fore.GREEN))

    _print_header("Confidence")
    print(_colorize(f"{extraction.confidence_score:.2%}", Fore.YELLOW))

    _print_header("Skills by Category")
    for category in ("Technical", "Tool/Framework", "Soft Skill"):
        items = grouped.get(category, [])
        if not items:
            continue
        print(_colorize(f"\n{category}:", Fore.MAGENTA))
        for idx, item in enumerate(items, start=1):
            print(f"  {idx}. {_colorize(item['name'], Fore.BLUE)}")
            print(f"     context: {item['context_snippet']}")

    _print_header("Raw JSON")
    print(json.dumps(extraction.model_dump(), indent=2, ensure_ascii=False))


async def run() -> None:
    """Run extraction with a realistic hardcoded sample description."""
    if not os.getenv("GEMINI_API_KEY"):
        print(
            _colorize(
                "Error: GEMINI_API_KEY is not set. Export it and rerun `python main.py`.",
                Fore.RED,
            )
        )
        return

    _print_header("Sample Job Description")
    print(SAMPLE_JOB_DESCRIPTION)

    _print_header("Running Skill Extraction")
    agent = SkillExtractorAgent()
    extraction = await agent.extract_skills(SAMPLE_JOB_DESCRIPTION)
    _print_extraction(extraction)


def main() -> None:
    """Synchronous wrapper for async CLI execution."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
