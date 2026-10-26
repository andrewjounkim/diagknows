"""Optional: ask OpenAI for a plain-English explanation of a ranking.

Only used when OPENAI_API_KEY is set (e.g. in .env). Without a key, or if
the call fails, explain() returns None and the app carries on without it.
"""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gpt-5.4-mini"
MAX_EXAMPLES = 6

INSTRUCTIONS = (
    "You explain the output of an educational symptom-matching tool. "
    "Write 3-4 plain-English sentences explaining why the listed conditions came up, "
    "pointing to the specific symptoms the user's description shares with the retrieved "
    "patient descriptions. Use ONLY the user's description and the retrieved descriptions "
    "provided. Do not add outside medical facts, do not diagnose, and do not suggest treatments."
)


def is_enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def build_input(result: dict) -> str:
    lines = [f'User description: "{result["query"]}"', "", "Retrieved patient descriptions:"]
    examples = [
        (c["condition"], ex["text"])
        for c in result["conditions"]
        for ex in c["examples"]
    ][:MAX_EXAMPLES]
    for condition, text in examples:
        lines.append(f'- [{condition}] "{text}"')
    lines.append("")
    lines.append("Conditions ranked: " + ", ".join(c["condition"] for c in result["conditions"]))
    return "\n".join(lines)


def explain(result: dict) -> str | None:
    """Return a short explanation for an "ok" ranking result, or None if unavailable."""
    if result.get("status") != "ok" or not is_enabled():
        return None
    try:
        from openai import OpenAI

        client = OpenAI(timeout=30)  # reads OPENAI_API_KEY from the environment
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            instructions=INSTRUCTIONS,
            input=build_input(result),
            # Reasoning models count thinking tokens here too, so leave headroom.
            max_output_tokens=1500,
        )
        return response.output_text.strip() or None
    except Exception as e:  # any failure just means no explanation
        print(f"(AI explanation unavailable: {type(e).__name__})")
        return None
