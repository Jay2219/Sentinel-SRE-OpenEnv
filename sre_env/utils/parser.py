import json
import re


def extract_json(text: str) -> dict:
    """Finds the first { and last } to extract JSON from conversational filler."""
    try:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            clean_content = match.group(1)
            clean_content = clean_content.replace("```json", "").replace("```", "")
            return json.loads(clean_content)
        return json.loads(text)
    except Exception:
        raise ValueError(f"Failed to parse LLM response as JSON. Content: {text[:100]}...")
