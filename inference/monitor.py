"""Monitor inference wrapper for vLLM OpenAI-compatible endpoints.

Parses a single-letter verdict (A-D, X-Y) or TRUE/FALSE from the last
non-empty line of the monitor's response.
"""
from __future__ import annotations

import re

from openai import OpenAI


_LETTER_TAIL_RE = re.compile(r"\b([A-DXY])\b\s*\.?\s*$", re.IGNORECASE)


def _parse_verdict(content: str) -> str:
    for raw_line in reversed(content.strip().splitlines()):
        line = raw_line.strip().strip("*`'\"").rstrip(".").strip()
        if not line:
            continue
        up = line.upper()
        if up in ("TRUE", "FALSE"):
            return up
        if len(line) == 1 and up in "ABCDXY":
            return up
        m = _LETTER_TAIL_RE.search(up)
        if m:
            return m.group(1).upper()
    return ""


def call_monitor(
    *,
    rendered_prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> tuple[str, str]:
    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": rendered_prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    raw = (resp.choices[0].message.content or "").strip()
    return _parse_verdict(raw), raw
