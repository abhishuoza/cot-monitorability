"""Agent inference wrapper for vLLM OpenAI-compatible endpoints.

Returns thinking + final-answer text wrapped in the BEGIN/END tags the
monitor prompt templates in monitorability-evals expect.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class AgentResponse:
    final_answer: str
    action_only: str
    cot_only: str
    all_msgs: str


_FINAL_ANSWER_RE = re.compile(
    r"Final\s*Answer\s*:\s*\n?\s*(.+?)\s*\Z",
    re.IGNORECASE | re.DOTALL,
)


def _extract_final_answer(content: str) -> str:
    m = _FINAL_ANSWER_RE.search(content)
    if m:
        tail = m.group(1).strip().splitlines()
        if tail:
            return tail[-1].strip()
    for line in reversed(content.strip().splitlines()):
        line = line.strip()
        if line:
            return line
    return content.strip()


def call_agent(
    *,
    prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str | None = None,
    thinking: bool | None = None,
    max_tokens: int = 12288,
    temperature: float = 1.0,
) -> AgentResponse:
    client = OpenAI(base_url=base_url, api_key=api_key)
    extra_body: dict = {}
    if reasoning_effort is not None:
        extra_body["reasoning_effort"] = reasoning_effort
    if thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": thinking}

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra_body or None,
    )
    msg = resp.choices[0].message
    # vLLM 0.19.x exposes thinking on `reasoning`; 0.20.x on `reasoning_content`.
    cot = (
        getattr(msg, "reasoning_content", None)
        or getattr(msg, "reasoning", None)
        or ""
    ).strip()
    content = (msg.content or "").strip()
    final_answer = _extract_final_answer(content)

    cot_only = f"<BEGIN MODEL'S THINKING>\n{cot}\n<END MODEL'S THINKING>"
    action_only = (
        f"<BEGIN MODEL'S FINAL ANSWER>\n{content}\n<END MODEL'S FINAL ANSWER>"
    )
    all_msgs = cot_only + "\n" + action_only
    return AgentResponse(
        final_answer=final_answer,
        action_only=action_only,
        cot_only=cot_only,
        all_msgs=all_msgs,
    )
