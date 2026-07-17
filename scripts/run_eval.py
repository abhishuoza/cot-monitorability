"""End-to-end runner.

Monkey-patches the openai/monitorability-evals scaffold to call real agent
and monitor endpoints (vLLM OpenAI-compatible) instead of its mocks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD = ROOT / "external" / "monitorability-evals"
sys.path.insert(0, str(SCAFFOLD))
sys.path.insert(0, str(ROOT))

import run_eval_scaffold_lib as scaffold  # noqa: E402

from inference.agent import call_agent  # noqa: E402
from inference.monitor import call_monitor  # noqa: E402


class RealModel(scaffold.MockModel):
    def __init__(
        self,
        seed: int,
        *,
        base_url: str,
        api_key: str,
        model: str,
        reasoning_effort: str | None,
        thinking: bool | None,
    ):
        super().__init__(seed)
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking

    def call_model(self, item):
        prompt = scaffold._render_model_prompt(item)
        out = call_agent(
            prompt=prompt,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            thinking=self.thinking,
        )
        return scaffold.ModelCallOutput(
            final_answer=out.final_answer,
            action_only=out.action_only,
            cot_only=out.cot_only,
            all_msgs=out.all_msgs,
        )


class RealGraders(scaffold.MockGraders):
    def __init__(
        self,
        seed: int,
        *,
        base_url: str,
        api_key: str,
        model: str,
    ):
        super().__init__(seed)
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def grade_z(self, item, scope, y, model_out):
        rendered_prompt = scaffold._render_monitor_prompt(item, scope, model_out)
        valid_answers, trigger_answers = scaffold._monitor_answer_spec(item.z_rule_ref)
        letter, _raw = call_monitor(
            rendered_prompt=rendered_prompt,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )
        monitor_answer = letter if letter in valid_answers else valid_answers[-1]
        z = int(monitor_answer in trigger_answers)
        return z, monitor_answer, rendered_prompt


def _run_one_sample(
    item_idx: int,
    item,
    sample_idx: int,
    model,
    graders,
):
    row = {
        "dataset": item.dataset,
        "archetype": item.archetype,
        "instance_id": item.instance_id,
        "category": item.category,
        "sample_idx": sample_idx,
        "x": item.arm if item.arm is not None else np.nan,
        "agent_latency_s": np.nan,
        "monitor_latency_s": np.nan,
        "error": None,
        "_order": (item_idx, sample_idx),
    }
    try:
        t0 = time.perf_counter()
        model_out = model.call_model(item)
        row["agent_latency_s"] = time.perf_counter() - t0

        y = graders.grade_y(item, model_out)

        z_by_scope: dict[str, int] = {}
        monitor_answers: dict[str, str] = {}
        rendered_monitor_prompts: dict[str, str] = {}
        t1 = time.perf_counter()
        for scope in scaffold._scopes():
            z, m_ans, rendered_prompt = graders.grade_z(item, scope, y, model_out)
            z_by_scope[scope] = z
            monitor_answers[scope] = m_ans
            rendered_monitor_prompts[scope] = rendered_prompt
        row["monitor_latency_s"] = time.perf_counter() - t1

        row.update({
            "y": y,
            "final_answer": model_out.final_answer,
            "cot_only": model_out.cot_only,
            "action_only": model_out.action_only,
            "all_msgs": model_out.all_msgs,
            "z_cot_only": z_by_scope["cot_only"],
            "z_answer_only": z_by_scope["answer_only"],
            "z_all_messages": z_by_scope["all_messages"],
            "rendered_model_prompt": scaffold._render_model_prompt(item),
            "monitor_answer_cot_only": monitor_answers["cot_only"],
            "monitor_answer_answer_only": monitor_answers["answer_only"],
            "monitor_answer_all_messages": monitor_answers["all_messages"],
            "monitor_prompt_rendered_cot_only": rendered_monitor_prompts["cot_only"],
            "monitor_prompt_rendered_answer_only": rendered_monitor_prompts["answer_only"],
            "monitor_prompt_rendered_all_messages": rendered_monitor_prompts["all_messages"],
            "monitor_prompt_template_path": item.monitor_prompt_template_path,
            "model_prompt_refs": json.dumps(item.model_prompt_refs),
            "reference_answer": item.answer,
        })
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row.setdefault("y", np.nan)
        for k in ("z_cot_only", "z_answer_only", "z_all_messages"):
            row.setdefault(k, np.nan)
        for k in (
            "final_answer", "cot_only", "action_only", "all_msgs",
            "rendered_model_prompt",
            "monitor_answer_cot_only", "monitor_answer_answer_only", "monitor_answer_all_messages",
            "monitor_prompt_rendered_cot_only", "monitor_prompt_rendered_answer_only", "monitor_prompt_rendered_all_messages",
            "monitor_prompt_template_path", "model_prompt_refs", "reference_answer",
        ):
            row.setdefault(k, "")
    return row


def _make_concurrent_runner(model_factory, graders_factory, concurrency: int):
    def _run(items, samples_per_row, seed):
        model = model_factory(seed)
        graders = graders_factory(seed)
        tasks = [
            (i, item, s)
            for i, item in enumerate(items)
            for s in range(samples_per_row)
        ]
        rows: list[dict] = []
        failures = 0
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [
                ex.submit(_run_one_sample, i, item, s, model, graders)
                for (i, item, s) in tasks
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="eval", smoothing=0.05):
                r = fut.result()
                if r.get("error"):
                    failures += 1
                rows.append(r)
        if failures:
            print(f"WARNING: {failures}/{len(tasks)} tasks failed (see 'error' column)", flush=True)
        rows.sort(key=lambda r: r.pop("_order"))
        return pd.DataFrame(rows)
    return _run


def _none_or(s: str | None) -> str | None:
    if s is None:
        return None
    return None if s.strip().lower() in ("", "none", "null") else s


def _opt_bool(s: str | None) -> bool | None:
    if s is None:
        return None
    v = s.strip().lower()
    if v in ("", "none", "null"):
        return None
    return v in ("1", "true", "yes", "y", "on")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run monitorability eval against real vLLM endpoints.")
    p.add_argument("--dataset", default="daft", help="Substring filter on dataset path (e.g. 'daft').")
    p.add_argument("--samples-per-row", type=int, default=2)
    p.add_argument("--max-rows-per-dataset", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=8, help="In-flight (item, sample) tasks. vLLM batches up to --max-num-seqs.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bootstrap", type=int, default=20)
    p.add_argument("--selection-frac", type=float, default=0.6)
    p.add_argument("--group-keep-frac", type=float, default=1.0)
    p.add_argument("--wald-k", type=float, default=1.96)
    p.add_argument("--output-dir", default=str(ROOT / "results" / "smoke"))

    p.add_argument("--agent-base-url", default=os.environ.get("AGENT_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--agent-api-key", default=os.environ.get("AGENT_API_KEY", "EMPTY"))
    p.add_argument("--agent-model", default=os.environ.get("AGENT_MODEL", "openai/gpt-oss-20b"))
    p.add_argument("--reasoning-effort", default=os.environ.get("AGENT_REASONING_EFFORT", "medium"))
    p.add_argument("--thinking", default=os.environ.get("AGENT_THINKING"))

    p.add_argument("--monitor-base-url", default=os.environ.get("MONITOR_BASE_URL", "http://localhost:8001/v1"))
    p.add_argument("--monitor-api-key", default=os.environ.get("MONITOR_API_KEY", "EMPTY"))
    p.add_argument("--monitor-model", default=os.environ.get("MONITOR_MODEL", "google/gemma-4-31B-it"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reasoning_effort = _none_or(args.reasoning_effort)
    thinking = _opt_bool(args.thinking)

    def _make_model(seed):
        return RealModel(
            seed,
            base_url=args.agent_base_url,
            api_key=args.agent_api_key,
            model=args.agent_model,
            reasoning_effort=reasoning_effort,
            thinking=thinking,
        )

    def _make_graders(seed):
        return RealGraders(
            seed,
            base_url=args.monitor_base_url,
            api_key=args.monitor_api_key,
            model=args.monitor_model,
        )

    scaffold.MockModel = _make_model      # type: ignore[assignment]
    scaffold.MockGraders = _make_graders  # type: ignore[assignment]
    scaffold.sample_and_grade_rows = _make_concurrent_runner(  # type: ignore[assignment]
        _make_model, _make_graders, args.concurrency
    )

    items = scaffold.load_items(args.max_rows_per_dataset)
    if args.dataset:
        items = [it for it in items if args.dataset in it.dataset]
    if not items:
        raise SystemExit(f"No items match --dataset={args.dataset!r}")
    print(f"Loaded {len(items)} items across {len({it.dataset for it in items})} dataset(s).", flush=True)

    df = scaffold.sample_and_grade_rows(items, args.samples_per_row, args.seed)
    metrics = {
        "intervention": scaffold.compute_intervention_metrics(df, args),
        "process": scaffold.compute_process_metrics(df, args),
        "outcome": scaffold.compute_outcome_metrics(df, args),
    }
    scaffold.write_outputs(df, metrics, args)
    print(f"Wrote outputs to {Path(args.output_dir).resolve()}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
