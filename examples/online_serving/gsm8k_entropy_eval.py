#!/usr/bin/env python3
"""Evaluate a vLLM entropy-enabled server on a GSM8K subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset


SYSTEM_PROMPT = """You are a careful grade-school mathematics solver.
Show the calculation briefly. End with exactly: Final answer: <number>
Do not put anything after the final answer."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("gsm8k_entropy_results"))
    parser.add_argument("--api-key", default="token-local")
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Request failed after 3 attempts: {exc}") from exc
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def extract_gold(answer: str) -> str:
    match = re.search(r"####\s*([^\n]+)", answer)
    if not match:
        raise ValueError(f"Could not extract GSM8K answer from: {answer!r}")
    return match.group(1).strip()


def extract_prediction(text: str) -> str | None:
    matches = re.findall(
        r"final\s+answer\s*:\s*([-+]?[$]?[\d,]+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return matches[-1]
    # Fallback for responses that ignore the requested format.
    numbers = re.findall(r"[-+]?[$]?[\d,]+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def as_decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    cleaned = value.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def is_correct(prediction: str | None, gold: str) -> bool:
    pred_num = as_decimal(prediction)
    gold_num = as_decimal(gold)
    return pred_num is not None and gold_num is not None and pred_num == gold_num


def finite_stats(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def normalized_curve(values: list[float], points: int = 50) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.full(points, np.nan)
    if array.size == 1:
        return np.full(points, array[0])
    source_x = np.linspace(0.0, 1.0, array.size)
    target_x = np.linspace(0.0, 1.0, points)
    return np.interp(target_x, source_x, array)


def create_plots(records: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    curves = np.stack([normalized_curve(r["token_entropy"]) for r in records])
    correct_mask = np.asarray([r["correct"] for r in records], dtype=bool)
    x = np.linspace(0, 100, curves.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for mask, label, color in (
        (correct_mask, "correct", "tab:green"),
        (~correct_mask, "incorrect", "tab:red"),
    ):
        if not mask.any():
            continue
        group = curves[mask]
        mean = np.nanmean(group, axis=0)
        sem = np.nanstd(group, axis=0) / math.sqrt(group.shape[0])
        axes[0].plot(x, mean, label=f"{label} (n={group.shape[0]})", color=color)
        axes[0].fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)

    axes[0].set(
        title="Entropy over normalized answer progress",
        xlabel="Generation progress (%)",
        ylabel="Token entropy (nats)",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    means = np.asarray([r["entropy_stats"]["mean"] for r in records], dtype=float)
    order = np.argsort(means)
    colors = ["tab:green" if records[i]["correct"] else "tab:red" for i in order]
    axes[1].bar(np.arange(len(records)), means[order], color=colors)
    axes[1].set(
        title="Mean generation entropy per GSM8K problem",
        xlabel="Sample (sorted by mean entropy)",
        ylabel="Mean token entropy (nats)",
    )
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    results_path.write_text("", encoding="utf-8")

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    records: list[dict[str, Any]] = []

    for index, example in enumerate(dataset):
        started = time.perf_counter()
        response = post_json(
            endpoint,
            {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["question"]},
                ],
                "temperature": args.temperature,
                "max_completion_tokens": args.max_tokens,
                "return_entropy": True,
                "return_token_ids": True,
                "seed": args.seed + index,
            },
            args.api_key,
        )

        choice = response["choices"][0]
        generated = choice["message"].get("content") or ""
        entropies = choice.get("token_entropy")
        token_ids = choice.get("token_ids")
        if entropies is None:
            raise RuntimeError(
                "The server returned no token_entropy. Verify that the server is "
                "running from the vllm-token-entropy checkout."
            )
        if token_ids is not None and len(token_ids) != len(entropies):
            raise RuntimeError(
                f"Token/entropy length mismatch: {len(token_ids)} != {len(entropies)}"
            )

        gold = extract_gold(example["answer"])
        prediction = extract_prediction(generated)
        record = {
            "index": index,
            "question": example["question"],
            "gold_answer": gold,
            "predicted_answer": prediction,
            "correct": is_correct(prediction, gold),
            "generated_text": generated,
            "token_ids": token_ids,
            "token_entropy": entropies,
            "entropy_stats": finite_stats(entropies),
            "latency_seconds": time.perf_counter() - started,
            "finish_reason": choice.get("finish_reason"),
        }
        records.append(record)
        print(
            f"[{index + 1:02d}/{args.num_samples}] "
            f"correct={record['correct']} gold={gold} pred={prediction} "
            f"tokens={len(entropies)} mean_H={record['entropy_stats']['mean']:.3f}"
        )

        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    accuracy = sum(r["correct"] for r in records) / len(records)
    correct_means = [
        r["entropy_stats"]["mean"] for r in records if r["correct"]
    ]
    incorrect_means = [
        r["entropy_stats"]["mean"] for r in records if not r["correct"]
    ]
    summary = {
        "model": args.model,
        "num_samples": len(records),
        "seed": args.seed,
        "temperature": args.temperature,
        "accuracy": accuracy,
        "correct_count": sum(r["correct"] for r in records),
        "mean_entropy_correct": finite_stats(correct_means)["mean"],
        "mean_entropy_incorrect": finite_stats(incorrect_means)["mean"],
        "mean_entropy_all": finite_stats(
            [r["entropy_stats"]["mean"] for r in records]
        )["mean"],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "correct",
                "gold_answer",
                "predicted_answer",
                "num_tokens",
                "mean_entropy",
                "std_entropy",
                "latency_seconds",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "index": record["index"],
                    "correct": record["correct"],
                    "gold_answer": record["gold_answer"],
                    "predicted_answer": record["predicted_answer"],
                    "num_tokens": len(record["token_entropy"]),
                    "mean_entropy": record["entropy_stats"]["mean"],
                    "std_entropy": record["entropy_stats"]["std"],
                    "latency_seconds": record["latency_seconds"],
                }
            )

    create_plots(records, args.output_dir / "entropy_analysis.png")
    print(json.dumps(summary, indent=2))
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
