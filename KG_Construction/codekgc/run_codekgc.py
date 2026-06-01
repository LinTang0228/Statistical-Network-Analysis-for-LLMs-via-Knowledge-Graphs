"""
CodeKGC benchmark runner.

Usage:
    export OPENAI_API_KEY=...       # for --backbone gpt4
    export ANTHROPIC_API_KEY=...    # for --backbone claude
    python run_codekgc.py --backbone gpt4   --dataset ade
    python run_codekgc.py --backbone claude --dataset scierc
"""
from __future__ import annotations
import argparse
import json
import logging
import re
import sys
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "metrics"))
sys.path.insert(0, str(HERE))

from g_bert_score import compute_strict_f1, compute_hungarian_g_bert_score
from helpers import call_openai, call_claude, parallel_call

DATASETS = {
    "ade": {
        "test":   ROOT / "data" / "ade" / "test.json",
        "schema": HERE / "prompts" / "ade_schema.py",
        "icl":    HERE / "prompts" / "ade_icl.py",
        "strict_instruction":
            "\nExtract triples using only the provided relations: adverse effect. "
            "Do not use or invent other relations.",
    },
    "conll04": {
        "test":   ROOT / "data" / "conll04" / "test.json",
        "schema": HERE / "prompts" / "conll04_schema.py",
        "icl":    HERE / "prompts" / "conll04_icl.py",
        "strict_instruction":
            "\nExtract triples using only the provided relations: "
            "Work For, Located In, Organization based in, Live In, Kill. "
            "Do not use or invent other relations.",
    },
    "scierc": {
        "test":   ROOT / "data" / "scierc" / "test.json",
        "schema": HERE / "prompts" / "scierc_schema.py",
        "icl":    HERE / "prompts" / "scierc_icl.py",
        "strict_instruction":
            "\nExtract triples using only the provided relations: "
            "Used-for, Feature-of, Hyponym-of, Part-of, Conjunction, Compare, Evaluate-for. "
            "Output 'NULL' or empty list if no valid triples. "
            "Do not use or invent other relations. "
            "Include all valid triples but avoid redundancies. "
            "Use exact entity phrases from the text without combining, adding, or removing words.",
    },
}

BACKBONES = {
    "gpt4":   {"model": "gpt-4",                "env": "OPENAI_API_KEY",    "call": call_openai},
    "claude": {"model": "claude-sonnet-4-6",    "env": "ANTHROPIC_API_KEY", "call": call_claude},
}


def normalize_entity(text) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"^(a|an|the)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\.$", "", text)
    text = text.strip("'\"")
    return text


def normalize_relation(rel) -> str:
    return str(rel).strip().lower().replace(" ", "_")


def parse_triples(completion: str, schema_prompt: str):
    """Extract `extract = Extract([...])` from the model output, exec it inside the
    schema namespace, return the parsed (normalized) triple list."""
    match = re.search(r"extract\s*=\s*Extract\s*\(\s*(\[.*?\])\s*\)",
                      completion, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    code = f"extract = Extract({match.group(1)})"
    ns: dict = {}
    try:
        exec(schema_prompt + "\n" + code, ns)
    except Exception:
        return []
    extract = ns.get("extract")
    if not extract or not hasattr(extract, "triples"):
        return []
    out = []
    for t in extract.triples:
        h = normalize_entity(t.head.name)
        tail = normalize_entity(t.tail.name)
        r = normalize_relation(t.relation.name)
        if h and tail:
            out.append((h, r, tail))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(BACKBONES))
    ap.add_argument("--dataset",  required=True, choices=list(DATASETS))
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens",  type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="Concurrent LLM calls.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N items (smoke test).")
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    bb = BACKBONES[args.backbone]

    if not os.environ.get(bb["env"]):
        sys.exit(f"ERROR: set {bb['env']} in the environment.")

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.dataset}_{args.backbone}"
    log_path  = out_dir / f"codekgc_{tag}.log"
    score_path = out_dir / f"SCORES_codekgc_{tag}.txt"

    logging.basicConfig(level=logging.INFO, filename=str(log_path), filemode="w",
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"CodeKGC | dataset={args.dataset} | backbone={args.backbone} | "
                 f"model={bb['model']} | temperature={args.temperature} | max_tokens={args.max_tokens}")

    schema_prompt = ds["schema"].read_text()
    icl_prompt    = ds["icl"].read_text()
    data = json.load(open(ds["test"]))
    if args.limit:
        data = data[: args.limit]
    print(f"[{args.dataset}/{args.backbone}] {len(data)} items   model={bb['model']}")
    print(f"  logging to {log_path}")

    # Build prompts in input order
    prompts = []
    for item in data:
        text_prompt = '\n""" ' + item["text"] + ' """'
        prompts.append(schema_prompt + "\n" + icl_prompt + text_prompt + ds["strict_instruction"])

    # Parallel LLM calls
    completions = parallel_call(
        bb["call"], prompts,
        model=bb["model"], temperature=args.temperature,
        max_tokens=args.max_tokens, concurrency=args.concurrency,
    )

    # Parse + normalize + score
    item_details = []
    fail_count = 0
    for i, (item, (completion, err)) in enumerate(zip(data, completions), start=1):
        gold_raw = item.get("triple_list", [])
        gold = sorted({
            (normalize_entity(h), normalize_relation(r), normalize_entity(t))
            for h, r, t in gold_raw
            if normalize_entity(h) and normalize_entity(t)
        })
        if err:
            fail_count += 1
            logging.error(f"Item {i}: API call failed: {err}")
            pred = []
        else:
            pred = parse_triples(completion, schema_prompt)
        logging.info(f"Item {i} Gold: {gold}")
        logging.info(f"Item {i} Pred: {pred}")
        item_details.append({"index": i, "gold_triples": gold, "pred_triples": pred})

    print("scoring (this loads roberta-large, may take a moment)...")
    scores = {**compute_strict_f1(item_details), **compute_hungarian_g_bert_score(item_details)}
    summary = (
        f"[{args.dataset}/{args.backbone}] model={bb['model']}  "
        f"n={scores['n_items']}  gold={scores['gold_total']}  pred={scores['pred_total']}  "
        f"failures={fail_count}\n"
        f"  Strict      Micro F1 = {scores['strict_micro_f1']:.4f}   "
        f"Macro F1 = {scores['strict_macro_f1']:.4f}\n"
        f"  G-BERTScore Micro F1 = {scores['gbert_micro_f1']:.4f}   "
        f"Macro F1 = {scores['gbert_macro_f1']:.4f}\n"
    )
    print(summary)
    score_path.write_text(summary)
    print(f"  scores: {score_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
