"""
KGGen benchmark runner.

Usage:
    export OPENAI_API_KEY=...       # for --backbone gpt4
    export ANTHROPIC_API_KEY=...    # for --backbone claude
    python run_kggen.py --backbone gpt4   --dataset ade
    python run_kggen.py --backbone claude --dataset scierc
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "metrics"))
sys.path.insert(0, str(HERE))

from g_bert_score import compute_strict_f1, compute_hungarian_g_bert_score
from prompts import PROMPTS

DATASETS = {
    "ade":     ROOT / "data" / "ade"     / "test.json",
    "conll04": ROOT / "data" / "conll04" / "test.json",
    "scierc":  ROOT / "data" / "scierc"  / "test.json",
}

BACKBONES = {
    "gpt4":   {"model": "openai/gpt-4",                 "env": "OPENAI_API_KEY"},
    "claude": {"model": "anthropic/claude-sonnet-4-6",  "env": "ANTHROPIC_API_KEY"},
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
    return str(rel).strip().lower().replace(" ", "-")


def normalize_triples(raw_triples):
    out = set()
    for t in raw_triples:
        if not hasattr(t, "__len__") or len(t) != 3:
            continue
        h, r, tail = t
        hn, tn, rn = normalize_entity(h), normalize_entity(tail), normalize_relation(r)
        if hn and tn:
            out.add((hn, rn, tn))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(BACKBONES))
    ap.add_argument("--dataset",  required=True, choices=list(DATASETS))
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--concurrency", type=int, default=2,
                    help="Concurrent KGGen calls. Each KGGen call internally makes "
                         "~2 LLM requests, so concurrency=2 ~ 4 requests in flight.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N items (smoke test).")
    args = ap.parse_args()

    backbone = BACKBONES[args.backbone]
    api_key = os.environ.get(backbone["env"])
    if not api_key:
        sys.exit(f"ERROR: set {backbone['env']} in the environment.")

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.dataset}_{args.backbone}"
    log_path  = out_dir / f"kggen_{tag}.log"
    pred_path = out_dir / f"kggen_{tag}_predictions.json"
    score_path = out_dir / f"SCORES_kggen_{tag}.txt"

    logging.basicConfig(level=logging.INFO, filename=str(log_path), filemode="w",
                        format="%(message)s")
    logging.info(f"KGGen | dataset={args.dataset} | backbone={args.backbone} | "
                 f"model={backbone['model']} | temperature={args.temperature}")

    from kg_gen import KGGen
    kg = KGGen(model=backbone["model"], temperature=args.temperature, api_key=api_key)

    data = json.load(open(DATASETS[args.dataset]))
    if args.limit:
        data = data[: args.limit]
    print(f"[{args.dataset}/{args.backbone}] {len(data)} items   model={backbone['model']}")
    print(f"  logging to {log_path}")

    context = PROMPTS[args.dataset]

    def _call(idx_text):
        i, text = idx_text
        try:
            response = kg.generate(input_data=text, context=context)
            raw = response.relations if response and hasattr(response, "relations") else []
            return i, list(raw), None
        except Exception as e:
            return i, [], f"{type(e).__name__}: {e}"

    inputs = [(i, item["text"]) for i, item in enumerate(data, start=1)]
    responses: dict[int, tuple] = {}
    n_done = 0
    t0 = time.time()
    print(f"  running with concurrency={args.concurrency}...")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(_call, x) for x in inputs]
        for f in as_completed(futures):
            i, raw, err = f.result()
            responses[i] = (raw, err)
            n_done += 1
            if n_done % 25 == 0 or n_done == len(inputs):
                rate = n_done / max(1e-9, time.time() - t0)
                print(f"    {n_done}/{len(inputs)}   ({rate:.1f} items/s)")

    item_details = []
    fail_count = 0
    for i, item in enumerate(data, start=1):
        text = item["text"]
        gold = normalize_triples(item.get("triple_list", []))
        raw, err = responses.get(i, ([], "no response captured"))
        logging.info(f"\n----- Item {i}/{len(data)} -----")
        logging.info(f"Text: {text}")
        if err:
            fail_count += 1
            logging.error(f"FAILED: {err}")
            pred = []
        else:
            logging.info(f"KGGen Extraction: {raw}")
            pred = normalize_triples(raw)
        logging.info(f"Gold (Normalized): {gold}")
        logging.info(f"Pred (Normalized): {pred}")
        item_details.append({"index": i, "text": text,
                             "gold_triples": gold, "pred_triples": pred})

    with open(pred_path, "w") as f:
        json.dump([{**d,
                    "gold_triples": [list(t) for t in d["gold_triples"]],
                    "pred_triples": [list(t) for t in d["pred_triples"]]}
                   for d in item_details], f, indent=2)
    print(f"  predictions: {pred_path}")

    print("scoring (this loads roberta-large, may take a moment)...")
    scores = {**compute_strict_f1(item_details), **compute_hungarian_g_bert_score(item_details)}
    summary = (
        f"[{args.dataset}/{args.backbone}] model={backbone['model']}  "
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
