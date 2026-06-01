"""
Strict F1 and Hungarian G-BERTScore for KG construction evaluation.

G-BERTScore is the Hungarian-matched variant from the ExplaGraphs formulation
(Saha et al., EMNLP 2021), computed with `roberta-large` embeddings.

Each item in `item_details` is a dict:
    {"gold_triples": [(h, r, t), ...], "pred_triples": [(h, r, t), ...]}
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

ROBERTA_MODEL = "roberta-large"


def _triple_to_str(triples):
    return [f"{h} {r} {t}" for (h, r, t) in triples]


def compute_strict_f1(item_details):
    """Exact-match Strict F1 (Micro)."""
    micro_tp = micro_fp = micro_fn = 0
    item_f1s = []
    for d in item_details:
        gold = set(map(tuple, d["gold_triples"]))
        pred = set(map(tuple, d["pred_triples"]))
        if not gold and not pred:
            item_f1s.append(1.0); continue
        if not gold or not pred:
            item_f1s.append(0.0)
            micro_fp += len(pred); micro_fn += len(gold)
            continue
        tp = len(gold & pred); fp = len(pred - gold); fn = len(gold - pred)
        micro_tp += tp; micro_fp += fp; micro_fn += fn
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        item_f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    if micro_tp + micro_fp == 0 or micro_tp + micro_fn == 0:
        micro_f1 = 0.0
    else:
        mp = micro_tp / (micro_tp + micro_fp)
        mr = micro_tp / (micro_tp + micro_fn)
        micro_f1 = 2 * mp * mr / (mp + mr) if (mp + mr) else 0.0
    return {
        "strict_micro_f1": micro_f1,
        "strict_macro_f1": float(np.mean(item_f1s)) if item_f1s else 0.0,
    }


def compute_hungarian_g_bert_score(item_details, idf=False):
    """
    Hungarian-matched G-BERTScore (ExplaGraphs, Saha et al. 2021).
    Reports Micro and Macro precision / recall / F1.
    """
    all_gold = [_triple_to_str(d["gold_triples"]) for d in item_details]
    all_pred = [_triple_to_str(d["pred_triples"]) for d in item_details]

    references, candidates, ref_cand_index = [], [], {}
    for graph_idx, (g_edges, p_edges) in enumerate(zip(all_gold, all_pred)):
        for ge in g_edges:
            for pe in p_edges:
                references.append(ge); candidates.append(pe)
                ref_cand_index[(graph_idx, ge, pe)] = len(references) - 1

    if not references:
        precisions = np.zeros(len(all_gold))
        recalls    = np.zeros(len(all_gold))
        f1s        = np.zeros(len(all_gold))
    else:
        from bert_score import score as bert_score_compute
        _, _, bs_F1 = bert_score_compute(
            candidates, references,
            lang="en", verbose=False, model_type=ROBERTA_MODEL, idf=idf,
        )
        bs_F1 = bs_F1.cpu().numpy()
        precisions, recalls, f1s = [], [], []
        for graph_idx, (g_edges, p_edges) in enumerate(zip(all_gold, all_pred)):
            ng, npred = len(g_edges), len(p_edges)
            if ng == 0 and npred == 0:
                precisions.append(1.0); recalls.append(1.0); f1s.append(1.0); continue
            if npred == 0:
                precisions.append(1.0); recalls.append(0.0); f1s.append(0.0); continue
            if ng == 0:
                precisions.append(0.0); recalls.append(1.0); f1s.append(0.0); continue
            score_matrix = np.zeros((ng, npred))
            for i, ge in enumerate(g_edges):
                for j, pe in enumerate(p_edges):
                    idx = ref_cand_index.get((graph_idx, ge, pe))
                    if idx is not None:
                        score_matrix[i, j] = bs_F1[idx]
            row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)
            matched = score_matrix[row_ind, col_ind]
            sp = matched.sum() / npred
            sr = matched.sum() / ng
            sf = 2 * sp * sr / (sp + sr) if (sp + sr) else 0.0
            precisions.append(sp); recalls.append(sr); f1s.append(sf)
        precisions, recalls, f1s = map(np.array, (precisions, recalls, f1s))

    pred_total = sum(len(p) for p in all_pred)
    gold_total = sum(len(g) for g in all_gold)
    matched_sum = sum(precisions[i] * len(all_pred[i]) for i in range(len(precisions)))
    micro_p = matched_sum / pred_total if pred_total else 0.0
    micro_r = matched_sum / gold_total if gold_total else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    return {
        "gbert_micro_f1": float(micro_f1),
        "gbert_macro_f1": float(np.nanmean(f1s)) if len(f1s) else 0.0,
        "n_items":    len(item_details),
        "pred_total": int(pred_total),
        "gold_total": int(gold_total),
    }
