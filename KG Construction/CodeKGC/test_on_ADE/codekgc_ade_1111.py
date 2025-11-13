import os
import argparse
from easyinstruct import BasePrompt
import json
from collections import Counter
import logging
import bert_score
from bert_score import score as bert_score_compute
import torch
from sentence_transformers import SentenceTransformer, util
import re
import numpy as np
from scipy.optimize import linear_sum_assignment

# --- Argument Parser for Configurability ---
parser = argparse.ArgumentParser(description="CodeKGC for ADE")
parser.add_argument('--config', default='config.json', help='Path to config JSON')
parser.add_argument('--limit', type=int, default=None, help='Limit number of items to process (for cost savings)')
parser.add_argument('--debug', action='store_true', help='Enable verbose logging (pairwise sims, full results)')
args = parser.parse_args()

# --- Set up Logging ---
logging.basicConfig(
    level=logging.INFO if not args.debug else logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run_log.txt", mode='w'),
        logging.StreamHandler()
    ]
)

# --- Normalization Logic ---
def normalize_entity(entity_text: str) -> str:
    if not isinstance(entity_text, str):
        return ""
    
    text = entity_text.strip().lower()
    text = re.sub(r'^(a|an|the)\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\.$', '', text)
    text = text.strip('\'"')
    
    return text if text else ""

def normalize_relation(relation_text: str) -> str:
    return relation_text.strip().lower().replace(" ", "_")

def codekgc(config_path: str = args.config):
    with open(config_path, 'r') as f:
        config = json.load(f)
    [schema_path, ICL_path, example_path, openai_key, engine, temperature, max_tokens, n] = [
        config["schema_path"], config["ICL_path"], config["example_path"],
        config["openai_key"], config["engine"], config["temperature"],
        config["max_tokens"], config["n"]
    ]
   
    os.environ['OPENAI_API_KEY'] = openai_key
    with open(schema_path, 'r') as f:
        schema_prompt = f.read()
    with open(ICL_path, 'r') as f:
        ICL_prompt = f.read()
   
    with open(example_path, 'r') as f:
        data = json.load(f)
    if args.limit:
        data = data[:args.limit]  # Limit items to save API costs
    
    all_pred_triples = []
    all_gold_triples = []
    all_pred_strings = []  # For BERTscore
    all_gold_strings = []  # For BERTscore
    item_details = []  # To store per-item info for later analysis
    
    for i, item in enumerate(data):
        text = item['text']
        gold_triples = item['triple_list']
       
        gold_list = []
        for h, r, t in gold_triples:
            h_norm = normalize_entity(h)
            t_norm = normalize_entity(t)
            r_norm = normalize_relation(r)
            if h_norm and t_norm:
                gold_list.append((h_norm, r_norm, t_norm))
        gold_list = sorted(list(set(gold_list)))
        all_gold_triples.extend(gold_list)
        text_prompt = '\n""" ' + text + ' """'
        strict_instruction = "\nExtract triples using only the provided relations: adverse effect. Do not use or invent other relations."
        prompt = schema_prompt + '\n' + ICL_prompt + text_prompt + strict_instruction
        
        logging.info(f"--- Processing Item {i+1}/{len(data)} ---")
        logging.info(f"PROMPT (length: {len(prompt)} chars)")
       
        code_prompt = BasePrompt()
        code_prompt.build_prompt(prompt)
        completion = code_prompt.get_openai_result(
            engine=engine,
            temperature=temperature,
            max_tokens=max_tokens,  # Suggestion: Reduce to 256 if extractions are short to save ~50% output tokens
            n=n
        )
        result = completion.strip()
        if args.debug:
            logging.debug(f"RESULT:\n{result}")  # Only log full result in debug mode
        try:
            # Look for the "Extract([ ... ])" part of the string
            # This regex specifically captures the content of the square brackets
            match = re.search(
                r"extract\s*=\s*Extract\s*\(\s*(\[.*?\])\s*\)", 
                result, 
                re.DOTALL | re.IGNORECASE
            )

            if match:
                # If found, run exec() ONLY on the matched part
                # match.group(1) will be "[...]" or "[]"
                result_code = f"extract = Extract({match.group(1)})"
                local_namespace = {}
                exec(schema_prompt + '\n' + result_code, local_namespace)
                
                extract = local_namespace.get('extract')
                if extract and hasattr(extract, 'triples'):
                    pred_triples = []
                    for t in extract.triples:
                        h_norm = normalize_entity(t.head.name)
                        t_norm = normalize_entity(t.tail.name)
                        r_norm = normalize_relation(t.relation.name)
                        if h_norm and t_norm:
                            pred_triples.append((h_norm, r_norm, t_norm))
                    pred_triples = sorted(list(set(pred_triples)))
                else:
                    pred_triples = []
            else:
                # If no "extract = Extract([...])" was found at all
                logging.warning(f"No valid 'Extract' call found in result for item {i+1}.")
                pred_triples = []

        except Exception as e:
            logging.error(f"Error parsing generated triples for text: '{text}'")
            logging.error(f"Error details: {e}")
            pred_triples = []
        
        # Log per-item comparison (minimal: only triples)
        logging.info(f"Item {i+1} Gold triples: {gold_list}")
        logging.info(f"Item {i+1} Pred triples: {pred_triples}")
       
        all_pred_triples.extend(pred_triples)
        
        # Prepare concatenated strings for BERTscore (sorted for order normalization)
        pred_triples_sorted = sorted(pred_triples)
        gold_triples_sorted = sorted(gold_list)
        pred_str = " | ".join([f"{h} {r} {t}" for h, r, t in pred_triples_sorted]) if pred_triples_sorted else ""
        gold_str = " | ".join([f"{h} {r} {t}" for h, r, t in gold_triples_sorted]) if gold_triples_sorted else ""
        all_pred_strings.append(pred_str)
        all_gold_strings.append(gold_str)
        
        # Store for later analysis
        item_details.append({
            'index': i+1,
            'gold_list': set(gold_list),  # For exact comparison
            'pred_list': set(pred_triples),
            'gold_str': gold_str,
            'pred_str': pred_str,
            'gold_triples': gold_list,  # Store original lists for per-triple eval
            'pred_triples': pred_triples
        })
    
    # --- Metrics Calculation ---
    logging.info("\n----- FINAL METRICS -----")

    # Strict Metrics
    gold_count = Counter(all_gold_triples)
    pred_count = Counter(all_pred_triples)
    tp = sum(min(gold_count[t], pred_count[t]) for t in gold_count)
    pred_total = len(all_pred_triples)
    gold_total = len(all_gold_triples)
    micro_p_strict = tp / pred_total if pred_total > 0 else 0.0
    micro_r_strict = tp / gold_total if gold_total > 0 else 0.0
    micro_f1_strict = 2 * micro_p_strict * micro_r_strict / \
        (micro_p_strict + micro_r_strict) if micro_p_strict + \
        micro_r_strict > 0 else 0.0

    item_ps_strict, item_rs_strict, item_f1s_strict = [], [], []
    for detail in item_details:
        tp_item = len(detail['pred_list'] & detail['gold_list'])
        pred_item_total = len(detail['pred_triples'])
        gold_item_total = len(detail['gold_triples'])
        p_item = tp_item / pred_item_total if pred_item_total > 0 else (1.0 if gold_item_total == 0 else 0.0)
        r_item = tp_item / gold_item_total if gold_item_total > 0 else (1.0 if pred_item_total == 0 else 0.0)
        f1_item = 2 * p_item * r_item / (p_item + r_item) if (p_item + r_item) > 0 else 0.0
        if pred_item_total == 0 and gold_item_total == 0:
            f1_item = 1.0
        item_ps_strict.append(p_item)
        item_rs_strict.append(r_item)
        item_f1s_strict.append(f1_item)
        logging.info(
            f"Item {detail['index']} Strict: P={p_item:.4f}, R={r_item:.4f}, F1={f1_item:.4f}")

    macro_p_strict = np.nanmean(item_ps_strict) if item_ps_strict else 0.0
    macro_r_strict = np.nanmean(item_rs_strict) if item_rs_strict else 0.0
    macro_f1_strict = np.nanmean(item_f1s_strict) if item_f1s_strict else 0.0

    # Base BERTScore (with IDF added for robustness)
    if any(all_pred_strings) and any(all_gold_strings):
        # Base BERTScore (P, R, F1) - Uses greedy max as per formulas
        P_macro, R_macro, F1_macro = bert_score_compute(
            all_pred_strings, all_gold_strings, lang="en", verbose=False, model_type="roberta-large", idf=True
        )
        macro_p_bert = P_macro.mean().item()
        macro_r_bert = R_macro.mean().item()
        macro_f1_bert = F1_macro.mean().item()

        all_pred_concat = ' | '.join(filter(None, all_pred_strings))
        all_gold_concat = ' | '.join(filter(None, all_gold_strings))
        P_micro, R_micro, F1_micro = bert_score_compute(
            [all_pred_concat], [all_gold_concat], lang="en", verbose=False, model_type="roberta-large", idf=False
        )
        micro_p_bert = P_micro.item()
        micro_r_bert = R_micro.item() if not np.isnan(R_micro.item()) else 0.0
        micro_f1_bert = F1_micro.item() if not np.isnan(F1_micro.item()) else 0.0
        
        # Greedy BERTScore
        item_greedy_p, item_greedy_r, item_greedy_f1 = [], [], []

        for detail in item_details:
            pred_strings = [f"{h} {r} {t}" for h, r, t in detail['pred_triples']]
            gold_strings = [f"{h} {r} {t}" for h, r, t in detail['gold_triples']]

            if not pred_strings and not gold_strings:
                item_greedy_p.append(1.0)
                item_greedy_r.append(1.0)
                item_greedy_f1.append(1.0)
                continue

            if not pred_strings:  # Has gold, but no preds (all missed)
                item_greedy_p.append(1.0)  # Vacuously true: all 0 preds are "correct"
                item_greedy_r.append(0.0)
                item_greedy_f1.append(0.0)
                continue

            if not gold_strings:  # Has preds, but no gold (all false pos)
                item_greedy_p.append(0.0)
                item_greedy_r.append(1.0)  # Vacuously true: all 0 golds are "found"
                item_greedy_f1.append(0.0)
                continue
            
            # Greedy-P (Precision): Avg. max similarity for each *predicted* triple
            P_greedy, _, _ = bert_score_compute(
                pred_strings,
                [gold_strings] * len(pred_strings),  # Compare each pred to ALL golds
                lang="en", verbose=False,
                model_type="roberta-large", idf=True
            )
            greedy_p_val = np.nanmean(P_greedy.cpu().numpy())
            greedy_p = 0.0 if np.isnan(greedy_p_val) else greedy_p_val

            # Greedy-R (Recall): Avg. max similarity for each *gold* triple
            P_recall, _, _ = bert_score_compute(  # Capture the first value (Precision)
                gold_strings,
                [pred_strings] * len(gold_strings),  # Compare each gold to ALL preds
                lang="en", verbose=False,
                model_type="roberta-large", idf=True
            )
            greedy_r_val = np.nanmean(P_recall.cpu().numpy())
            greedy_r = 0.0 if np.isnan(greedy_r_val) else greedy_r_val
            
            greedy_f1 = 0.0
            if (greedy_p + greedy_r) > 0:
                greedy_f1 = 2 * greedy_p * greedy_r / (greedy_p + greedy_r)

            item_greedy_p.append(greedy_p)
            item_greedy_r.append(greedy_r)
            item_greedy_f1.append(greedy_f1)

        # Macro averages (mean over items)
        macro_greedy_p = np.nanmean(item_greedy_p) if item_greedy_p else 0.0
        macro_greedy_r = np.nanmean(item_greedy_r) if item_greedy_r else 0.0
        macro_greedy_f1 = np.nanmean(item_greedy_f1) if item_greedy_f1 else 0.0

        # Micro for Greedy: Flatten all triples and compute global greedy
        all_pred_triple_strs = [f"{h} {r} {t}" for triples in item_details for h, r, t in triples['pred_triples']]
        all_gold_triple_strs = [f"{h} {r} {t}" for triples in item_details for h, r, t in triples['gold_triples']]
        
        if all_pred_triple_strs and all_gold_triple_strs:
            # Micro P: Avg max sim for each pred vs all golds
            P_micro_g, _, _ = bert_score_compute(
                all_pred_triple_strs,
                [all_gold_triple_strs] * len(all_pred_triple_strs),
                lang="en", verbose=False, model_type="roberta-large", idf=True
            )
            micro_greedy_p = np.nanmean(P_micro_g.cpu().numpy())

            # Micro R: Avg max sim for each gold vs all preds
            P_micro_r, _, _ = bert_score_compute(  # Capture the first value (Precision)
                all_gold_triple_strs,
                [all_pred_triple_strs] * len(all_gold_triple_strs),
                lang="en", verbose=False, model_type="roberta-large", idf=True
            )
            micro_greedy_r = np.nanmean(P_micro_r.cpu().numpy())
            
            micro_greedy_f1 = 0.0
            if (micro_greedy_p + micro_greedy_r) > 0:
                micro_greedy_f1 = 2 * micro_greedy_p * micro_greedy_r / (micro_greedy_p + micro_greedy_r)
        else:
            micro_greedy_p, micro_greedy_r, micro_greedy_f1 = 0.0, 0.0, 0.0

        # G-BERTScore (Hungarian Matching from EXPLAGRAPHS)
        def get_g_bert_score(all_gold_edges, all_pred_edges, idf=False):
            references = []
            candidates = []
            ref_cand_index = {}
            for graph_idx, (gold_edges, pred_edges) in enumerate(zip(all_gold_edges, all_pred_edges)):
                for gold_edge in gold_edges:
                    for pred_edge in pred_edges:
                        references.append(gold_edge)
                        candidates.append(pred_edge)
                        ref_cand_index[(graph_idx, gold_edge, pred_edge)] = len(references) - 1

            if not references:
                return np.zeros(len(all_gold_edges)), np.zeros(len(all_gold_edges)), np.zeros(len(all_gold_edges))

            _, _, bs_F1 = bert_score_compute(candidates, references, lang='en', verbose=False, model_type="roberta-large", idf=idf)
            bs_F1 = bs_F1.cpu().numpy()

            precisions, recalls, f1s = [], [], []
            for graph_idx, (gold_edges, pred_edges) in enumerate(zip(all_gold_edges, all_pred_edges)):
                num_gold = len(gold_edges)
                num_pred = len(pred_edges)
                if num_gold == 0 and num_pred == 0:
                    precisions.append(1.0)
                    recalls.append(1.0)
                    f1s.append(1.0)
                    continue
                if num_pred == 0:
                    precisions.append(1.0)
                    recalls.append(0.0)
                    f1s.append(0.0)
                    continue
                if num_gold == 0:
                    precisions.append(0.0)
                    recalls.append(1.0)
                    f1s.append(0.0)
                    continue

                score_matrix = np.zeros((num_gold, num_pred))
                for i, gold_edge in enumerate(gold_edges):
                    for j, pred_edge in enumerate(pred_edges):
                        idx = ref_cand_index.get((graph_idx, gold_edge, pred_edge))
                        if idx is not None:
                            score_matrix[i, j] = bs_F1[idx]

                row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)
                matched_sim = score_matrix[row_ind, col_ind]

                sample_precision = matched_sim.sum() / num_pred if num_pred > 0 else 0.0
                sample_recall = matched_sim.sum() / num_gold if num_gold > 0 else 0.0
                sample_f1 = 2 * sample_precision * sample_recall / (sample_precision + sample_recall) if sample_precision + sample_recall > 0 else 0.0

                precisions.append(sample_precision)
                recalls.append(sample_recall)
                f1s.append(sample_f1)

            return np.array(precisions), np.array(recalls), np.array(f1s)

        all_gold_edges = [[f"{h} {r} {t}" for h, r, t in detail['gold_triples']] for detail in item_details]
        all_pred_edges = [[f"{h} {r} {t}" for h, r, t in detail['pred_triples']] for detail in item_details]

        precisions_h, recalls_h, f1s_h = get_g_bert_score(all_gold_edges, all_pred_edges, idf=False)

        macro_gbs_p_h = np.nanmean(precisions_h) if precisions_h.size > 0 else 0.0
        macro_gbs_r_h = np.nanmean(recalls_h) if recalls_h.size > 0 else 0.0
        macro_gbs_f1_h = np.nanmean(f1s_h) if f1s_h.size > 0 else 0.0

        # Micro for G-BERTScore: Sum matched similarities from per-sample computations
        total_matched_sum = 0.0
        for i in range(len(precisions_h)):
            total_matched_sum += precisions_h[i] * len(all_pred_edges[i])
        micro_gbs_p_h = total_matched_sum / pred_total if pred_total > 0 else 0.0
        micro_gbs_r_h = total_matched_sum / gold_total if gold_total > 0 else 0.0
        micro_gbs_f1_h = 2 * micro_gbs_p_h * micro_gbs_r_h / (micro_gbs_p_h + micro_gbs_r_h) if micro_gbs_p_h + micro_gbs_r_h > 0 else 0.0
        
    else:
        # All zeros for empties
        macro_p_bert, macro_r_bert, macro_f1_bert = 0.0, 0.0, 0.0
        micro_p_bert, micro_r_bert, micro_f1_bert = 0.0, 0.0, 0.0
        macro_greedy_p, macro_greedy_r, macro_greedy_f1 = 0.0, 0.0, 0.0
        micro_greedy_p, micro_greedy_r, micro_greedy_f1 = 0.0, 0.0, 0.0
        macro_gbs_p_h, macro_gbs_r_h, macro_gbs_f1_h = 0.0, 0.0, 0.0
        micro_gbs_p_h, micro_gbs_r_h, micro_gbs_f1_h = 0.0, 0.0, 0.0
        
    # Soft Semantic Score
    logging.info("\n--- Sentence Transformer Semantic Similarity Evaluation ---")
    model = SentenceTransformer('all-mpnet-base-v2')
    threshold = 0.8  # Keep as is, or tune per dataset (literature often uses 0.75-0.9)
    global_matched_preds, global_matched_golds = 0, 0
    item_ps_soft, item_rs_soft, item_f1s_soft = [], [], []

    for detail in item_details:
        pred_triples, gold_triples = detail['pred_triples'], detail['gold_triples']
        
        if not pred_triples and not gold_triples:
            item_ps_soft.append(1.0)
            item_rs_soft.append(1.0)
            item_f1s_soft.append(1.0)
            continue

        pred_strings = [f"{h} {r} {t}" for h, r, t in pred_triples]
        gold_strings = [f"{h} {r} {t}" for h, r, t in gold_triples]

        if not pred_strings or not gold_strings:
            matched_preds, matched_golds = 0, 0
        else:
            pred_embs = model.encode(pred_strings, convert_to_tensor=True).cpu().numpy()
            gold_embs = model.encode(gold_strings, convert_to_tensor=True).cpu().numpy()
            
            # Cosine similarity matrix (normalize to [0,1] for matching)
            sim_matrix = np.dot(pred_embs, gold_embs.T) / (np.linalg.norm(pred_embs, axis=1)[:, np.newaxis] * np.linalg.norm(gold_embs, axis=1))
            sim_matrix = np.maximum(sim_matrix, 0)  # Clip negatives
            
            # Hungarian for optimal assignment (maximize similarity)
            row_ind, col_ind = linear_sum_assignment(sim_matrix, maximize=True)
            matched_sim = sim_matrix[row_ind, col_ind]
            
            # Count matches above threshold
            matched_preds = sum(1 for s in matched_sim if s > threshold)
            matched_golds = matched_preds  # Symmetric in bipartite

        p_item = matched_preds / len(pred_triples) if pred_triples else (1.0 if not gold_triples else 0.0)
        r_item = matched_golds / len(gold_triples) if gold_triples else (1.0 if not pred_triples else 0.0)
        f1_item = 2 * p_item * r_item / (p_item + r_item) if p_item + r_item > 0 else 0.0

        item_ps_soft.append(p_item)
        item_rs_soft.append(r_item)
        item_f1s_soft.append(f1_item)
        global_matched_preds += matched_preds
        global_matched_golds += matched_golds
        logging.info(f"Item {detail['index']} Semantic: P={p_item:.4f}, R={r_item:.4f}, F1={f1_item:.4f}")

    micro_p_soft = global_matched_preds / pred_total if pred_total > 0 else 0.0
    micro_r_soft = global_matched_golds / gold_total if gold_total > 0 else 0.0
    micro_f1_soft = 2 * micro_p_soft * micro_r_soft / (micro_p_soft + micro_r_soft) if micro_p_soft + micro_r_soft > 0 else 0.0

    macro_p_soft = np.nanmean(item_ps_soft) if item_ps_soft else 0.0
    macro_r_soft = np.nanmean(item_rs_soft) if item_rs_soft else 0.0
    macro_f1_soft = np.nanmean(item_f1s_soft) if item_f1s_soft else 0.0

    # --- Overall Metrics Summary ---
    logging.info("\n--- Overall Metrics Summary ---")
    logging.info(f"Strict Exact-Match (Micro): P: {micro_p_strict:.4f} R: {micro_r_strict:.4f} F1: {micro_f1_strict:.4f}")
    logging.info(f"Strict Exact-Match (Macro): P: {macro_p_strict:.4f} R: {macro_r_strict:.4f} F1: {macro_f1_strict:.4f}")
    logging.info(f"BERTScore (Micro): P: {micro_p_bert:.4f} R: {micro_r_bert:.4f} F1: {micro_f1_bert:.4f}")
    logging.info(f"BERTScore (Macro): P: {macro_p_bert:.4f} R: {macro_r_bert:.4f} F1: {macro_f1_bert:.4f}")
    logging.info(f"Greedy BS (Micro): P: {micro_greedy_p:.4f} R: {micro_greedy_r:.4f} F1: {micro_greedy_f1:.4f}")
    logging.info(f"Greedy BS (Macro): P: {macro_greedy_p:.4f} R: {macro_greedy_r:.4f} F1: {macro_greedy_f1:.4f}")
    logging.info(f"G-BERTScore (Micro): P: {micro_gbs_p_h:.4f} R: {micro_gbs_r_h:.4f} F1: {micro_gbs_f1_h:.4f}")
    logging.info(f"G-BERTScore (Macro): P: {macro_gbs_p_h:.4f} R: {macro_gbs_r_h:.4f} F1: {macro_gbs_f1_h:.4f}")
    logging.info(f"Soft Semantic (Micro): P: {micro_p_soft:.4f} R: {micro_r_soft:.4f} F1: {micro_f1_soft:.4f}")
    logging.info(f"Soft Semantic (Macro): P: {macro_p_soft:.4f} R: {macro_r_soft:.4f} F1: {macro_f1_soft:.4f}")

if __name__ == "__main__":
    codekgc()