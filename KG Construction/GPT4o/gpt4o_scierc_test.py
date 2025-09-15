import json
import time
import logging
import os
from collections import Counter
from typing import List, Tuple, Dict, Set
from openai import OpenAI
import bert_score
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
import re


API_KEY = "YOUR_API_KEY"

TEST_DATASET_PATH = '../LLM_based_Datasets/scierc_test_converted.json'
LOG_FILE_PATH = 'benchmark_scierc_test_gpt4o_test_all.log' 
MODEL_NAME = "gpt-4o"
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
# Set to None or 0 to run on the full test set
NUM_EXAMPLES_TO_TEST = None

# --- Set up logging ---
logging.basicConfig(level=logging.INFO, filename=LOG_FILE_PATH,
                    filemode='w', format='%(message)s')

# --- Main Script ---
if not API_KEY or "YOUR_API_KEY_HERE" in API_KEY:
    logging.warning("API key not set. Please set the API_KEY variable or use an environment variable.")
    print("WARNING: API key not found. Please set the API_KEY variable.")
    # exit()

try:
    client = OpenAI(api_key=API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    print(f"Failed to initialize OpenAI client: {e}")
    exit()



SCIERC_PROMPT = """Your task is to extract scientific entities and their relationships from a given text and structure them as a JSON object. You must follow these rules precisely.

**--- Relation Types ---**
You MUST use ONLY ONE of the following seven relation types:
- **Used-for**: An entity is used for a task or purpose.
- **Feature-of**: An entity is a feature, property, or characteristic of another.
- **Hyponym-of**: An entity is a subtype or specific example of a broader category.
- **Part-of**: An entity is a physical or logical part of a larger whole.
- **Compare**: Two entities of the same type are being compared.
- **Evaluate-for**: An entity is used to evaluate another.
- **Conjunction**: Two entities are presented as a conjoined list.

**--- Extraction & Formatting Rules ---**
1.  **Chain of Thought (Internal)**: First, identify entities. Second, find relationships between them. Third, assign one of the seven relation types. Finally, form `(Head, Relation, Tail)` triples.
2.  **Entity Granularity**: Extract entities precisely as they appear in the gold standard examples. Do not include verbs in entities (e.g., extract "NE items", not "recognizes NE items").
3.  **Handle Conjunctions**: If one entity relates to a list of other entities (e.g., "A is used for B, C, and D"), you MUST create a single triple where the tail is the complete conjoined phrase ("B, C, and D"). Do NOT create separate triples for B, C, and D.
4.  **Pronoun Resolution**: If an entity is a pronoun (e.g., 'it', 'they', 'this'), you MUST try to replace it with the specific noun it refers to from the context.
5.  **Output Format**: Your final output MUST BE ONLY a raw JSON string. It should be a list of objects, where each object has three keys: `"head"`, `"relation"`, and `"tail"`. If no relations are found, return an empty list `[]`.

**--- EXAMPLES ---**

**Text 1 (Handling Conjunctions)**: Our morphological analyzer has done all the necessary work for the recognition and classification of proper names , numerical and temporal expressions , i.e. Named Entity ( NE ) items in the Japanese text .
**JSON Output 1**:
[
    {
        "head": "morphological analyzer",
        "relation": "Used-for",
        "tail": "recognition and classification of proper names , numerical and temporal expressions"
    },
    {
        "head": "Named Entity ( NE ) items",
        "relation": "Hyponym-of",
        "tail": "proper names , numerical and temporal expressions"
    }
]

**Text 2 (Simple Relation)**: In this paper , a novel method to learn the intrinsic object structure for robust visual tracking is proposed .
**JSON Output 2**:
[
    {
        "head": "method",
        "relation": "Used-for",
        "tail": "intrinsic object structure"
    },
    {
        "head": "intrinsic object structure",
        "relation": "Used-for",
        "tail": "robust visual tracking"
    }
]

**Text 3 (No Relations)**: The analyzer is called `` Amorph '' .
**JSON Output 3**:
[]

Now, process the provided text and return ONLY the raw JSON string.
"""


def parse_and_normalize_triples(raw_json_string: str) -> List[Tuple[str, str, str]]:
    """Parses a raw JSON string for SciERC, normalizes entities, and constructs triples."""
    if not raw_json_string or not raw_json_string.strip().startswith('['):
        logging.warning(f"Received non-JSON or empty response: '{raw_json_string}'")
        return []

    json_match = re.search(r'\[.*\]', raw_json_string, re.DOTALL)
    if not json_match:
        logging.warning(f"Could not find a JSON list in the response string: '{raw_json_string}'")
        return []
    
    clean_json_string = json_match.group(0)

    try:
        data = json.loads(clean_json_string)
        if not isinstance(data, list):
            logging.warning(f"JSON is not a list: {data}")
            return []
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON: {clean_json_string}")
        return []

    processed = set()
    for item in data:
        
        if not isinstance(item, dict) or "head" not in item or "relation" not in item or "tail" not in item:
            logging.warning(f"Skipping malformed JSON object: {item}")
            continue

        head = item["head"]
        relation = item["relation"]
        tail = item["tail"]

        if not isinstance(head, str) or not isinstance(relation, str) or not isinstance(tail, str):
            logging.warning(f"Skipping object with non-string values: {item}")
            continue

        
        head_norm = head.strip().lower()
        relation_norm = relation.strip() 
        tail_norm = tail.strip().lower()


        if not head_norm or not relation_norm or not tail_norm or head_norm == tail_norm:
            continue
        

        valid_relations = {'Used-for', 'Feature-of', 'Hyponym-of', 'Part-of', 'Compare', 'Evaluate-for', 'Conjunction'}
        if relation_norm not in valid_relations:
            logging.warning(f"Skipping triple with invalid relation type '{relation_norm}': {item}")
            continue

        processed.add((head_norm, relation_norm, tail_norm))

    return sorted(list(processed))

# --- Load dataset ---
try:
    with open(TEST_DATASET_PATH, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    logging.error(f"FATAL: Test dataset not found at path: {TEST_DATASET_PATH}")
    print(f"FATAL: Test dataset not found at path: {TEST_DATASET_PATH}")
    exit()

if NUM_EXAMPLES_TO_TEST:
    data = data[:NUM_EXAMPLES_TO_TEST]

print(f"Loading sentence transformer model: {SENTENCE_MODEL_NAME}...")
sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME)
print("Model loaded.")


# --- Benchmark Loop ---
all_gold_triples, all_pred_triples, all_pred_strings, all_gold_strings = [], [], [], []
item_details = []

for i, item in enumerate(data):
    text = item['text']
    gold_triples_raw = item.get('triple_list', [])
    
    # Normalize the gold standard for fair comparison
    gold_triples = []
    for h, r, t in gold_triples_raw:
        if isinstance(h, str) and isinstance(r, str) and isinstance(t, str):
            h_norm = h.strip().lower()
            r_norm = r.strip()
            t_norm = t.strip().lower()
            if h_norm and r_norm and t_norm:
                gold_triples.append((h_norm, r_norm, t_norm))
    gold_triples = sorted(list(set(gold_triples)))


    logging.info(f"\n----- Processing Item {i+1}/{len(data)} -----")
    logging.info(f"Text: {text}")
    print(f"Processing Item {i+1}/{len(data)}...")
    
    pred_triples = []
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SCIERC_PROMPT},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
        )
        response_string = response.choices[0].message.content
        
        logging.info(f"Raw JSON String from API: {response_string}")
        pred_triples = parse_and_normalize_triples(response_string)

    except Exception as e:
        logging.error(f"An API or parsing error occurred for item {i+1}: {e}", exc_info=True)
        pred_triples = []

    logging.info(f"Gold (Normalized): {gold_triples}")
    logging.info(f"Pred (from JSON & Cleaned): {pred_triples}")

    all_gold_triples.extend(gold_triples)
    all_pred_triples.extend(pred_triples)

    pred_str = " | ".join(sorted([f"{h} {r} {t}" for h, r, t in set(pred_triples)]))
    gold_str = " | ".join(sorted([f"{h} {r} {t}" for h, r, t in set(gold_triples)]))
    all_pred_strings.append(pred_str)
    all_gold_strings.append(gold_str)

    item_details.append({
        'index': i+1,
        'gold_list': set(gold_triples), 'pred_list': set(pred_triples),
        'gold_str': gold_str, 'pred_str': pred_str,
        'gold_triples': gold_triples, 'pred_triples': pred_triples
    })

    time.sleep(1.5) 

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
    p_item = tp_item / len(detail['pred_triples']) if detail['pred_triples'] else (1.0 if not detail['gold_triples'] else 0.0)
    r_item = tp_item / len(detail['gold_triples']) if detail['gold_triples'] else (1.0 if not detail['pred_triples'] else 0.0)

    if p_item + r_item == 0:
        f1_item = 0.0
    else:
        f1_item = 2 * p_item * r_item / (p_item + r_item)

    if not detail['pred_triples'] and not detail['gold_triples']:
        f1_item = 1.0

    item_ps_strict.append(p_item)
    item_rs_strict.append(r_item)
    item_f1s_strict.append(f1_item)
    logging.info(
        f"Item {detail['index']} Strict: P={p_item:.4f}, R={r_item:.4f}, F1={f1_item:.4f}")

macro_p_strict = sum(
    item_ps_strict) / len(item_ps_strict) if item_ps_strict else 0.0
macro_r_strict = sum(
    item_rs_strict) / len(item_rs_strict) if item_rs_strict else 0.0
macro_f1_strict = sum(
    item_f1s_strict) / len(item_f1s_strict) if item_f1s_strict else 0.0

# BERTScore
if any(all_pred_strings) and any(all_gold_strings):
    P_macro, R_macro, F1_macro = bert_score.score(
        all_pred_strings, all_gold_strings, lang="en", verbose=False, model_type="bert-base-uncased")
    macro_p_bert, macro_r_bert, macro_f1_bert = P_macro.mean(
    ).item(), R_macro.mean().item(), F1_macro.mean().item()

    all_pred_concat = ' | '.join(filter(None, all_pred_strings))
    all_gold_concat = ' | '.join(filter(None, all_gold_strings))
    P_micro, R_micro, F1_micro = bert_score.score(
        [all_pred_concat], [all_gold_concat], lang="en", verbose=False, model_type="bert-base-uncased")
    micro_p_bert, micro_r_bert, micro_f1_bert = P_micro.item(
    ), R_micro.item(), F1_micro.item()
else:
    macro_p_bert, macro_r_bert, macro_f1_bert = 0.0, 0.0, 0.0
    micro_p_bert, micro_r_bert, micro_f1_bert = 0.0, 0.0, 0.0

# Soft Semantic Score
threshold = 0.8
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
        pred_embs = sentence_model.encode(
            pred_strings, convert_to_tensor=True)
        gold_embs = sentence_model.encode(
            gold_strings, convert_to_tensor=True)
        sim_matrix = cos_sim(pred_embs, gold_embs).cpu().numpy()
        matched_preds = sum(1 for row in sim_matrix if max(row, default=0) > threshold)
        matched_golds = sum(
            1 for col in sim_matrix.T if max(col, default=0) > threshold)

    p_item = matched_preds / len(pred_triples) if pred_triples else (1.0 if not gold_triples else 0.0)
    r_item = matched_golds / len(gold_triples) if gold_triples else (1.0 if not pred_triples else 0.0)

    if p_item + r_item == 0:
        f1_item = 0.0
    else:
        f1_item = 2 * p_item * r_item / (p_item + r_item)

    if not pred_triples and not gold_triples:
        f1_item = 1.0

    item_ps_soft.append(p_item)
    item_rs_soft.append(r_item)
    item_f1s_soft.append(f1_item)
    global_matched_preds += matched_preds
    global_matched_golds += matched_golds
    logging.info(
        f"Item {detail['index']} Semantic: P={p_item:.4f}, R={r_item:.4f}, F1={f1_item:.4f}")

micro_p_soft = global_matched_preds / pred_total if pred_total > 0 else 0.0
micro_r_soft = global_matched_golds / gold_total if gold_total > 0 else 0.0
micro_f1_soft = 2 * micro_p_soft * micro_r_soft / \
    (micro_p_soft + micro_r_soft) if micro_p_soft + \
    micro_r_soft > 0 else 0.0

macro_p_soft = sum(
    item_ps_soft) / len(item_ps_soft) if item_ps_soft else 0.0
macro_r_soft = sum(
    item_rs_soft) / len(item_rs_soft) if item_rs_soft else 0.0
macro_f1_soft = sum(
    item_f1s_soft) / len(item_f1s_soft) if item_f1s_soft else 0.0

# --- Final Output ---
output_str = f"""
Benchmark Results for SciERC:
---------------------------------
Number of examples: {len(data)}

Strict Averages:
  - Macro: P={macro_p_strict:.4f}, R={macro_r_strict:.4f}, F1={macro_f1_strict:.4f}
  - Micro: P={micro_p_strict:.4f}, R={micro_r_strict:.4f}, F1={micro_f1_strict:.4f}

BERTScore Averages:
  - Macro: P={macro_p_bert:.4f}, R={macro_r_bert:.4f}, F1={macro_f1_bert:.4f}
  - Micro: P={micro_p_bert:.4f}, R={micro_r_bert:.4f}, F1={micro_f1_bert:.4f}

Semantic Score Averages (Threshold: {threshold}):
  - Macro: P={macro_p_soft:.4f}, R={macro_r_soft:.4f}, F1={macro_f1_soft:.4f}
  - Micro: P={micro_p_soft:.4f}, R={micro_r_soft:.4f}, F1={micro_f1_soft:.4f}
"""
print(output_str)
logging.info(output_str)