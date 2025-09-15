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

TEST_DATASET_PATH = "../LLM_based_Datasets/conll04_test_triples.json"
LOG_FILE_PATH = 'benchmark_conll04_gpt4o_test_all_test.log'
MODEL_NAME = "gpt-4o"
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
# Set to None or 0 to run on the full test set
NUM_EXAMPLES_TO_TEST = None



logging.basicConfig(level=logging.INFO, filename=LOG_FILE_PATH,
                    filemode='w', format='%(message)s')

# --- Main Script ---
if not API_KEY or "YOUR_OPENAI_API_KEY" in API_KEY:
    logging.warning("API key not set. Please set the API_KEY variable or use an environment variable.")
    print("WARNING: API key not found. Please set the API_KEY variable.")


try:
    client = OpenAI(api_key=API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    print(f"Failed to initialize OpenAI client: {e}")
    exit()



ULTIMATE_CONTEXT = """Your task is to perform relation extraction on the given text. You must identify relationships between entities and structure them as a JSON object.

**--- The only 5 relations you can extract are ---**
1.  `Kill`: An agent (person or group) kills a victim (person or group).
2.  `Live in`: A person resides in a geographical location.
3.  `Located in`: A geographical location is situated within another geographical location.
4.  `Organization based in`: An organization has its headquarters or a significant presence in a location.
5.  `Work for`: A person is employed by or is a member of an organization.

**--- Constraints and Clarifications ---**
- The object of a `Kill` relation must be a person or group of people, never a location.
- Use `Organization based in` for companies, governments, teams, or institutions. Use `Located in` only for geographical places (cities, states, countries).
- Extract entities exactly as they appear in the text.

**--- PROCESS ---**

**Step 1: Chain of Thought (Internal Monologue)**
1.  **Identify Entities**: Read the text and list all potential named entities (people, organizations, locations).
2.  **Find Relationships**: For each pair of entities, check if one of the 5 allowed relationships exists between them.
3.  **Map Triples**: Create triples in the format `(Head Entity -> Relation -> Tail Entity)`.

**Step 2: Format as JSON**
- Take the triples from your chain of thought and format them into a JSON list of objects.
- Each object in the list MUST have exactly three keys: `"head_entity"`, `"relation"`, and `"tail_entity"`.
- The entities in the JSON must appear **exactly** as they do in the source text.

**Step 3: Final Output**
- Your final output MUST BE ONLY THE RAW JSON STRING. Do not include any other text, explanations, or markdown formatting like ```json.

**--- EXAMPLES ---**

**Text 1**: John Wilkes Booth wasn't a shiftless actor who killed Abraham Lincoln out of despondency.
**JSON Output 1**:
[
    {
        "head_entity": "John Wilkes Booth",
        "relation": "Kill",
        "tail_entity": "Abraham Lincoln"
    }
]

**Text 2**: Meanwhile, Shi Liming at the Institute of Zoology of Kunming found that pandas lack variety.
**JSON Output 2**:
[
    {
        "head_entity": "Shi Liming",
        "relation": "Work for",
        "tail_entity": "Institute of Zoology"
    },
    {
        "head_entity": "Institute of Zoology",
        "relation": "Organization based in",
        "tail_entity": "Kunming"
    }
]

**Text 3**: The meeting was held on Tuesday.
**JSON Output 3**:
[]

**Text 4**: GENEVA, Switzerland (Reuters)
**JSON Output 4**:
[
    {
        "head_entity": "GENEVA",
        "relation": "Located in",
        "tail_entity": "Switzerland"
    },
    {
        "head_entity": "Reuters",
        "relation": "Organization based in",
        "tail_entity": "GENEVA"
    },
    {
        "head_entity": "Reuters",
        "relation": "Organization based in",
        "tail_entity": "Switzerland"
    }
]

Now, process the provided text and return ONLY the raw JSON string.
"""


def parse_and_normalize_triples(raw_json_string: str) -> List[Tuple[str, str, str]]:
    """Parses a raw JSON string from the API for CoNLL04, normalizes entities, and constructs triples."""
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
        if not isinstance(item, dict) or "head_entity" not in item or "relation" not in item or "tail_entity" not in item:
            logging.warning(f"Skipping malformed JSON object: {item}")
            continue

        head = item["head_entity"]
        relation = item["relation"]
        tail = item["tail_entity"]

        if not isinstance(head, str) or not isinstance(relation, str) or not isinstance(tail, str):
            logging.warning(f"Skipping object with non-string values: {item}")
            continue


        head_norm = head.lower().replace("'s", "").strip()
        relation_norm = relation.strip().lower() 
        tail_norm = tail.lower().replace("'s", "").strip()

        if head_norm.startswith('the '): head_norm = head_norm[4:]
        if tail_norm.startswith('the '): tail_norm = tail_norm[4:]


        if head_norm.endswith('.'): head_norm = head_norm[:-1].strip()
        if tail_norm.endswith('.'): tail_norm = tail_norm[:-1].strip()

        if not head_norm or not relation_norm or not tail_norm:
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
    

    gold_triples = []
    for h, r, t in gold_triples_raw:
        if isinstance(h, str) and isinstance(r, str) and isinstance(t, str):

            h_norm = h.lower().replace("'s", "").strip()
            r_norm = r.strip().lower()
            t_norm = t.lower().replace("'s", "").strip()
            
            if h_norm.startswith('the '): h_norm = h_norm[4:]
            if t_norm.startswith('the '): t_norm = t_norm[4:]


            if h_norm.endswith('.'): h_norm = h_norm[:-1].strip()
            if t_norm.endswith('.'): t_norm = t_norm[:-1].strip()

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
                {"role": "system", "content": ULTIMATE_CONTEXT},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
        )
        response_string = response.choices[0].message.content
        
        logging.info(f"Raw JSON String from API: {response_string}")
        pred_triples = parse_and_normalize_triples(response_string)

    except Exception as e:
        logging.error(f"An API or parsing error occurred for item {i+1}: {e}", exc_info=True)
        pred_triples = [] # Ensure pred_triples is an empty list on error

    logging.info(f"Gold (Normalized): {gold_triples}")
    logging.info(f"Pred (from JSON & Cleaned): {pred_triples}")

    all_gold_triples.extend(gold_triples)
    all_pred_triples.extend(pred_triples)

    pred_str = " | ".join(sorted([f"{h}::{r}::{t}" for h, r, t in set(pred_triples)]))
    gold_str = " | ".join(sorted([f"{h}::{r}::{t}" for h, r, t in set(gold_triples)]))
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
threshold = 0.85
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
        
        # A prediction is matched if it has at least one gold triple above the threshold
        matched_preds = sum(1 for row in sim_matrix if max(row, default=0) > threshold)
        # A gold triple is matched if it has at least one prediction above the threshold
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
Benchmark Results for CoNLL04:
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