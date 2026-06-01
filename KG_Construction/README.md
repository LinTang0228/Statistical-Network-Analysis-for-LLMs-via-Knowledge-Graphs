# KG Construction Benchmark — Code and Data

Reproduces the KG construction results (Table 2). Three methods, three datasets,
two LLM backbones for the LLM-based methods.

## Layout

```
data/                  test splits used by KGGen and CodeKGC
  ade/test.json        427 sentences  (fold 1 of ADE 10-fold)
  conll04/test.json    288 sentences
  scierc/test.json     551 sentences
kggen/
  run_kggen.py         --backbone {gpt4,claude} --dataset {ade,conll04,scierc}
  prompts.py           zero-shot extraction prompts
codekgc/
  run_codekgc.py       --backbone {gpt4,claude} --dataset {ade,conll04,scierc}
  helpers.py           OpenAI / Anthropic call helpers
  prompts/             per-dataset schema + ICL prompts
iter/
  README.md            instructions for the supervised ITER baseline
metrics/
  g_bert_score.py      Strict F1 + Hungarian G-BERTScore (Saha et al., 2021)
```

## Install

```bash
pip install kg-gen openai anthropic bert_score sentence-transformers scipy
```

## API keys

```bash
export OPENAI_API_KEY=...        # required for --backbone gpt4
export ANTHROPIC_API_KEY=...     # required for --backbone claude
```

## Run

```bash
# KGGen
cd kggen
python run_kggen.py --backbone gpt4   --dataset ade
python run_kggen.py --backbone claude --dataset scierc

# CodeKGC
cd ../codekgc
python run_codekgc.py --backbone gpt4   --dataset conll04
python run_codekgc.py --backbone claude --dataset ade

# ITER
# See iter/README.md
```

## Output

Each run writes three files into `<method>/outputs/`, tagged by dataset and backbone.
For example, `python run_kggen.py --backbone claude --dataset ade` produces:

```
kggen/outputs/
  kggen_ade_claude.log                  per-item log (text, gold, pred)
  kggen_ade_claude_predictions.json     normalized gold + pred triples per item
  SCORES_kggen_ade_claude.txt           summary with the headline F1 numbers
```

CodeKGC writes the analogous files (no predictions JSON; per-item gold/pred are in the log).

`SCORES_*.txt` is a three-line summary:

```
[ade/claude] model=claude-sonnet-4-6  n=427  gold=696  pred=2617  failures=0
  Strict      Micro F1 = 0.0000   Macro F1 = 0.0000
  G-BERTScore Micro F1 = 0.3759   Macro F1 = 0.4075
```

## Models and hyperparameters

LLM calls use temperature 0.1 and max_tokens 512 for both backbones.
`gpt4` resolves to `gpt-4` (OpenAI snapshot `gpt-4-0613`).
`claude` resolves to `claude-sonnet-4-6` (Anthropic).
