# ITER

ITER is a supervised baseline. We use the official released code and pre-trained checkpoints
from https://github.com/fleonce/ITER without any modifications.

## Setup

Follow the upstream README to clone the repository, set up the environment, prepare the
datasets, and download the model weights `iter-ade-deberta-large`,
`iter-conll04-deberta-large`, and `iter-scierc-scideberta-full` into the upstream `models/`
folder.

## Reproducing the ITER row of Table 2

```bash
python3 evaluate.py --model models/iter-ade-deberta-large      --dataset ade_split1
python3 evaluate.py --model models/iter-conll04-deberta-large  --dataset conll04
python3 evaluate.py --model models/iter-scierc-scideberta-full --dataset scierc
```

Each command prints `ERE :: f1=...` (entity-relation extraction F1) and
`G-BERT-F1 (micro) :: f1=...` (G-BERTScore micro F1).
