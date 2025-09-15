# Reproducing ITER Benchmark Results

This repository contains the instructions to reproduce the benchmark results for the ITER model on the **ADE**, **CoNLL04**, and **SciERC** datasets, as presented in the original paper.


## 🚀 Getting Started

Follow the instructions in the ITER repository https://github.com/fleonce/ITER, to clone the ITER repository, prepare the environment, prepare the datasets and download the corresponding model weights `iter-conll04-deberta-large`, `iter-ade-deberta-large`, and `iter-scierc-scideberta-full` to the `models` folder.

Then, running commands to evaluate the models on the corresponding dataset. For example:

`python3 evaluate.py --model models/iter-conll04-deberta-large --dataset conll04`