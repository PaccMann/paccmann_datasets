[![Updates](https://pyup.io/repos/github/PaccMann/paccmann_datasets/shield.svg)](https://pyup.io/repos/github/PaccMann/paccmann_datasets/)
[![Build Status](https://travis-ci.org/PaccMann/paccmann_datasets.svg?branch=master)](https://travis-ci.org/PaccMann/paccmann_datasets)
# pytoda

## Overview

pytoda - PaccMann P*yTo*rch *Da*taset Classes

`pytoda` is a package that simplifies using biochemcial data for training models
in `pytorch`. It implements datasets to handle SMILES (`SMILESDataset`, `PolymerDataset`), protein sequences
(`ProteinSequenceDataset`) or multimodal datasets for downstream prediction
tasks (`DrugSensitivityDataset`, `ProteinProteinInteractionDataset`).

## Requirements

- `conda>=3.7`

## Installation

Create a conda environment:

```sh
conda env create -f conda.yml
```

Activate the environment:

```sh
conda activate pytoda
```

Install:

```sh
pip install .
```

### development

Install in editable mode for development:

```sh
pip install -e .
```

## Examples

For some examples on how to use `pytoda` see [here](./examples)

## References

If you use `pytoda` in your projects, please cite the following:

```bib
@misc{born2019paccmannrl,
    title={PaccMann^RL: Designing anticancer drugs from transcriptomic data via reinforcement learning},
    author={Jannis Born and Matteo Manica and Ali Oskooei and Joris Cadow and Maria Rodriguez Martinez},
    year={2019},
    eprint={1909.05114},
    archivePrefix={arXiv},
    primaryClass={q-bio.BM}
}
```
