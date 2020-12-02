[![Build Status](https://travis-ci.org/PaccMann/paccmann_datasets.svg?branch=master)](https://travis-ci.org/PaccMann/paccmann_datasets)[![codecov](https://codecov.io/gh/PaccMann/paccmann_datasets/branch/master/graph/badge.svg?token=C10ICE7S0Q)](https://codecov.io/gh/PaccMann/paccmann_datasets)[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# pytoda

## Overview

pytoda - PaccMann P*yTo*rch *Da*taset Classes

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
pip install --user --no-use-pep517 -e .
```

## Examples

For some examples on how to use `pytoda` see [here](./examples)

## References

If you use `pytoda` in your projects, please cite the following:

```bib
@inproceedings{born2020paccmann,
  title={Paccmann rl: Designing anticancer drugs from transcriptomic data via reinforcement learning},
  author={Born, Jannis and Manica, Matteo and Oskooei, Ali and Cadow, Joris and Mart{\'\i}nez, Mar{\'\i}a Rodr{\'\i}guez},
  booktitle={International Conference on Research in Computational Molecular Biology},
  pages={231--233},
  year={2020},
  organization={Springer}
}
```
