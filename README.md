# PyToDa

[![PyPI version](https://badge.fury.io/py/pytoda.svg)](https://badge.fury.io/py/pytoda)
[![build](https://github.com/PaccMann/paccmann_datasets/workflows/build/badge.svg)](https://github.com/PaccMann/paccmann_datasets/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://static.pepy.tech/badge/pytoda)](https://pepy.tech/project/pytoda)
[![Downloads](https://static.pepy.tech/badge/pytoda/month)](https://pepy.tech/project/pytoda)
[![GitHub Super-Linter](https://github.com/PaccMann/paccmann_datasets/workflows/style/badge.svg)](https://github.com/marketplace/actions/super-linter)

## Overview

pytoda - PaccMann P*yTo*rch *Da*taset Classes

A python package that eases handling biochemical data for deep learning applications
with pytorch.

## Installation

`pytoda` ships via [PyPI](https://pypi.org/project/pytoda):

```sh
pip install pytoda
```

## Documentation

Please find the full documentation [here](https://paccmann.github.io/paccmann_datasets/).

## Development

For development setup, we recommend to work in a dedicated conda environment:

```sh
conda env create -f conda.yml
```

Activate the environment:

```sh
conda activate pytoda
```

Install in editable mode:

```sh
pip install -r dev_requirements.txt
pip install --user --no-use-pep517 -e .
```

## Examples

For some examples on how to use `pytoda` see [here](./examples)

## References

If you use `pytoda` in your projects, please cite the following:

```bib
@article{born2021data,
  title={Data-driven molecular design for discovery and synthesis of novel ligands: a case study on SARS-CoV-2},
  author={Born, Jannis and Manica, Matteo and Cadow, Joris and Markert, Greta and Mill, Nil Adell and Filipavicius, Modestas and Janakarajan, Nikita and Cardinale, Antonio and Laino, Teodoro and Martinez, Maria Rodriguez},
  journal={Machine Learning: Science and Technology},
  volume={2},
  number={2},
  pages={025024},
  year={2021},
  publisher={IOP Publishing}
}
```
