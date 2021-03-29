# pytoda
[![build](https://github.com/PaccMann/paccmann_datasets/workflows/build/badge.svg)](https://github.com/PaccMann/paccmann_datasets/actions)[![codecov](https://codecov.io/gh/PaccMann/paccmann_datasets/branch/master/graph/badge.svg?token=C10ICE7S0Q)](https://codecov.io/gh/PaccMann/paccmann_datasets)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

pytoda - PaccMann P*yTo*rch *Da*taset Classes

A python package that eases handling biochemical data for deep learning applications with pytorch.
Please find the full documentation [here](https://paccmann.github.io/paccmann_datasets/).

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

Create the `conda` environment as before, then install in editable mode for development:

```sh
pip install -r dev_requirements.txt
pip install --user --no-use-pep517 -e .
```

## Examples

For some examples on how to use `pytoda` see [here](./examples)

## References

If you use `pytoda` in your projects, please cite the following:

```bib
@article{born2021paccmannrl,
    title = {PaccMann^{RL}: De novo generation of hit-like anticancer molecules from transcriptomic data via reinforcement learning},
    journal = {iScience},
    volume = {24},
    number = {4},
    pages = {102269},
    year = {2021},
    issn = {2589-0042},
    doi = {https://doi.org/10.1016/j.isci.2021.102269},
    url = {https://www.cell.com/iscience/fulltext/S2589-0042(21)00237-6},
    author = {Jannis Born and Matteo Manica and Ali Oskooei and Joris Cadow and Greta Markert and María {Rodríguez Martínez}}
}
```
