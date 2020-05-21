import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
from faerun import Faerun
import numpy as np

from typing import Optional, List


def compute_tsne(
    data: np.ndarray,
    number_components: int = 3,
    perplexity: int = 70,
    n_iter: int = 1000
):
    out = TSNE(
        n_components=number_components, perplexity=perplexity, n_iter=n_iter
    ).fit_transform(data)
    return [out[:, i] for i in range(out.shape[1])]


def tsne_from_fingerprints(
    data_file: str,
    pandas_csv_params: dict = {},
    tsne_params: dict = {'number_components': 3},
    radius: int = 2,
    nBits: int = 256
):
    df = pd.read_csv(data_file, **pandas_csv_params)
    fingerprints = []
    for val in df.SMILES.values:
        mol = Chem.MolFromSmiles(val)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=nBits
        )
        fingerprint = [int(x) for x in fingerprint.ToBitString()]
        fingerprints.append(fingerprint)
    fingerprints = np.array(fingerprints)

    return compute_tsne(fingerprints, **tsne_params)


def tsne_from_properties_file(
    data_file: str,
    properties_file: str,
    pandas_csv_params: dict = {},
    tsne_params: dict = {'number_components': 3},
    radius: int = 2,
    nBits: int = 256
):
    df = pd.read_csv(data_file, **pandas_csv_params)
    properties = _load_properties_file(properties_file)
    properties = _match_properties(properties, df)

    return compute_tsne(properties, **tsne_params)


def fareum_from_pandas(
    df: pd.DataFrame,
    x: List[float],
    y: List[float],
    z: Optional[List[float]] = None,
    c: List[float] = [],
    smiles: List[str] = [],
    names: List[str] = [],
    pandas_csv_params: dict = {},
    plot_name: str = 'fareum_plot',
    plot_path: str = 'fareum'
):

    source_unique = df.source.unique()
    source_dict = {
        k: i / len(source_unique)
        for i, k in enumerate(source_unique)
    }
    c = [source_dict[k] for k in df.source]
    legend_labels = [[(k, i) for i, k in source_dict.items()]]

    params = craete_fareum_params(x, y, z, c, smiles, names)

    create_fareum_viz(params, legend_labels, plot_name, plot_path)


def craete_fareum_params(
    x: List[float],
    y: List[float],
    z: Optional[List[float]] = None,
    c: List[float] = [],
    smiles: List[str] = [],
    names: List[str] = []
):
    params = {
        'x': x,
        'y': y,
        'z': z,
        'c': c,
        'labels': [f'{i}__test__{i}' for i in smiles],
    }
    if z is not None:
        params['z'] = z

    if smiles and names:
        params['labels'] = [
            f'{sm}__{nam}__{sm}' for nam, sm in zip(names, smiles)
        ]
    return params


def create_fareum_viz(
    params: dict,
    legend_labels: list,
    plot_name: str,
    plot_path: str,
    scale: int = 5
):
    if params.get('c', False):
        params['c'] = [0.5 for _ in range(len(params['x']))]

    f = Faerun(scale=3000, clear_color='#ffffff', coords=False, view='free')
    f.add_scatter(
        'Properties',
        params,
        point_scale=15,
        colormap='ocean_r',
        series_title=['Class'],
        has_legend=True,
        categorical=[True],
        legend_labels=legend_labels,
        legend_title=['Source'],
    )

    os.makedirs(plot_path, exist_ok=True)
    f.plot(file_name=plot_name, path=plot_path, template='smiles')


if __name__ == "__main__":
    # TODO Add loading functions and DATA_FILE path
    df = pd.read_csv(DATA_FILE)
    df.columns
    df.rename(columns={'other': 'drugs'}, inplace=True)
