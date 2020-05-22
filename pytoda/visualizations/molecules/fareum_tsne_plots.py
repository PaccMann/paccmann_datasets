import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
from faerun import Faerun
import numpy as np
import umap

from typing import Optional, List


def compute_tsne(
    data: np.ndarray,
    number_components: int = 3,
    perplexity: int = 70,
    n_iter: int = 1000
) -> List[np.ndarray]:
    out = TSNE(
        n_components=number_components, perplexity=perplexity, n_iter=n_iter
    ).fit_transform(data)
    return [out[:, i] for i in range(out.shape[1])]


def tsne_from_fingerprints(
    df: pd.DataFrame,
    tsne_params: dict = {'number_components': 3},
    radius: int = 2,
    nBits: int = 256
) -> List[np.ndarray]:
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


def _load_properties_file(filename: str) -> dict:
    data = {}
    # NOTE Assuming that the ids could be unordered
    # REVIEW Plaintext is probably not the best formate to store this data
    with open(filename, 'r') as f:
        for line in f.readlines():
            idx, latent = line.split('\t')
            latent = [float(x) for x in latent.split(',')]
            data[int(idx)] = latent
    return data


def _match_properties(properties: dict, dataframe: pd.DataFrame) -> list:
    data = []
    # NOTE need to enforce (check) that the IDs are conserved between files
    for i in dataframe.index:
        # Should I account for those samples w/o data? (e.g. malformed SMILEs)
        data.append(properties[i])
    return data


def tsne_from_properties_file(
    df: pd.DataFrame,
    properties_file: str,
    tsne_params: dict = {'number_components': 3},
) -> List[np.ndarray]:
    properties = _load_properties_file(properties_file)
    properties_data = _match_properties(properties, df)

    return compute_tsne(properties_data, **tsne_params)


def compute_umap(
    data: np.ndarray,
    number_components: int = 3,
    min_dist: float = 0.25,
    n_neighbors: int = 50
) -> List[np.ndarray]:
    out = umap.UMAP(n_components=number_components).fit_transform(data)
    return [out[:, i] for i in range(out.shape[1])]


def umap_from_properties_file(
    df: pd.DataFrame,
    properties_file: str,
    umap_params: dict = {
        'number_components': 3,
        'n_neighbors': 50,
        'min_dist': 0.25
    },
) -> List[np.ndarray]:
    properties = _load_properties_file(properties_file)
    properties_data = _match_properties(properties, df)

    return compute_umap(properties_data, **umap_params)


def umap_from_fingerprints(
    df: pd.DataFrame,
    umap_params: dict = {
        'number_components': 3,
        'n_neighbors': 50,
        'min_dist': 0.25
    },
    radius: int = 2,
    nBits: int = 256
) -> List[np.ndarray]:
    fingerprints = []
    for val in df.SMILES.values:
        mol = Chem.MolFromSmiles(val)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=nBits
        )
        fingerprint = [int(x) for x in fingerprint.ToBitString()]
        fingerprints.append(fingerprint)
    fingerprints = np.array(fingerprints)

    return compute_umap(fingerprints, **umap_params)


def fareum_from_pandas(
    df: pd.DataFrame,
    x: List[float],
    y: List[float],
    z: Optional[List[float]] = None,
    smiles_col: Optional[str] = None,
    names_col: Optional[str] = None,
    plot_name: str = 'fareum_plot',
    plot_path: str = 'fareum'
):
    smiles = names = None
    if smiles_col in df.columns:
        smiles = df[smiles_col]
    if names_col in df.columns:
        names = df[names_col]

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
    smiles: Optional[List[str]] = None,
    names: Optional[List[str]] = None
):
    params = {'x': x, 'y': y, 'c': c if c else [0.5 for _ in range(len(x))]}
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
    from pytoda.visualizations.molecules.tmap_plotter import fareum_plot
    DATA_FILE = os.path.expanduser(
        '~/Box/Molecular_SysBio/data/paccmann/'
        'paccmann_affinity/all_molecules.csv'
    )
    ENCODE_FILE = os.path.expanduser(
        '~/Box/Molecular_SysBio/data/paccmann/'
        'paccmann_affinity/samples_latent_code.tsv'
    )

    df = pd.read_csv(DATA_FILE)

    # TODO Clean this file and tmap_plotter

    continous_columns = [
        x for x in list(df.columns) if x not in
        ['Unnamed: 0', 'Unnamed: 0.1', 'source', 'SMILES', 'other']
    ]

    output_tsne = tsne_from_fingerprints(df)
    fareum_plot(
        df,
        output_tsne,
        smiles_column='SMILES',
        drugs_column='other',
        categorical_columns=['source'],
        continous_columns=continous_columns,
        plot_filename='tsne_fingerprint'
    )

    output_tsne = tsne_from_properties_file(df, ENCODE_FILE)
    fareum_plot(
        df,
        output_tsne,
        smiles_column='SMILES',
        drugs_column='other',
        categorical_columns=['source'],
        continous_columns=continous_columns,
        plot_filename='tsne_latentcode'
    )

    print('Done')
