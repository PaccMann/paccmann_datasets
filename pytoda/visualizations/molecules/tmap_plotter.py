"""Wrapper around TMAP"""
from typing import Iterable, List, Union
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as ss
import tmap as tm
from faerun import Faerun
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

import logging
logger = logging.getLogger(__name__)


def thumbnails_with_pubchem_reference(
    smiles: Iterable[str],
    drugs: Iterable[str] = None,
    titles: Iterable[str] = None,
) -> list:
    """Generates a set of Fareum labels, which will be used for the plot
     pop-up

    Args:
        smiles (Iterable[str]): Iterable with the SMILES
        drugs (Iterable[str]): Iterable with the drug names, if None it
            will fill it with with 'N.A.'. Default to None.
        titles (Iterable[str]): list of titles. If None, the SMILES will
            be used as titles instead. Defaults to None.

    Returns:
        list: List of the formatted labels
    """
    root = 'https://pubchem.ncbi.nlm.nih.gov/#query='

    if titles is None:
        titles = smiles

    if drugs is None:
        drugs = ['N.A.' for _ in range(len(smiles))]

    labels = []
    for ind, (smile, drug, title) in enumerate(zip(smiles, drugs, titles)):
        if drug != 'N.A.':
            labels.append(
                f'{title}__<a href="{root}{drug}">{drug}</a>__{smile}'
            )
        else:
            labels.append(f'{title}__No link available__{smile}')
    return labels


def tm_morgan_vector(smiles):
    return tm.VectorUint(
        AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smiles), radius=2, nBits=512
        )
    )


def rank_and_normalize_field(data: list) -> np.ndarray:
    """Ranks the data and normalizes it in [0, 1].

    Args:
        data (list): List of values

    Returns:
        np.array: [description]
    """
    return ss.rankdata(np.array(data)) / len(data)


def tmap(
    df: pd.DataFrame,
    categorical_columns: List[str] = [],
    continous_columns: List[str] = [],
    plot_folder: str = f'tmap_{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}',
    plot_filename: str = 'tmap',
    categorical_cmap: str = 'gist_rainbow',
    continous_cmap: str = 'viridis',
    shader: str = 'sphere',
    lshforest_dim: int = 512,
    lshforest_i: int = 32,
    thumbnail_titles: Union[Iterable[str], str] = None,
    store_data: bool = True
):
    """TMAP plotting utility for molecule data.

    Args:
        df (pd.DataFrame): Data.
        categorical_columns (List[str], optional): Names of the
            columns to be plotted that contain categorical values.
            Defaults to [] (i.e. none).
        continous_columns (List[str], optional): Names of the
            dataframe columns to be plotted that contain
            continous/numerical values. Defaults to [].
        plot_folder (str, optional): Folder where to store the plot.
            Defaults to `tmap_{datetime.now()}`.
        plot_filename (str, optional): Defaults to `tmap`.
        categorical_cmap (str, optional): Colormap for the categorical
            entries. Defaults to `gist_rainbow`. Available colormaps
            listed in the matplotlib documentation.
        continous_cmap (str, optional): Colormap for the continous
            entries. Defaults to `viridis`. Available colormaps listed
            in the matplotlib documentation.
        shader (str, optional): Shader for the nodes. Defaults to
            `sphere`.
        lshforest_dim (int, optional): Defaults to 512.
        lshforest_i (int, optional): Defaults to 32.
        thumbnail_titles (Union[Iterable[str], str], optional): List of
            titles for the thumbnails. If None it uses the SMILES as
            titles. Defaults to None.
        store_data (bool, optional): Store the TMAP tree. Defaults to
            True.

    Raises:
        KeyError: If column name (entry in `categorical_columns` or
        `categorical_columns`) is not in the dataframe.
    """
    # Check if all categories are in the dataframe
    for cat in [*categorical_columns, *continous_columns]:
        if cat not in df.columns:
            raise KeyError(f'Unknown column name {cat}')

    tmap_fps = list(df.SMILES.apply(tm_morgan_vector, convert_dtype=False))

    lf = tm.LSHForest(lshforest_dim, lshforest_i)
    lf.batch_add(tmap_fps)
    lf.index()

    # Extact columns that will be plotted
    categorical_values = [
        list(map(str, df[col])) for col in categorical_columns
    ]
    continous_values = [
        rank_and_normalize_field(df[col]) for col in continous_columns
    ]

    # Thubnails
    drugs = df.drugs if 'drugs' in df.columns else None
    thumbnail_titles = df[
        thumbnail_titles
    ] if thumbnail_titles in df.columns else thumbnail_titles

    thumbnails = thumbnails_with_pubchem_reference(
        df.SMILES, drugs, thumbnail_titles
    )

    # Store data
    if store_data:
        save_path = os.path.join(plot_folder, 'data')
        os.makedirs(save_path, exist_ok=True)

        lf.store(os.path.join(save_path, 'data.dat'))
        with open(os.path.join(save_path, 'properties.dat'), 'wb+') as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Create categories for all the categorical columns
    graph_categorical_labels = []
    graph_categorical = []
    bin_cmap = []
    for cat in categorical_values:
        _lab, _grp = Faerun.create_categories(cat)
        graph_categorical_labels.append(_lab)
        graph_categorical.append(_grp)
        bin_cmap.append(plt.cm.get_cmap(categorical_cmap, len(set(_grp))))

    # Layout settings
    cfg = tm.LayoutConfiguration()
    cfg.k = 20
    cfg.sl_extra_scaling_steps = 10
    # cfg.sl_repeats = 12
    # cfg.mmm_repeats = 2
    cfg.node_size = 50

    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, config=cfg)
    x = list(x)
    y = list(y)
    s = list(s)
    t = list(t)
    pickle.dump(
        (x, y, s, t),
        open(os.path.join(save_path, 'coords.dat'), 'wb+'),
        protocol=pickle.HIGHEST_PROTOCOL
    )

    def _create_faerum_opts(options: list, lenghts: list) -> list:
        """Helper function that creates a list of distint option
        arguments for the categorical and the numerical variables.

        Args:
            options (list): Distinct options for categorical and & num.
            lenghts (list): Lenghts of the different types of columns,
                namely: lenght of categorical vars & lenght of the
                numerical vars.

        Returns:
            list: [description]
        """
        out = []
        for opt, l in zip(options, lenghts):
            out.extend(l * [opt])
        return out

    types_columns_lengths = [len(categorical_values), len(continous_values)]
    is_categorical = _create_faerum_opts([True, False], types_columns_lengths)

    color_maps = [*bin_cmap, *(len(continous_values) * ['viridis'])]

    f = Faerun(
        clear_color='#222222',
        coords=False,
        view='front',
        impress=(
            'made with <a href="http://tmap.gdb.tools" target="_blank">tmap</a>'
            '<br />and <a href="https://github.com/reymond-group/faerun-python"'
            'target="_blank">faerun</a>'
        )
    )

    f.add_scatter(
        'molecules',
        {
            'x': x,
            'y': y,
            'c': [*graph_categorical, *continous_values],
            'labels': thumbnails
        },
        shader='sphere',
        colormap=color_maps,
        point_scale=20,
        max_point_size=100,
        categorical=is_categorical,
        has_legend=True,
        legend_labels=graph_categorical_labels,
        selected_labels=['SMILES', 'Dashboard', 'Name'],
        series_title=[*categorical_columns, *continous_columns],
        max_legend_label=[
            None,
        ],
        min_legend_label=[
            None,
        ],
        title_index=2,
        legend_title='',
    )

    f.add_tree(plot_filename, {'from': s, 'to': t}, point_helper='molecules')
    os.makedirs(plot_folder, exist_ok=True)
    f.plot(file_name=plot_filename, path=plot_folder, template='smiles')


if __name__ == "__main__":
    logger.info('Running example TMAP plot with Paccmann data')

    # Read data and rename columns
    drug_df = pd.read_csv(
        os.path.join(
            os.path.expanduser('~'), 'Box', 'Molecular_SysBio', 'data',
            'paccmann', 'paccmann_rl', 'panel_drugs.csv'
        ),
        index_col=0
    )
    gen_df = pd.read_csv(
        os.path.join(
            os.path.expanduser('~'), 'Box', 'Molecular_SysBio', 'data',
            'paccmann', 'paccmann_rl', 'biased_models',
            'best_generated_drugs.csv'
        ),
        index_col=0
    )
    gen_df['drug'] = 'N.A.'
    gen_df['source'] = 'Generated'
    drug_df['source'] = 'GDSC/CCLE'
    drug_df.rename(
        columns={
            'IC50_best_site': 'IC50',
            'scscore': 'SCScore'
        }, inplace=True
    )
    gen_df.rename(
        columns={
            'cell_line': 'cancer_site',
            'esol': 'ESOL'
        }, inplace=True
    )

    # Join the dataframes
    mol_df = pd.concat([gen_df, drug_df], join='inner')

    # Select site
    site = 'prostate'

    df = mol_df[mol_df.cancer_site == site]

    # Plot
    tmap(
        df,
        categorical_columns=['source'],
        continous_columns=[
            'IC50', 'QED', 'SAS', 'mol_weight', 'ESOL', 'SCScore'
        ],
        plot_folder='example_tmap',
        plot_filename='example'
    )
    logger.info('Plot done.')
