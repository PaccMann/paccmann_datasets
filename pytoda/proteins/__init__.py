import pandas as pd
from importlib_resources import as_file, files

from .protein_feature_language import ProteinFeatureLanguage  # noqa
from .protein_language import ProteinLanguage  # noqa
from .utils import aas_to_smiles  # noqa

with as_file(
    files('pytoda.proteins.metadata').joinpath('kinase_activesite_alignment.smi')
) as alignment_filepath:
    kinase_as_alignment = pd.read_csv(alignment_filepath, sep='\t')
