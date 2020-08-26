from rdkit import Chem

FORBIDDEN = set(['B', 'O', 'U', 'X', 'Z'])


def aas_to_smiles(aas, sanitize=True):
    """Converts an amino acid sequence (IUPAC) into SMILES.

    Args:
        aas (str): The amino acid sequence to be converted.
            Following IUPAC notation.
        sanitize (bool, optional): [description]. Defaults to True.

    Raises:
        TypeError: If aas is not a string.
        ValueError: If string cannot be converted to mol.

    Returns:
        smiles: SMILES string of the AA sequence.
    """
    if not isinstance(aas, str):
        raise TypeError(f'Provide string not {type(aas)}.')
    if len(set(aas).intersection(FORBIDDEN)) > 0:
        raise ValueError(
            f'Characters from: {FORBIDDEN} cant be parsed. Found one in: {aas}'
        )
    mol = Chem.MolFromFASTA(aas, sanitize=sanitize)
    if mol is None:
        raise ValueError(f'Sequence could not be converted to SMILES: {aas}')
    smiles = Chem.MolToSmiles(mol)
    return smiles
