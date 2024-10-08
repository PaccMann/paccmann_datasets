import logging
import urllib.request as urllib_request
from itertools import filterfalse
from typing import Iterable, List, Tuple, Union
from urllib.error import HTTPError, URLError

import pubchempy as pcp
from pubchempy import BadRequestError, PubChemHTTPError
from unidecode import unidecode

from pytoda.smiles.transforms import Canonicalization

logger = logging.getLogger(__name__)

PUBCHEM_START = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound'
PUBCHEM_MID = 'property'
PUBCHEM_END = 'TXT'


def get_smiles_from_pubchem(
    drug: Union[str, int],
    query_type: str = 'name',
    use_isomeric: bool = True,
    kekulize: bool = False,
    sanitize: bool = True,
) -> str:
    """
    Uses the PubChem database to retrieve the SMILES of a drug name given as string
    (default) or a PubChem ID.

    Args:
        drug (str): string with a drug name (or a PubChem ID as a string).
        query_type (str): Either 'name' or 'cid'.
            Identifies whether the argument provided as drug is a name (e.g 'Tacrine') or
            a PubChem ID (1935). Defaults to name.
        use_isomeric (bool, optional) - If available, returns the isomeric
            SMILES, not the canonical one.
        kekulize (bool, optional): whether kekulization is used. PubChem uses
            kekulization per default, so setting this to 'True' will not
            perform any operation on the retrieved SMILES.
            NOTE: Setting it to 'False' will convert aromatic atoms to lower-
            case characters and *induces a RDKit dependency*
        sanitize (bool, optional): Sanitize SMILE
    Returns:
        smiles (str): The SMILES string of the drug name.
    """

    if not kekulize and not sanitize:
        raise ValueError(
            'If Kekulize is False, molecule has to be sanitize '
            '(sanitize cannot be False).'
        )

    if type(drug) != str and type(drug) != int:
        raise TypeError(
            f'Please insert drug of type str or int, given was {type(drug)}({drug}).'
        )
    if not kekulize:
        from rdkit import Chem

    options = ['CanonicalSMILES']
    if use_isomeric:
        options = ['IsomericSMILES'] + options

    # Parse name
    if isinstance(drug, str):
        drug = unidecode(drug).strip().replace(' ', '%20')

    # Search in PubChem for compound name
    for option in options:
        try:
            path = '{}/{}/{}/{}/{}/{}'.format(
                PUBCHEM_START, query_type, drug, PUBCHEM_MID, option, PUBCHEM_END
            )
            smiles = urllib_request.urlopen(path).read().decode('UTF-8').split()[0]
            if not kekulize:
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=sanitize))
            return smiles
        except HTTPError:
            if option == 'CanonicalSMILES':
                logger.warning(f'Did not find any result for drug: {drug}')
                return ''
            continue


def remove_pubchem_smiles(smiles_list: Iterable[str]) -> List:
    """
    Function for removing PubChem molecules from an iterable of smiles.
    Args:
        smiles_list (Iterable[str]): many SMILES strings.
    Returns:
        List[str]:  Filtered list of SMILES, all SMILES pointing to PubChem
            molecules are removed.
    """

    if not isinstance(smiles_list, Iterable):
        raise TypeError(f'Please pass Iterable, not {type(smiles_list)}')

    canonicalizer = Canonicalization(sanitize=False)
    filtered = filterfalse(is_pubchem, smiles_list)
    # Canonicalize molecules and filter again (sanity check)
    filtered_canonical = filterfalse(lambda x: is_pubchem(canonicalizer(x)), filtered)
    return list(filtered_canonical)


def query_pubchem(smiles: str) -> Tuple[bool, int]:
    """
    Queries pubchem for a given SMILES.

    Args:
        smiles (str): A SMILES string.

    Returns:
        Tuple[bool, int]:
            bool: Whether or not SMILES is known to PubChem.

            int: PubChem ID of matched SMILES, -1 if SMILES was not found.
                Instead, -2 means an error in the PubChem query.
    """

    if not isinstance(smiles, str):
        raise TypeError(f'Please pass str, not {type(smiles)}')
    try:
        result = pcp.get_compounds(smiles, 'smiles')[0]
        return (False, -1) if result.cid is None else (True, result.cid)
    except BadRequestError:
        logger.warning(f'Skipping SMILES. BadRequestError with: {smiles}')
    except HTTPError:
        logger.warning(f'Skipping SMILES. HTTPError with: {smiles}')
    except TimeoutError:
        logger.warning(f'Skipping SMILES. TimeoutError with: {smiles}')
    except ConnectionResetError:
        logger.warning(f'Skipping SMILES. ConnectionResetError with: {smiles}')
    except PubChemHTTPError:
        logger.warning(f'Skipping SMILES, server busy. with: {smiles}')
    except URLError:
        logger.error(f"Skipping SMILES, Network unreachable {smiles}")
    return (False, -2)


def is_pubchem(smiles: str) -> bool:
    """Whether a given SMILES in PubChem.
    Args:
        smiles (str): A SMILES string.
    Returns:
        bool: Whether or not SMILES is known to PubChem.
    """
    return query_pubchem(smiles)[0]
