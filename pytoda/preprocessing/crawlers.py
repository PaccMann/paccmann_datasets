import urllib
import urllib.error as urllib_error
import urllib.request as urllib_request
from typing import Union

ZINC_DRUG_SEARCH_ROOT = 'http://zinc.docking.org/substances/search/?q='
ZINC_ID_SEARCH_ROOT = 'http://zinc.docking.org/substances/'

PUBCHEM_START = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/'
PUBCHEM_MID = '/property/'
PUBCHEM_END = '/TXT'


def get_smiles_from_zinc(drug: Union[str, int]) -> str:
    """
    Uses the ZINC databases to retrieve the SMILES of a ZINC ID (int) or a drug
    name (str). 

    Args:
        drug (Union[str, int]) -- A string with a drug name or an int of a ZINC
            ID.
    Returns:
        smiles (str) -- The SMILES string of the drug name or ZINC ID.
    """

    if type(drug) != str and type(drug) != int:
        raise TypeError(
            f'Please insert drug of type {{str, int}}, given was {type(drug)}'
            f'({drug}).'
        )

    if type(drug) == str:

        # Parse name, then retrieve ZINC ID from it
        stripped_drug = drug.strip()
        zinc_ids = []
        try:
            drug_url = urllib_request.pathname2url(stripped_drug)
            path = '{}{}'.format(ZINC_DRUG_SEARCH_ROOT, drug_url)
            response = urllib.request.urlopen(path)

            for line in response:
                line = line.decode(encoding='UTF-8').strip()
                if 'href="/substances/ZINC' in line:
                    zinc_ids.append(line.split('/')[-2])
            zinc_id = zinc_ids[0]

        except urllib_error.HTTPError:
            print(f'Did not find any result for drug: {drug}')
            return []

    elif type(drug) == int:
        zinc_id = str(drug)

    zinc_id_url = ZINC_ID_SEARCH_ROOT + zinc_id
    id_response = urllib_request.urlopen(zinc_id_url)

    for id_line in id_response:
        id_line = id_line.decode(encoding='UTF-8').strip()
        if 'id="substance-smiles-field" readonly value=' in id_line:
            smiles = id_line.split('"')[-2]

    return smiles


def get_smiles_from_pubchem(
    drug: str, use_isomeric: bool = True, kekulize: bool = False
) -> str:
    """

    Uses the PubChem database to retrieve the SMILES of a drug name (str). 

    Args:
        drug (str) -- A string with a drug name (or a PubChem ID as a string).
        use_isomeric (bool, optional) - If available, returns the isomeric
            SMILES, not the canonical one.
        kekulize (bool, optional) -- Whether kekulization is used. PubChem uses
            kekulization per default, so setting this to 'True' will not
            perform any operation on the retrieved SMILES. 
            NOTE: Setting it to 'False' will convert aromatic atoms to lower-
            case characters and *induces a RDKit dependency*
    Returns:
        smiles (str) -- The SMILES string of the drug name.
    """

    if type(drug) != str:
        raise TypeError(
            f'Please insert drug of type str, given was {type(drug)}({drug}).'
        )
    if not kekulize:
        from rdkit import Chem

    options = ['CanonicalSMILES']
    if use_isomeric:
        options = ['IsomericSMILES'] + options

    # Parse name
    stripped_drug = drug.strip()

    # Search ZINC for compound name
    for option in options:
        try:
            path = '{}{}{}{}{}'.format(
                PUBCHEM_START, stripped_drug, PUBCHEM_MID, option, PUBCHEM_END
            )
            smiles = urllib_request.urlopen(path).read(
            ).decode('UTF-8').replace('\n', '')
            if not kekulize:
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            return smiles
        except urllib_error.HTTPError:
            if option == 'CanonicalSMILES':
                print(f'Did not find any result for drug: {drug}')
                return []
            continue
