"""Amino Acid Sequence transforms."""
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from ..files import read_smi
from ..transforms import Transform
from ..types import Indexes
from .processing import REAL_AAS
from .protein_language import ProteinLanguage


class SequenceToTokenIndexes(Transform):
    """Transform Sequence to token indexes using Sequence language."""

    def __init__(self, protein_language: ProteinLanguage) -> None:
        """
        Initialize a Sequence to token indexes object.

        Args:
            protein_language (ProteinLanguage): a Protein language.
        """
        self.protein_language = protein_language

    def __call__(self, smiles: str) -> Indexes:
        """
        Apply the Sequence tokenization transformation

        Args:
            smiles (str): a Sequence representation.

        Returns:
            Indexes: indexes representation for the Sequence provided.
        """
        return self.protein_language.sequence_to_token_indexes(smiles)


class ReplaceByFullProteinSequence(Transform):
    """
    A transform to replace short amino acid sequences with the full protein sequence.
    For example, replace active site sequence of a kinase with its full sequence.
    """

    def __init__(self, alignment_path: Union[str, Path]) -> None:
        """
        Loads alignment info with two "classes" (or types) of residues.

        Args:
            alignment_path (str): path to `.smi` or `.tsv` file which allows to map
                between shortened and full, aligned sequences. Do not use a header in
                the file.

                NOTE: By convention, residues in upper case are important and will be
                kept and residues in lower case are less important and are (usually)
                discarded.
                NOTE: The first column has to be the full protein sequence (use upper
                case only for residues to be used). E.g., ggABCggDEFgg
                NOTE: The second column has to be the condensed sequence (ABCDEF).
                NOTE: The third column has to be a protein id (can be duplicated).
        """

        if not (isinstance(alignment_path, str) or isinstance(alignment_path, Path)):
            raise TypeError(
                f"alignment_path must be string or Path, not {type(alignment_path)}"
            )
        self.alignment_path = alignment_path

        alignment_info = read_smi(
            self.alignment_path,
            index_col=None,
            header=None,
            names=['full_sequence', 'short_sequence', 'id'],
        )
        # We use a combination of ID and the shortened sequence as keys to
        # enable support if proteins have >1 short sequence.
        self.id_to_full = dict(
            zip(
                alignment_info['id'] + '_' + alignment_info['short_sequence'],
                alignment_info['full_sequence'],
            )
        )
        if len(self.id_to_full) < len(alignment_info):
            raise ValueError(f'Duplicate IDs not allowed in: {self.alignment_path}')

    def __call__(self, sample_dict: Dict) -> str:
        """
        Replace the shortened sequence (usually uppercase only) with an aligned
        sequence where usually uppercase is for residues of interest and lowercase
        for the remaining ones.

        Args:
            sample_dict (Dict): a dictionary with the following keys:
                - 'id': the protein id.
                - 'sequence': the shortened protein sequence. (E.g., ABCDEF)
            NOTE: This has to be a dictionary because otherwise the shortened protein
                sequence has to be unique.

        Returns:
            str: the full protein sequence (e.g., abABChijDEF).
        """
        return self.id_to_full[f"{sample_dict['id']}_{sample_dict['sequence']}"]


def extract_active_sites_info(
    aligned_seq: str,
) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Processes and extracts useful information from an aligned protein sequence.
    Expects lower case amino acids to be outside of the relevant area (e.g., active site)
    and upper case amino acids to be inside it.

    Args:
        aligned_seq: A string containing the aligned protein sequence including
            lower case amino acids and high case amino acids.

    Returns:
        4-Tuple of:
        aligned_seq (str): The input sequence.
        non_active_sites (List[str]): A list of strings, one item for each contiguous
            subsequence NOT belonging to active site.
        active_sites (List[str]): A list of strings, one item for each contiguous
            subsequence belonging to active site.
        all_seqs (List[str]): A list of strings, one item for each contiguous
            subsequence that either belongs to the active site or not.
    """

    non_active_sites = ''
    active_sites = ''
    prev_was_highcase = False
    for c in aligned_seq:
        next_is_highcase = c <= 'Z'
        if next_is_highcase ^ prev_was_highcase:
            if next_is_highcase:
                active_sites += '#'
            else:
                non_active_sites += '#'

        if next_is_highcase:
            active_sites += c
            prev_was_highcase = True
        else:
            non_active_sites += c
            prev_was_highcase = False

    non_active_sites = [s for s in non_active_sites.split('#') if s != '']
    active_sites = [s for s in active_sites.split('#') if s != '']

    if aligned_seq[0] <= 'Z':
        zip_obj = zip(active_sites, non_active_sites)
    else:
        zip_obj = zip(non_active_sites, active_sites)

    all_seqs = [i for one_tuple in zip_obj for i in one_tuple]

    if len(active_sites) > len(non_active_sites):
        assert len(active_sites) == len(non_active_sites) + 1
        all_seqs.append(active_sites[-1])
    elif len(active_sites) < len(non_active_sites):
        assert len(active_sites) + 1 == len(non_active_sites)
        all_seqs.append(non_active_sites[-1])

    return aligned_seq, non_active_sites, active_sites, all_seqs


def verify_aligned_info(sequence: str) -> None:
    """
    Verify that the sequence is aligned.

    Args:
        sequence: An amino acid sequence.

    Raises:
        Exception: If alignment could not be detected.
    """
    isinstance(sequence, str)
    found_lower_case = False
    found_upper_case = False
    for c in sequence:
        if c >= 'A' and c <= 'Z':
            found_upper_case = True
        elif c >= 'a' and c <= 'z':
            found_lower_case = True
    if not (found_lower_case and found_upper_case):
        raise Exception(
            'Expected aligned residues sequence! Did you forget to use ReplaceByFullProteinSequence?'
        )


class ProteinAugmentFlipSubstrs(Transform):
    """Augment a protein sequence by randomly flipping each contiguous subsequence."""

    def __init__(self, p: float = 0.5) -> None:
        """
        Args:
            p (float): Probability that reverting occurs.
        """
        if not isinstance(p, float):
            raise TypeError(f'Please pass float, not {type(p)}.')
        self._p = np.clip(p, 0.0, 1.0)

    def __call__(self, sequence: str) -> str:
        """
        Apply the transform.

        Args:
            sequence (str): an aligned sequence (example: abcDEfgHI).

        Returns:
            str: an aligned sequence with optional flipping (example: abcEDfgHI).
        """
        verify_aligned_info(sequence)
        (
            aligned_seq,
            non_active_sites,
            active_sites,
            all_seqs,
        ) = extract_active_sites_info(sequence)

        ans = ''
        for substr in all_seqs:
            if substr[0] <= 'Z':
                if np.random.rand() < self._p:
                    ans += substr[::-1]
                else:
                    ans += substr
            else:
                ans += substr

        return ans


class MutateResidues(Transform):
    """
    Augment a protein sequence by injecting (possibly different) noise to residues
    inside and outside the relevant part (e.g., active site).
    NOTE: Noise means single-residue point mutations.
    """

    def __init__(self, mutate_upper: float = 0.01, mutate_lower: float = 0.1) -> None:
        """
        Args:
            mutate_lower (float): probability for mutating lowercase residues
            mutate_upper (float): probability for mutating uppercase residues.
        """

        if not isinstance(mutate_upper, float):
            raise TypeError(
                f'Please pass float for mutate_prob_in_active_site, not {type(mutate_upper)}.'
            )
        self.mutate_upper = mutate_upper

        if not isinstance(mutate_lower, float):
            raise TypeError(
                f'Please pass float for mutate_lower, not {type(mutate_lower)}.'
            )
        self.mutate_lower = mutate_lower
        self.num_aas = len(REAL_AAS)

    def __call__(self, sequence: str) -> str:
        """
        Apply the transform.

        Args:
            sequence (str): an aligned sequence (example: acDEFg).

        Returns:
            str: a possibly mutated aligned sequence (example: afDEFg).
        """
        # import ipdb;ipdb.set_trace()
        verify_aligned_info(sequence)
        (
            aligned_seq,
            non_active_sites,
            active_sites,
            all_seqs,
        ) = extract_active_sites_info(sequence)

        ans = ''
        for curr_sub_seq in all_seqs:
            for c in curr_sub_seq:
                if (
                    curr_sub_seq[0] <= 'Z'
                ):  # it's uppercase, so it's inside an active site
                    if np.random.rand() < self.mutate_upper:
                        ans += REAL_AAS[np.random.randint(self.num_aas)]
                    else:
                        ans += c
                else:
                    if np.random.rand() < self.mutate_lower:
                        ans += REAL_AAS[np.random.randint(self.num_aas)].lower()
                    else:
                        ans += c

        return ans


class ProteinAugmentSwapSubstrs(Transform):
    """Augment a protein sequence by randomly swapping neighboring subsequences."""

    def __init__(self, p: float = 0.2) -> None:
        """
        Args:
            p (float): Probability that any substr switches places with its "neighbour".

        """
        if not isinstance(p, float):
            raise TypeError(f'Please pass float, not {type(p)}.')
        self._p = np.clip(p, 0.0, 1.0)

    def __call__(self, sequence: str) -> str:
        """
        Apply the transform.

        Args:
            sequence (str): an aligned sequence (example: abCDefGHi).

        Returns:
            str: an aligned sequence with swapped substrings (example: abGHefCDi).
        """
        verify_aligned_info(sequence)
        (
            aligned_seq,
            non_active_sites,
            active_sites,
            all_seqs,
        ) = extract_active_sites_info(sequence)

        order = list(range(len(active_sites)))

        for pos in range(len(order) - 1):
            if np.random.rand() < self._p:
                # switch
                order[pos], order[pos + 1] = order[pos + 1], order[pos]

        curr_active_site_substr_idx = -1
        ans = ''
        for substr in all_seqs:
            if substr[0] <= 'Z':
                curr_active_site_substr_idx += 1
                ans += active_sites[order[curr_active_site_substr_idx]]
            else:
                ans += substr

        return ans
