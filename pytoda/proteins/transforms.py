"""Amino Acid Sequence transforms."""
from ..transforms import Transform
from ..types import Indexes
from .protein_language import ProteinLanguage
import numpy as np
import torch
from ._smi_eager_dataset import _SmiEagerDataset

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


#### active site alignment info and related augmentation

class LoadActiveSiteAlignmentInfo(Transform):
    """Loads active site alignment info.
    residues outside the active site are lower case and residues inside the active site are upper case    
    """

    def __init__(self, active_site_alignment_info_smi:str) -> None:
        """
        An op to replace the active site sequence with an aligned sequence.
                
        Args:
            load_active_site_alignment_info:str - smi file path which allows to maps between active site sequence and aligned sequence.
            first column (0) is expected to contain "active_site_seq" - example content: LGKGTFGKVAKELLTLFVMEYANGGEFVVENMTDL
            second column (1) is expected to be named "aligned_protein_seq" - example content: feylklLGKGTFGKVilvkekatgryyAmKilkkevivakdevahtltEnrvLqnsrhpfLTaLkysfqthdrlcFVMEYANGGElfFhlsrervfsedrarfygaeivsaldylhseknVVyrdlklENlMldkdghikiTDfgLckegikdgatmktfcgtpeylapevledndygravdwwglgvvmyemmcgrlpfynqdheklfelilmeeirfprtlgpeaksllsgllkkdpkqrlgggsedakeimqhrff            
                lower case residues are outside the active site, upper case residues are inside the active site
        """        
        import ipdb;ipdb.set_trace()
        assert isinstance(smi_paths, str)
        self.active_site_alignment_info_smi = active_site_alignment_info_smi       
        self.active_sites_alignment_info_tbl = _SmiEagerDataset(
            self.active_site_alignment_info_smi,
            index_col=1, 
            name='aligned_protein_seq',
            names = ['aligned_protein_seq', 'active_site_seq', 'protein_id'],
        )

    def __call__(self, sequence: str) -> str:
        """
        Example expected input: LGKGTFGKVAKELLTLFVMEYANGGEFVVENMTDL
        Example expected output: feylklLGKGTFGKVilvkekatgryyAmKilkkevivakdevahtltEnrvLqnsrhpfLTaLkysfqthdrlcFVMEYANGGElfFhlsrervfsedrarfygaeivsaldylhseknVVyrdlklENlMldkdghikiTDfgLckegikdgatmktfcgtpeylapevledndygravdwwglgvvmyemmcgrlpfynqdheklfelilmeeirfprtlgpeaksllsgllkkdpkqrlgggsedakeimqhrff

        in the output: lower case residues are outside the active site, upper case residues are inside the active site
        """
        import ipdb;ipdb.set_trace()
        aligned_seq = self.active_sites_alignment_info_tbl[sequence]        
        return aligned_seq

def extract_active_sites_info(aligned_seq:str):
    '''
    processes and extracts useful information from an aligned active site sequence,
    expects low case amino acids to be outside of the active site and high case amino acids to be inside it
    '''
    non_active_sites = ''
    active_sites = ''
    #total_len = len(aligned_seq)
    prev_was_highcase = False
    for c in aligned_seq:            
        next_is_highcase = c<='Z'
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

    non_active_sites = [_ for _ in non_active_sites.split('#') if _!='']
    active_sites = [_ for _ in active_sites.split('#') if _!='']

    if aligned_seq[0]<='Z':
        zip_obj = zip(active_sites, non_active_sites)
    else:
        zip_obj = zip(non_active_sites, active_sites)

    all_seqs = [i  for one_tuple in zip_obj for i in one_tuple]

    if len(active_sites) > len(non_active_sites):
        assert len(active_sites) == len(non_active_sites)+ 1
        all_seqs.append(active_sites[-1])
    elif len(active_sites) < len(non_active_sites):
        assert len(active_sites) + 1 == len(non_active_sites)
        all_seqs.append(non_active_sites[-1])
        
    return aligned_seq, non_active_sites, active_sites, all_seqs

def verify_aligned_info(sequence:str):
    assert isinstance(sequence, str)
    found_lower_case = False
    found_upper_case = False
    for c in sequence:
        if c>='A' and c<='Z':
            found_upper_case = True
        elif c>='a' and c<='z':
            found_lower_case = True
    if not (found_lower_case and found_upper_case):
        raise Exception('Expected aligned residues sequence! Did you forget to use LoadActiveSiteAlignmentInfo?')

class ProteinAugmentFlipActiveSiteSubstrs(Transform):
    """Augment a kinase active-site sequence by randomly flipping each individual contiguous """
    def __init__(self,  p: float = 0.5) -> None:
        """
        Args:
            p (float): Probability that reverting occurs.

        """
        if not isinstance(p, float):
            raise TypeError(f'Please pass float, not {type(p)}.')
        self.p = np.clip(p, 0.0, 1.0)               

    def __call__(self, sequence: str) -> str:
        """
        Apply the transform.

        Args:
            sequence (str): an active-site aligned sequence (example: feylklLGKGTFGKVilvkekatgryyAmKilkkevivakdevahtltEnrvLqnsrhpfLTaLkysf)

        Returns:
            str: an active-site aligned sequence (example: feylklLGKGTFGKVilvkekatgryyAmKilkkevivakdevahtltEnrvLqnsrhpfLTaLkysf)
        """        
        import ipdb;ipdb.set_trace()
        verify_aligned_info(sequence)        
        aligned_seq, non_active_sites, active_sites, all_seqs = extract_active_sites_info(sequence)
        
        ans = ''
        for substr in all_seqs:
            if substr[0]<='Z':
                if np.random.rand()<self.p:
                    ans += substr[::-1]
                else:
                    ans += substr
            else:
                ans += substr
        
        return ans

G_AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

class ProteinAugmentActiveSiteGuidedNoise(Transform):
    """Augment a kinase active-site sequence by injecting (possibly different) noise to residues inside and outside an active site """

    def __init__(self,  
        mutate_prob_in_active_site=0.01, 
        mutate_prob_outside_active_site=0.1,
        ) -> None:
        """
        Args:
            mutate_prob_in_active_site:float - probability for mutating a residue INSIDE the active site
            mutate_prob_outside_active_site - probability for mutating a residue OUTSIDE the active site
        """
        
        if not isinstance(mutate_prob_in_active_site, float):
            raise TypeError(f'Please pass float for mutate_prob_in_active_site, not {type(mutate_prob_in_active_site)}.')
        self.mutate_prob_in_active_site = mutate_prob_in_active_site

        if not isinstance(mutate_prob_outside_active_site, float):
            raise TypeError(f'Please pass float for mutate_prob_outside_active_site, not {type(mutate_prob_outside_active_site)}.')
        self.mutate_prob_outside_active_site = mutate_prob_outside_active_site

    def __call__(self, sequence: str) -> str:
        """
        Apply the transform.

        Args:
            sequence (str): an active-site aligned sequence (example: feylklLGKGTFGKVilvkekatgryyAmKilkkevivakdevahtltEnrvLqnsrhpfLTaLkysf)

        Returns:
            str: an active-site aligned sequence (example: feylklLGKGTFGKVilvkekatgryyAmKilkkevivakdevahtltEnrvLqnsrhpfLTaLkysf)
        """
        import ipdb;ipdb.set_trace()
        verify_aligned_info(sequence)        
        aligned_seq, non_active_sites, active_sites, all_seqs = extract_active_sites_info(sequence)

        amino_acids_num = len(G_AMINO_ACIDS)                                     
        
        ans = ''
        for curr_sub_seq in all_seqs:
            for c in curr_sub_seq:
                if curr_sub_seq[0]<='Z': #it's uppercase, so it's inside an active site
                    if np.random.rand()<self.mutate_prob_in_active_site:
                        #print('mutate inside active site')
                        ans += G_AMINO_ACIDS[np.random.randint(amino_acids_num)]
                    else:
                        ans += c
                else:
                    if np.random.rand()<self.mutate_prob_outside_active_site:
                        #print('mutate outside active site')
                        ans += G_AMINO_ACIDS[np.random.randint(amino_acids_num)].lower()
                    else:
                        ans += c

        return ans

class KeepOnlyUpperCase(Transform):
    """Keeps only upper-case letters and discards the rest"""

    def __init__(self,  
        ) -> None:
        """        
        """        
        pass

    def __call__(self, sequence: str) -> str:
        """
        Apply the transform.

        Args:
            sequence (str): 

        Returns:
            str: 
        """
        ans = ''.join([x for x in sequence if (x>='A')and (x<='Z')])                
        return ans

class ToUpperCase(Transform):
    """convert all characters to uppercase"""

    def __init__(self,  
        ) -> None:
        """        
        """        
        pass

    def __call__(self, sequence: str) -> str:
        """
        Apply the transform.

        Args:
            sequence (str): 

        Returns:
            str: 
        """
        ans = ans.upper()      
        return ans