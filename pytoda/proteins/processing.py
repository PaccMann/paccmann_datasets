'Amino Acid sequence processing utilities.'
from collections import OrderedDict

IUPAC_CODES = OrderedDict(
    [
        ('Ala', 'A'),
        ('Asx', 'B'),  # Aspartate or Asparagine
        ('Cys', 'C'),
        ('Asp', 'D'),
        ('Glu', 'E'),
        ('Phe', 'F'),
        ('Gly', 'G'),
        ('His', 'H'),
        ('Ile', 'I'),
        ('Lys', 'K'),
        ('Leu', 'L'),
        ('Met', 'M'),
        ('Asn', 'N'),
        ('Hyp', 'O'),
        ('Pro', 'P'),
        ('Gln', 'Q'),
        ('Arg', 'R'),
        ('Ser', 'S'),
        ('Thr', 'T'),
        ('Sec', 'U'),
        ('Val', 'V'),
        ('Trp', 'W'),
        ('Xaa', 'X'),  # Any AA
        ('Tyr', 'Y'),
        ('Glx', 'Z')  # Glutamate or Glutamine
    ]
)

IUPAC_VOCAB = OrderedDict(
    [
        ('<PAD>', 0),
        ('<MASK>', 1),
        ('<CLS>', 2),
        ('<SEP>', 3),
        ('<UNK>', 4),
        ('A', 5),
        ('B', 6),
        ('C', 7),
        ('D', 8),
        ('E', 9),
        ('F', 10),
        ('G', 11),
        ('H', 12),
        ('I', 13),
        ('K', 14),
        ('L', 15),
        ('M', 16),
        ('N', 17),
        ('O', 18),
        ('P', 19),
        ('Q', 20),
        ('R', 21),
        ('S', 22),
        ('T', 23),
        ('U', 24),
        ('V', 25),
        ('W', 26),
        ('X', 27),
        ('Y', 28),
        ('Z', 29),
        ('<START>', 30),
        ('<STOP>', 31),
    ]
)

UNIREP_VOCAB = OrderedDict(
    [
        ('<PAD>', 0),
        ('M', 1),
        ('R', 2),
        ('H', 3),
        ('K', 4),
        ('D', 5),
        ('E', 6),
        ('S', 7),
        ('T', 8),
        ('N', 9),
        ('Q', 10),
        ('C', 11),
        ('U', 12),
        ('G', 13),
        ('P', 14),
        ('A', 15),
        ('V', 16),
        ('I', 17),
        ('F', 18),
        ('Y', 19),
        ('W', 20),
        ('L', 21),
        ('O', 22),
        ('X', 23),
        ('Z', 23),
        ('B', 23),
        ('J', 23),
        ('<CLS>', 24),
        ('<SEP>', 25),
        ('<START>', 26),
        ('<STOP>', 27),
        ('<UNK>', 28),
    ]
)
"""
Polarity, Charge, Hydophobicity, Aromaticity, Ionizability, StartStop
Nonpolar = -1, Polar = 1
Negative = -1, Neutral = 0, Positive = 1
Hydrophobic = -1, Hydrophilic = 1
NonAromatic = -1, Aromatic = 1
NonIonizable = -1, Ionizable = 1
Stop = -1, Start = 1, Neither = 0
"""
AA_PROPERTIES_NUM = OrderedDict(
    [
        ('<PAD>', (0, 0, 0, 0, 0, 0)),
        ('A', (-1, 0, -1, -1, -1, 0)),
        ('B', (1, -0.5, 1, -1, 0, 0)),  # mean of D, N
        ('C', (1, 0, 1, -1, 1, 0)),
        ('D', (1, -1, 1, -1, 1, 0)),
        ('E', (1, -1, 1, -1, 1, 0)),
        ('F', (-1, 0, -1, 1, -1, 0)),
        ('G', (-1, 0, -1, -1, -1, 0)),
        ('H', (1, 1, 1, -1, 1, 0)),
        ('I', (-1, 0, -1, -1, -1, 0)),
        ('K', (1, 1, 1, -1, 1, 0)),
        ('L', (-1, 0, -1, -1, -1, 0)),
        ('M', (-1, 0, -1, -1, -1, 0)),
        ('N', (1, 0, 1, -1, -1, 0)),
        ('P', (-1, 0, -1, -1, -1, 0)),
        ('Q', (1, 0, 1, -1, -1, 0)),
        ('R', (1, 1, 1, -1, 1, 0)),
        ('S', (1, 0, 1, -1, -1, 0)),
        ('T', (1, 0, 1, -1, -1, 0)),
        # ('U', (1, 0, 1, -1, 1, 0)),  # Selenocyteine - check this again!
        ('V', (-1, 0, -1, -1, -1, 0)),
        ('W', (-1, 0, -1, 1, -1, 0)),
        ('Y', (1, 0, -1, 1, 1, 0)),
        ('X', (0.2, 0, 0.1, -0.7, -0.2, 0)),  # mean AA (Unknown)
        ('Z', (1, -0.5, 1, -1, 0, 0)),  # mean of E, Q
        ('<START>', (0, 0, 0, 0, 0, 1)),
        ('<STOP>', (0, 0, 0, 0, 0, -1)),
    ]
)
"""
Molecular Weight, Residue Weight, pKa, pKb, pKx, pI, Hydrophobicity at pH2
"""
AA_FEAT = OrderedDict(
    [
        ('<PAD>', (0, 0, 0, 0, 0, 0, 0, 0)),
        ('A', (89.1, 71.08, 2.34, 9.69, 0, 6, 47, 0)),
        ('B', (132.615, 114.6, 1.95, 9.2, 1.825, 4.09, -29.5, 0)),  # D/N mean
        ('C', (121.16, 103.15, 1.96, 10.28, 8.18, 5.07, 52, 0)),
        ('D', (133.11, 115.09, 1.88, 9.6, 3.65, 2.77, -18, 0)),
        ('E', (147.13, 129.12, 2.19, 9.67, 4.25, 3.22, 8, 0)),
        ('F', (165.19, 147.18, 1.83, 9.13, 0, 5.48, 92, 0)),
        ('G', (75.07, 57.05, 2.34, 9.6, 0, 5.97, 0, 0)),
        ('H', (155.16, 137.14, 1.82, 9.17, 6, 7.59, -42, 0)),
        ('I', (131.18, 113.16, 2.36, 9.6, 0, 6.02, 100, 0)),
        ('K', (146.19, 128.18, 2.18, 8.95, 10.53, 9.74, -37, 0)),
        ('L', (131.18, 113.16, 2.36, 9.6, 0, 5.98, 100, 0)),
        ('M', (149.21, 131.2, 2.28, 9.21, 0, 5.74, 74, 0)),
        ('N', (132.12, 114.11, 2.02, 8.8, 0, 5.41, -41, 0)),
        ('O', (131.13, 113.11, 1.82, 9.65, 0, 0, 0, 0)),
        ('P', (115.13, 97.12, 1.99, 10.6, 0, 6.3, -46, 0)),
        ('Q', (146.15, 128.13, 2.17, 9.13, 0, 5.65, -18, 0)),
        ('R', (174.2, 156.19, 2.17, 9.04, 12.48, 10.76, -26, 0)),
        ('S', (105.09, 87.08, 2.21, 9.15, 0, 5.68, -7, 0)),
        ('T', (119.12, 101.11, 2.09, 9.1, 0, 5.6, 13, 0)),
        # ('U', (, 0)) #Selenocyteine - check this again!
        ('V', (117.15, 99.13, 2.32, 9.62, 0, 5.96, 97, 0)),
        ('W', (204.23, 186.22, 2.83, 9.39, 0, 5.89, 84, 0)),
        ('Y', (181.19, 163.18, 2.2, 9.11, 10.07, 5.66, 49, 0)),
        ('X', (136.74, 118.73, 2.06, 9.00, 2.51, 5.74, 21.86,
               0)),  # What do we set for any aa?
        ('Z', (146.64, 128.625, 2.18, 9.4, 2.125, 4.435, -5,
               0)),  # mean of E, Q
        ('<START>', (0, 0, 0, 0, 0, 0, 0, 1)),
        ('<STOP>', (0, 0, 0, 0, 0, 0, 0, -1))
    ]
)

BLOSUM62 = OrderedDict(
    [
        (
            '<PAD>', (
                -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
                -4, -4, -4, -4, -4, -4, -4, -4, 10, -4, -4
            )
        ),
        (
            'A', (
                4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0,
                -3, -2, 0, -2, -1, 0, -4, -4, -4, -4
            )
        ),
        (
            'B', (
                -2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1,
                -4, -3, -3, 4, 1, -1, -4, -4, -4, -4
            )
        ),
        (
            'C', (
                0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1,
                -1, -2, -2, -1, -3, -3, -2, -4, -4, -4, -4
            )
        ),
        (
            'D', (
                -2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1,
                -4, -3, -3, 4, 1, -1, -4, -4, -4, -4
            )
        ),
        (
            'E', (
                -1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3,
                -2, -2, 1, 4, -1, -4, -4, -4, -4
            )
        ),
        (
            'F', (
                -2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2,
                1, 3, -1, -3, -3, -1, -4, -4, -4, -4
            )
        ),
        (
            'G', (
                0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2,
                -2, -3, -3, -1, -2, -1, -4, -4, -4, -4
            )
        ),
        (
            'H', (
                -2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2,
                -2, 2, -3, 0, 0, -1, -4, -4, -4, -4
            )
        ),
        (
            'I', (
                -1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1,
                -3, -1, 3, -3, -3, -1, -4, -4, -4, -4
            )
        ),
        (
            'K', (
                -1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1,
                -3, -2, -2, 0, 1, -1, -4, -4, -4, -4
            )
        ),
        (
            'L', (
                -1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1,
                -2, -1, 1, -4, -3, -1, -4, -4, -4, -4
            )
        ),
        (
            'M', (
                -1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1,
                -1, -1, 1, -3, -1, -1, -4, -4, -4, -4
            )
        ),
        (
            'N', (
                -2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4,
                -2, -3, 3, 0, -1, -4, -4, -4, -4
            )
        ),
        (
            'P', (
                -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1,
                -1, -4, -3, -2, -2, -1, -2, -4, -4, -4, -4
            )
        ),
        (
            'Q', (
                -1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2,
                -1, -2, 0, 3, -1, -4, -4, -4, -4
            )
        ),
        (
            'R', (
                -1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1,
                -3, -2, -3, -1, 0, -1, -4, -4, -4, -4
            )
        ),
        (
            'S', (
                1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3,
                -2, -2, 0, 0, 0, -4, -4, -4, -4
            )
        ),
        (
            'T', (
                0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5,
                -2, -2, 0, -1, -1, 0, -4, -4, -4, -4
            )
        ),
        (
            'V', (
                0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0,
                -3, -1, 4, -3, -2, -1, -4, -4, -4, -4
            )
        ),
        (
            'W', (
                -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3,
                -2, 11, 2, -3, -4, -3, -2, -4, -4, -4, -4
            )
        ),
        (
            'X', (
                0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0,
                0, -2, -1, -1, -1, -1, -1, -4, -4, -4, -4
            )
        ),
        (
            'Y', (
                -2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2,
                -2, 2, 7, -1, -3, -2, -1, -4, -4, -4, -4
            )
        ),
        (
            'Z', (
                -1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3,
                -2, -2, 1, 4, -1, -4, -4, -4, -4
            )
        ),
        (
            '<START>', (
                -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
                -4, -4, -4, -4, -4, -4, -4, -4, -4, 10, -4
            )
        ),
        (
            '<STOP>', (
                -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
                -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 10
            )
        )
    ]
)
