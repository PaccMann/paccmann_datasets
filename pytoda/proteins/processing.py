"""Amino Acid sequence processing utilities."""
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
