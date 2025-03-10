import jaxtyping as jtp
import numpy as np

CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
    'p', 'n', 'r', 'k', 'q',
    'P', 'B', 'N', 'R', 'Q', 'K',
    'w', 'b', '-', '.'
]

CHARS_INDEX = {c : i for i, c in enumerate(CHARS)}
SPACES_CHARS = frozenset({'1', '2', '3', '4', '5', '6', '7', '8'})

SEQUENCE_LEN = 77

def tokenize(fen: str) -> jtp.Int32[jtp.Array, 'T']:
    board, side, castling, en_passant, halfmoves, fullmoves = fen.split(' ')
    board = board.replace('/', '')
    board = side + board

    indices = []

    # piece placement
    for char in board:
        if char in SPACES_CHARS:
            indices.extend(int(char) * [CHARS_INDEX['.']])
        else:
            indices.append(CHARS_INDEX[char])

    # castling rights
    if castling == '-':
        indices.extend(4 * [CHARS_INDEX['.']])
    else:
        for char in castling:
            indices.append(CHARS_INDEX[char])
        if len(castling) < 4:
            indices.extend((4 - len(castling)) * [CHARS_INDEX['.']])

    # en passant
    if en_passant == '-':
        indices.extend(2 * [CHARS_INDEX['.']])
    else:
        for char in en_passant:
          indices.append(CHARS_INDEX[char])

    # move counters
    indices.extend([CHARS_INDEX[x] for x in halfmoves.zfill(3)])
    indices.extend([CHARS_INDEX[x] for x in fullmoves.zfill(3)])

    # 77 dim vector
    assert len(indices) == SEQUENCE_LEN

    return np.array(indices, dtype=np.uint8)

# example usage:
#tokenize('8/8/8/4p1K1/2k1P3/8/8/8 b - - 0 1')
