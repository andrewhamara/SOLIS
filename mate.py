import chess
import chess.engine

STOCKFISH_PATH = '/data/hamaraa/Stockfish/src/stockfish'

CHECKMATE_WHITE = 444
CHECKMATE_BLACK = -444

def get_k(fen):
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:

        # create chess board from fen
        board = chess.Board(fen)

        if board.is_checkmate():
            return CHECKMATE_BLACK if board.turn == chess.WHITE else CHECKMATE_WHITE

        # get mate in k
        result = engine.analyse(board, chess.engine.Limit(depth=20))
        k = result['score'].relative.mate()

        # handle edge case
        if k is None:
            return None

        if k > 0: # player to move is winning
            k = 2 * k - 1
            if board.turn == chess.BLACK:
                k = -k
        else: # opponent has mate
            k = 2 * abs(k)
            if board.turn == chess.WHITE:
                k = -k

        return k
