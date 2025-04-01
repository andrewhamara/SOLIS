import chess
import chess.engine
import chess.pgn
import torch
import torch.nn.functional as F
import random
import numpy as np
from tokenizer import tokenize
from tqdm import tqdm
from solis import SOLIS

# === CONFIG ===
STOCKFISH_PATH = "/data/hamaraa/Stockfish/src/stockfish"
GAMES = 100
STOCKFISH_ELOS = [2600, 2700, 2800]
TIME_LIMIT = 0.05  # stockfish seconds per board
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# === LOAD SOLIS MODEL + ADVANTAGE AXIS ===
model = SOLIS(embed_dim=1024, ff_dim=1024).to(DEVICE)
latest = False

PGN_SAVE_PATH = '/data/hamaraa/solis_d8_vs_stockfish.pgn'

if latest:
    model = SOLIS().to(DEVICE)
    model.load_state_dict(torch.load("/data/hamaraa/solis_latest.pth", map_location=DEVICE))
    v_white = np.load("/data/hamaraa/mean_white_checkmate_large.npy")
    v_black = np.load("/data/hamaraa/mean_black_checkmate_large.npy")
else:
    model.load_state_dict(torch.load("/data/hamaraa/solis_good.pth", map_location=DEVICE))
    v_white = np.load("/data/hamaraa/mean_white_checkmate.npy")
    v_black = np.load("/data/hamaraa/mean_black_checkmate.npy")

model.eval()

v_white = torch.tensor(v_white, dtype=torch.float32).to(DEVICE)
print(v_white.shape)
v_black = torch.tensor(v_black, dtype=torch.float32).to(DEVICE)
v_advantage = F.normalize(v_white - v_black, dim=0)

# === TOKENIZER ===
def embed_fen(fen):
    tokens = torch.tensor(tokenize(fen), dtype=torch.long, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        return model(tokens).squeeze(0)

def embed_fens(fens):
    tokens = np.array([tokenize(fen) for fen in fens])
    batch = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        return model(batch)  # shape [N, D]

# === SOLIS MOVE SELECTION (project onto advantage axis) ===
def solis_policy(board):
    z_cur = embed_fen(board.fen())
    direction = v_advantage if board.turn == chess.WHITE else -v_advantage
    best_score = -float("inf")
    best_move = None

    for move in board.legal_moves:
        board.push(move)
        z_next = embed_fen(board.fen())
        board.pop()
        delta = z_next - z_cur
        score = torch.dot(F.normalize(delta, dim=0), direction).item()
        if score > best_score:
            best_score = score
            best_move = move

    return best_move

def solis_distance_policy(board):
    z_cur = embed_fen(board.fen())
    v_goal = v_white if board.turn == chess.WHITE else v_black

    best_score = float("inf")
    best_move = None

    for move in board.legal_moves:
        board.push(move)
        z_next = embed_fen(board.fen())
        board.pop()

        score = torch.norm(z_next - v_goal).item()  # lower = better
        if score < best_score:
            best_score = score
            best_move = move

    return best_move

def latent_minimax_policy_batched(board, depth=2):
    player = board.turn == chess.WHITE
    direction = v_advantage  # global white-advantage axis

    if depth == 1 or board.is_game_over():
        z_cur = embed_fens([board.fen()])[0]

        scores = []
        for move in board.legal_moves:
            board.push(move)
            z_next = embed_fens([board.fen()])[0]
            score = torch.dot(z_next, direction).item()
            scores.append(score)
            board.pop()

        best_idx = int(np.argmax(scores)) if player else int(np.argmin(scores))
        return list(board.legal_moves)[best_idx]

    root_moves = list(board.legal_moves)
    reply_fens_per_root = []
    root_boards = []

    for move in root_moves:
        board.push(move)
        root_board = board.copy()
        board.pop()

        replies = list(root_board.legal_moves)
        fens = []
        for reply in replies:
            root_board.push(reply)
            fens.append(root_board.fen())
            root_board.pop()
        reply_fens_per_root.append(fens)
        root_boards.append(root_board)

    # Flatten and embed all reply positions in batch
    all_reply_fens = [fen for group in reply_fens_per_root for fen in group]
    all_embeddings = embed_fens(all_reply_fens)
    all_scores = torch.matmul(all_embeddings, direction)

    scores_per_root = []
    i = 0
    for root_board, fens in zip(root_boards, reply_fens_per_root):
        n = len(fens)
        if n == 0:
            if root_board.is_checkmate():
                score = float("inf") if player else -float("inf")
            elif root_board.is_stalemate():
                score = 0.0
            else:
                raise RuntimeError("Unexpected terminal state")
        else:
            group_scores = all_scores[i : i + n]
            i += n
            score = torch.min(group_scores).item() if player else torch.max(group_scores).item()

        scores_per_root.append(score)

    best_idx = int(np.argmax(scores_per_root)) if player else int(np.argmin(scores_per_root))
    return root_moves[best_idx]

def latent_beam_search_policy(board, beam_width=3, depth=3):
    """
    Beam-style latent search using the advantage axis.
    From the current position, expand top-k moves at each level,
    selecting paths that maximize (or minimize) alignment with advantage direction.
    """
    direction = v_advantage  # white-advantage axis
    player_root = board.turn == chess.WHITE

    def score_position(b):
        if b.is_checkmate():
            return float('inf') if b.turn == chess.BLACK else -float('inf')
        elif b.is_stalemate():
            return 0.0
        else:
            return torch.dot(embed_fens([b.fen()])[0], direction).item()

    def search(b, d, player):
        if d == 0 or b.is_game_over():
            return score_position(b), []

        legal_moves = list(b.legal_moves)
        boards = []
        for move in legal_moves:
            b.push(move)
            boards.append((b.copy(), move))
            b.pop()

        if not boards:
            return score_position(b), []

        # score all boards from current player's perspective
        fens = [b.fen() for b, _ in boards]
        embeddings = embed_fens(fens)
        scores = torch.matmul(embeddings, direction)

        # select top-k moves for the current player
        k = min(beam_width, len(boards))
        sorted_indices = torch.argsort(scores, descending=player)[:k]

        best_score = None
        best_path = []

        for idx in sorted_indices:
            new_board, move = boards[idx]
            score, path = search(new_board, d - 1, not player)
            if best_score is None or (player and score > best_score) or (not player and score < best_score):
                best_score = score
                best_path = [move] + path

        return best_score, best_path

    _, best_move_path = search(board.copy(), depth, player_root)
    return best_move_path[0] if best_move_path else None


# === PLAY A SINGLE GAME ===
def play_game(engine, solis_color="white", STOCKFISH_ELO=2000):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers['Event'] = "SOLIS vs Stockfish"

    white_name = "SOLIS" if solis_color == "white" else f"Stockfish_{STOCKFISH_ELO}"
    black_name = f"Stockfish_{STOCKFISH_ELO}" if solis_color == "white" else "SOLIS"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name

    node = game

    while not board.is_game_over():
        if (board.turn == chess.WHITE and solis_color == "white") or (board.turn == chess.BLACK and solis_color == "black"):
            #move = solis_policy(board)
            #move = solis_distance_policy(board)
            #move = latent_minimax_policy_batched(board, depth=2)
            move = latent_beam_search_policy(board, beam_width=3, depth=8)
        else:
            result = engine.play(board, chess.engine.Limit(time=TIME_LIMIT))
            move = result.move
        board.push(move)
        node = node.add_variation(move)
      

    result = board.result()
    game.headers["Result"] = board.result()

    with open(PGN_SAVE_PATH, "a") as pgn_file:
        print(game, file=pgn_file, flush=True)

    if result == "1-0":
        return 1 if solis_color == "white" else 0
    elif result == "0-1":
        return 1 if solis_color == "black" else 0
    else:
        return 0.5  # draw

# === MAIN BENCHMARK LOOP ===
with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
    for STOCKFISH_ELO in STOCKFISH_ELOS:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": STOCKFISH_ELO})

        results = []
        for i in tqdm(range(GAMES)):
            solis_color = "white" if i % 2 == 0 else "black"
            result = play_game(engine, solis_color, STOCKFISH_ELO)
            results.append(result)
            wins, draws, losses = results.count(1), results.count(0.5), results.count(0)
            print(f'wins: {wins}, draws: {draws}, losses: {losses}')

# === EVALUATE ===
solis_score = sum(results)
win_rate = solis_score / GAMES

print(f"\nSOLIS vs Stockfish {STOCKFISH_ELO}")
print(f"Wins: {results.count(1)}  Draws: {results.count(0.5)}  Losses: {results.count(0)}")
print(f"Win rate: {win_rate:.3f}")

# Approximate Elo diff from win rate using logistic model
from math import log
def elo_diff_from_winrate(p):
    if p in {0, 1}:
        return float('inf') if p == 1 else -float('inf')
    return -400 * log(1/p - 1, 10)

elo_diff = elo_diff_from_winrate(win_rate)
print(f"Estimated Elo: {STOCKFISH_ELO + elo_diff:.1f}")
