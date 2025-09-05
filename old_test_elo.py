from functools import lru_cache
import chess
import chess.engine
import chess.pgn
import umap
import torch
import torch.nn.functional as F
import random
import numpy as np
from tokenizer import tokenize
from tqdm import tqdm
from solis import SOLIS
import time
import matplotlib.pyplot as plt

# === CONFIG ===
STOCKFISH_PATH = "/data/hamaraa/Stockfish/src/stockfish"
GAMES = 100
STOCKFISH_ELOS = [2800]
TIME_LIMIT = 0.05  # stockfish seconds per board
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# === LOAD SOLIS MODEL + ADVANTAGE AXIS ===
model = SOLIS(embed_dim=1024, ff_dim=1024).to(DEVICE)
#model = SOLIS().to(DEVICE)
small = False

DEPTH = 2
WIDTH = 3

def get_embeddings(x, batch_size=256):
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = torch.tensor(x[i:i+batch_size], dtype=torch.long, device=DEVICE)
            emb = model(batch).cpu().numpy()
            all_embeddings.append(emb)
    return np.concatenate(all_embeddings, axis=0)


if DEPTH == 2:
    STOCKFISH_ELOS = [2100, 2050, 2000, 1950, 1900, 1850, 1800]
#if DEPTH == 3:
#    STOCKFISH_ELOS = [2000, 2050, 2100, 2150, 2200, 2250, 2300]
if DEPTH == 5:
    STOCKFISH_ELOS = [2350, 2300, 2250, 2200]
if DEPTH == 6 and not small:
    STOCKFISH_ELOS = [2300, 2400, 2450]

print(STOCKFISH_ELOS)

PGN_SAVE_PATH = f'/data/hamaraa/solis_base_d{DEPTH}_w{WIDTH}_allratings_vs_stockfish.pgn'

if small:
    model = SOLIS().to(DEVICE)
    model.load_state_dict(torch.load("/data/hamaraa/solis_latest_small.pth", map_location=DEVICE))
    v_white = np.load("/data/hamaraa/mean_white_checkmate_small.npy")
    v_black = np.load("/data/hamaraa/mean_black_checkmate_small.npy")
else:
    model.load_state_dict(torch.load("/data/hamaraa/solis_good.pth", map_location=DEVICE))
    v_white = np.load("/data/hamaraa/mean_white_checkmate.npy")
    v_black = np.load("/data/hamaraa/mean_black_checkmate.npy")


model.eval()

def compute_projection_trajectory(fens, v_advantage):
    embeddings = embed_fens(fens)
    projections = []
    for i in range(len(embeddings) - 1):
        delta = embeddings[i + 1] - embeddings[i]
        proj = np.dot(delta.cpu(), v_advantage.cpu().numpy())
        projections.append(proj)
    return projections

# convert to tensor for rest of script
v_white = torch.tensor(v_white, dtype=torch.float32).to(DEVICE)
v_black = torch.tensor(v_black, dtype=torch.float32).to(DEVICE)
v_advantage = F.normalize(v_white - v_black, dim=0)

# === TOKENIZER ===
def embed_fen(fen):
    tokens = torch.tensor(tokenize(fen), dtype=torch.long, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        return model(tokens).squeeze(0)

@lru_cache(maxsize=200_000)
def token_for_fen(fen):
    return np.asarray(tokenize(fen), dtype=np.int64)

def embed_fens(fens):
    tokens = np.array([token_for_fen(fen) for fen in fens])
    batch = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        return model(batch)  # shape [N, D]


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
            move = latent_beam_search_policy(board, beam_width=WIDTH, depth=DEPTH)
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
