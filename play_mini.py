from functools import lru_cache
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
import time
import matplotlib.pyplot as plt
from chess.polyglot import zobrist_hash
from collections import OrderedDict

import sys, os

DEPTH = int(sys.argv[1])     # e.g. 2
WIDTH = int(sys.argv[2])     # e.g. 3
GPU   = int(sys.argv[3])     # e.g. 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

torch.set_grad_enabled(False)

class LRU:
    def __init__(self, capacity=200_000):
        self.cap = capacity
        self.d = OrderedDict()

    def get(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return self.d[k]
        return None

    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)

EMB_CACHE = LRU(capacity=400_000)   # zobrist -> torch.Tensor embedding (on CPU)
SCORE_CACHE = LRU(capacity=400_000) # zobrist -> float alignment score
TT = {}

# === CONFIG ===
STOCKFISH_PATH = "/data/hamaraa/Stockfish/src/stockfish"
GAMES = 100
TIME_LIMIT = 0.05  # stockfish seconds per board
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# === LOAD SOLIS MODEL + ADVANTAGE AXIS ===
model = SOLIS(embed_dim=128, ff_dim=256, num_heads=8, num_layers=6).to(DEVICE)
#model = SOLIS().to(DEVICE)
mini = True

def get_embeddings(x, batch_size=256):
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = torch.tensor(x[i:i+batch_size], dtype=torch.long, device=DEVICE)
            emb = model(batch).cpu().numpy()
            all_embeddings.append(emb)
    return np.concatenate(all_embeddings, axis=0)

if mini:
    model.load_state_dict(torch.load("/data/hamaraa/solis_mini_stepstep=400000.ckpt", map_location=DEVICE)["state_dict"])
    v_white = np.load("/data/hamaraa/mean_white_checkmate_mini.npy")
    v_black = np.load("/data/hamaraa/mean_black_checkmate_mini.npy")
else:
    model.load_state_dict(torch.load("/data/hamaraa/solis_good.pth", map_location=DEVICE))
    v_white = np.load("/data/hamaraa/mean_white_checkmate.npy")
    v_black = np.load("/data/hamaraa/mean_black_checkmate.npy")

model.eval()

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

@torch.no_grad()
def score_positions(boards):
    """return list[float] of (z - μ_black) · (μ_white - μ_black), with Zobrist+LRU cache"""
    keys = [zobrist_hash(b) for b in boards]

    hit_vals = {}
    miss_idx = []
    for i, k in enumerate(keys):
        v = SCORE_CACHE.get(k)
        if v is not None:
            hit_vals[i] = v
        else:
            miss_idx.append(i)

    if not miss_idx:
        return [hit_vals[i] for i in range(len(boards))]

    fens = [boards[i].fen() for i in miss_idx]
    toks = np.stack([token_for_fen(f) for f in fens], axis=0)
    batch = torch.tensor(toks, dtype=torch.long, device=DEVICE)

    model.eval()
    with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
        emb = model(batch)   # [M, D]
        offset = emb - v_black
        scores = torch.matmul(offset, v_white - v_black)  # projection onto advantage axis

    out_scores = scores.detach().float().cpu().numpy()
    for j, idx in enumerate(miss_idx):
        SCORE_CACHE.put(keys[idx], float(out_scores[j]))

    out = []
    mj = 0
    for i in range(len(boards)):
        if i in hit_vals:
            out.append(hit_vals[i])
        else:
            out.append(float(out_scores[mj])); mj += 1
    return out

def latent_beam_search_policy(board, beam_width=3, depth=3):
    TT.clear()  # per-move table; keep SCORE_CACHE global

    direction = v_advantage
    player_root = board.turn == chess.WHITE

    def score_position(b):
        if b.is_checkmate():
            return float('inf') if b.turn == chess.BLACK else -float('inf')
        if b.is_stalemate():
            return 0.0
        return score_positions([b])[0]

    def search(b, d, player_max):
        # TT lookup
        key = (zobrist_hash(b), d, player_max)
        if key in TT:
            return TT[key]

        if d == 0 or b.is_game_over():
            val = score_position(b)
            TT[key] = val
            return val

        moves = list(b.legal_moves)
        if not moves:
            val = score_position(b)
            TT[key] = val
            return val

        # generate children once
        children = []
        for mv in moves:
            b.push(mv)
            children.append((mv, b.copy(stack=False)))
            b.pop()

        # batch-score children
        child_scores = score_positions([cb for _, cb in children])

        # pick top-k for this side
        k = min(beam_width, len(children))
        order = sorted(range(len(children)),
                       key=lambda i: child_scores[i],
                       reverse=player_max)[:k]

        best = None
        for i in order:
            _, cb = children[i]
            v = search(cb, d - 1, not player_max)
            if best is None or (player_max and v > best) or ((not player_max) and v < best):
                best = v

        TT[key] = best
        return best

    # root: choose move
    legal = list(board.legal_moves)
    if not legal:
        return None
    children = []
    for mv in legal:
        board.push(mv); children.append((mv, board.copy(stack=False))); board.pop()
    scores = score_positions([cb for _, cb in children])

    k = min(beam_width, len(children))
    order = sorted(range(len(children)),
                   key=lambda i: scores[i],
                   reverse=player_root)[:k]

    best_mv, best_val = None, None
    for i in order:
        mv, cb = children[i]
        v = search(cb, depth - 1, not player_root)
        if best_val is None or (player_root and v > best_val) or ((not player_root) and v < best_val):
            best_mv, best_val = mv, v
    return best_mv

# === PLAY A SINGLE GAME ===
def play_game(engine, solis_color, PGN_SAVE_PATH, DEPTH):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers['Event'] = "SOLIS vs Stockfish"

    white_name = "SOLIS" if solis_color == "white" else f"Stockfish_d{DEPTH}"
    black_name = f"Stockfish_d{DEPTH}" if solis_color == "white" else "SOLIS"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name

    node = game

    while not board.is_game_over():
        if (board.turn == chess.WHITE and solis_color == "white") or (board.turn == chess.BLACK and solis_color == "black"):
            move = latent_beam_search_policy(board, beam_width=WIDTH, depth=DEPTH)
        else:
            result = engine.play(board, chess.engine.Limit(depth=DEPTH, time=TIME_LIMIT))
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

PGN_SAVE_PATH = f'/data/hamaraa/solis_mini_relative_d{DEPTH}_w{WIDTH}_vs_stockfish_d{DEPTH}.pgn'
# === MAIN BENCHMARK LOOP ===
with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
    results = []
    for i in tqdm(range(GAMES)):
        solis_color = "white" if i % 2 == 0 else "black"
        result = play_game(engine, solis_color, PGN_SAVE_PATH=PGN_SAVE_PATH, DEPTH=DEPTH)
        results.append(result)
        wins, draws, losses = results.count(1), results.count(0.5), results.count(0)
        print(f'wins: {wins}, draws: {draws}, losses: {losses}')