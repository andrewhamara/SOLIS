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
model = SOLIS(embed_dim=512, ff_dim=512, num_layers=6, num_heads=16).to(DEVICE)
#model = SOLIS().to(DEVICE)
small = True

DEPTH = 3
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
if DEPTH == 2 and WIDTH < 3:
    STOCKFISH_ELOS = [1350, 1400, 1450, 1500, 1550, 1600, 1650]

if DEPTH == 3:
    STOCKFISH_ELOS = [2000, 2050, 2100, 2150, 2200, 2250, 2300]
if DEPTH == 3 and WIDTH < 3:
    STOCKFISH_ELOS = [1500, 1550, 1600, 1650, 1700, 1750, 1800]

if DEPTH == 4:
    STOCKFISH_ELOS = [2400, 2350, 2300, 2250, 2200, 2150]
if DEPTH == 4 and WIDTH < 3:
    STOCKFISH_ELOS = [1600, 1650, 1700, 1750, 1800, 1850]

if DEPTH == 5:
    STOCKFISH_ELOS = [2500, 2400, 2350, 2300, 2250, 2200]
if DEPTH == 5 and WIDTH < 3:
    STOCKFISH_ELOS = [1700, 1750, 1800, 1850, 1900, 1950]
    
if DEPTH == 6:
    STOCKFISH_ELOS = [2600, 2550, 2500, 2400, 2300, 2200]

print(STOCKFISH_ELOS)

STEPS = 50000
PGN_SAVE_PATH = f'/data/hamaraa/solis_small_{STEPS}_steps_d{DEPTH}_w{WIDTH}_allratings_vs_stockfish.pgn'

if small:
    #model.load_state_dict(torch.load("/data/hamaraa/solis_latest_small.pth", map_location=DEVICE))
    model.load_state_dict(torch.load(f"/data/hamaraa/solis_small_stepstep={STEPS}.ckpt", map_location=DEVICE)["state_dict"])
    v_white = np.load("/data/hamaraa/mean_white_checkmate_small.npy")
    v_black = np.load("/data/hamaraa/mean_black_checkmate_small.npy")
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
    """return list[float] of ⟨embedding, v_advantage⟩, with Zobrist+LRU cache"""
    # collect zobrist keys
    keys = [zobrist_hash(b) for b in boards]

    # split hits/misses
    hit_vals = {}
    miss_idx = []
    for i, k in enumerate(keys):
        v = SCORE_CACHE.get(k)
        if v is not None:
            hit_vals[i] = v
        else:
            miss_idx.append(i)

    # fast path: all cached
    if not miss_idx:
        return [hit_vals[i] for i in range(len(boards))]

    # batch tokenize misses
    fens = [boards[i].fen() for i in miss_idx]
    toks = np.stack([token_for_fen(f) for f in fens], axis=0)  # uses your @lru_cache
    batch = torch.tensor(toks, dtype=torch.long, device=DEVICE)

    # forward + score (AMP on cuda)
    model.eval()
    with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
        emb = model(batch)                           # [M, D], already L2-normalized
        scores = torch.matmul(emb, v_advantage)      # [M]

    # write cache (store floats; optional: also cache emb.cpu())
    out_scores = scores.detach().float().cpu().numpy()
    for j, idx in enumerate(miss_idx):
        k = keys[idx]
        val = float(out_scores[j])
        SCORE_CACHE.put(k, val)

    # merge back in order
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
