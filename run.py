import sys, os
import torch
import chess, chess.engine, chess.pgn
import torch.nn.functional as F
import numpy as np
from solis import SOLIS
from tokenizer import tokenize
from collections import OrderedDict
from chess.polyglot import zobrist_hash

# === CLI args: depth, width, gpu, elo ===
DEPTH = int(sys.argv[1])
WIDTH = int(sys.argv[2])
GPU   = int(sys.argv[3])
ELO   = int(sys.argv[4])

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}, GPU {GPU}, Elo {ELO}")

torch.set_grad_enabled(False)

# === caches ===
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

EMB_CACHE   = LRU(capacity=400_000)
SCORE_CACHE = LRU(capacity=400_000)
TT = {}

# === paths / configs ===
STOCKFISH_PATH = "/data/hamaraa/Stockfish/src/stockfish"
TIME_LIMIT     = 0.05
PGN_SAVE_PATH  = f'/data/hamaraa/solis_base_relative_d{DEPTH}_w{WIDTH}_elo{ELO}.pgn'

# === load SOLIS model + vectors ===
model = SOLIS(embed_dim=1024, ff_dim=1024).to(DEVICE)
model.load_state_dict(torch.load("/data/hamaraa/solis_good.pth", map_location=DEVICE))
model.eval()

v_white = np.load("/data/hamaraa/mean_white_checkmate.npy")
v_black = np.load("/data/hamaraa/mean_black_checkmate.npy")
v_white = torch.tensor(v_white, dtype=torch.float32).to(DEVICE)
v_black = torch.tensor(v_black, dtype=torch.float32).to(DEVICE)
v_advantage = F.normalize(v_white - v_black, dim=0)

# === helper funcs (unchanged core) ===
def token_for_fen(fen):
    return np.asarray(tokenize(fen), dtype=np.int64)

def score_positions(boards):
    keys = [zobrist_hash(b) for b in boards]
    hit_vals, miss_idx = {}, []
    for i, k in enumerate(keys):
        v = SCORE_CACHE.get(k)
        if v is not None: hit_vals[i] = v
        else: miss_idx.append(i)

    if miss_idx:
        fens = [boards[i].fen() for i in miss_idx]
        toks = np.stack([token_for_fen(f) for f in fens], axis=0)
        batch = torch.tensor(toks, dtype=torch.long, device=DEVICE)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda")):
            emb = model(batch)
            offset = emb - v_black
            scores = torch.matmul(offset, v_white - v_black)
        out_scores = scores.detach().cpu().numpy()
        for j, idx in enumerate(miss_idx):
            SCORE_CACHE.put(keys[idx], float(out_scores[j]))

    out, mj = [], 0
    for i in range(len(boards)):
        if i in hit_vals: out.append(hit_vals[i])
        else: out.append(float(out_scores[mj])); mj += 1
    return out

def latent_beam_search_policy(board, beam_width=3, depth=3):
    TT.clear()
    player_root = board.turn == chess.WHITE
    def search(b, d, player_max):
        key = (zobrist_hash(b), d, player_max)
        if key in TT: return TT[key]
        if d == 0 or b.is_game_over():
            val = score_positions([b])[0]; TT[key] = val; return val
        moves = list(b.legal_moves)
        children = [(mv, b.copy(stack=False)) for mv in moves for _ in [b.push(mv)] or [b.pop()]]
        child_scores = score_positions([cb for _, cb in children])
        order = sorted(range(len(children)), key=lambda i: child_scores[i], reverse=player_max)[:beam_width]
        best = None
        for i in order:
            _, cb = children[i]
            v = search(cb, d-1, not player_max)
            if best is None or (player_max and v > best) or ((not player_max) and v < best):
                best = v
        TT[key] = best
        return best
    legal = list(board.legal_moves)
    children = [(mv, board.copy(stack=False)) for mv in legal for _ in [board.push(mv)] or [board.pop()]]
    scores = score_positions([cb for _, cb in children])
    order = sorted(range(len(children)), key=lambda i: scores[i], reverse=player_root)[:beam_width]
    best_mv, best_val = None, None
    for i in order:
        mv, cb = children[i]
        v = search(cb, depth-1, not player_root)
        if best_val is None or (player_root and v > best_val) or ((not player_root) and v < best_val):
            best_mv, best_val = mv, v
    return best_mv

def play_game(engine, solis_color="white", STOCKFISH_ELO=2000, num_games=50):
    results = []
    for i in range(num_games):
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = "SOLIS vs Stockfish"
        game.headers["White"] = "SOLIS" if solis_color=="white" else f"Stockfish_{STOCKFISH_ELO}"
        game.headers["Black"] = f"Stockfish_{STOCKFISH_ELO}" if solis_color=="white" else "SOLIS"
        node = game
        while not board.is_game_over():
            if (board.turn==chess.WHITE and solis_color=="white") or (board.turn==chess.BLACK and solis_color=="black"):
                move = latent_beam_search_policy(board, beam_width=WIDTH, depth=DEPTH)
            else:
                result = engine.play(board, chess.engine.Limit(time=TIME_LIMIT))
                move = result.move
            board.push(move); node = node.add_variation(move)
        game.headers["Result"] = board.result()
        with open(PGN_SAVE_PATH, "a") as pgn_file: print(game, file=pgn_file, flush=True)
        r = board.result()
        if r=="1-0": results.append(1 if solis_color=="white" else 0)
        elif r=="0-1": results.append(1 if solis_color=="black" else 0)
        else: results.append(0.5)
    return results

# === main ===
with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": ELO})
    results = play_game(engine, solis_color="white", STOCKFISH_ELO=ELO, num_games=50)
    results += play_game(engine, solis_color="black", STOCKFISH_ELO=ELO, num_games=50)
    print(f"Elo {ELO} done. Wins={results.count(1)}, Draws={results.count(0.5)}, Losses={results.count(0)}")
