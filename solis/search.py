"""
SOLIS search engine for inference and evaluation.

Provides the core latent beam search algorithm and game-playing infrastructure.
The engine scores chess positions by projecting their embeddings onto a learned
"advantage direction" and uses beam search with transposition tables and LRU
caching for efficient move selection.
"""

from collections import OrderedDict
from functools import lru_cache

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F
from chess.polyglot import zobrist_hash

from solis.model import SOLIS
from solis.tokenizer import tokenize


class LRU:
    """Least-recently-used cache with fixed capacity."""

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

    def clear(self):
        self.d.clear()


class SolisEngine:
    """
    SOLIS chess engine that selects moves via latent beam search.

    Loads a trained SOLIS model and precomputed advantage vectors, then
    plays chess by scoring positions in the learned embedding space.

    Args:
        model_config: Dict of model hyperparameters (embed_dim, ff_dim, etc.).
        checkpoint_path: Path to model weights (.pth or .ckpt).
        v_white_path: Path to mean white checkmate embedding (.npy).
        v_black_path: Path to mean black checkmate embedding (.npy).
        device: Torch device to use.
        is_checkpoint: If True, loads from Lightning checkpoint (state_dict key).
        cache_capacity: Size of the LRU caches.
    """

    def __init__(
        self,
        model_config,
        checkpoint_path,
        v_white_path,
        v_black_path,
        device=None,
        is_checkpoint=False,
        cache_capacity=400_000,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = SOLIS(**model_config).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        if is_checkpoint:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

        # Load advantage vectors
        v_white = np.load(v_white_path)
        v_black = np.load(v_black_path)
        self.v_white = torch.tensor(v_white, dtype=torch.float32).to(self.device)
        self.v_black = torch.tensor(v_black, dtype=torch.float32).to(self.device)
        self.v_advantage = F.normalize(self.v_white - self.v_black, dim=0)

        # Caches
        self.score_cache = LRU(capacity=cache_capacity)
        self.tt = {}

        torch.set_grad_enabled(False)

    @lru_cache(maxsize=200_000)
    def _token_for_fen(self, fen):
        return np.asarray(tokenize(fen), dtype=np.int64)

    @torch.no_grad()
    def score_positions(self, boards):
        """Score positions by projecting embeddings onto the advantage axis.

        Returns list[float] of (embedding - v_black) . (v_white - v_black),
        with Zobrist+LRU caching.
        """
        keys = [zobrist_hash(b) for b in boards]

        hit_vals = {}
        miss_idx = []
        for i, k in enumerate(keys):
            v = self.score_cache.get(k)
            if v is not None:
                hit_vals[i] = v
            else:
                miss_idx.append(i)

        if not miss_idx:
            return [hit_vals[i] for i in range(len(boards))]

        fens = [boards[i].fen() for i in miss_idx]
        toks = np.stack([self._token_for_fen(f) for f in fens], axis=0)
        batch = torch.tensor(toks, dtype=torch.long, device=self.device)

        with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            emb = self.model(batch)
            offset = emb - self.v_black
            scores = torch.matmul(offset, self.v_white - self.v_black)

        out_scores = scores.detach().float().cpu().numpy()
        for j, idx in enumerate(miss_idx):
            self.score_cache.put(keys[idx], float(out_scores[j]))

        out = []
        mj = 0
        for i in range(len(boards)):
            if i in hit_vals:
                out.append(hit_vals[i])
            else:
                out.append(float(out_scores[mj]))
                mj += 1
        return out

    def select_move(self, board, beam_width=3, depth=3):
        """Select the best move using latent beam search.

        Args:
            board: chess.Board representing the current position.
            beam_width: Number of top moves to expand at each depth level.
            depth: Search depth (number of half-moves to look ahead).

        Returns:
            The best chess.Move, or None if no legal moves exist.
        """
        self.tt.clear()
        player_root = board.turn == chess.WHITE

        def score_terminal(b):
            if b.is_checkmate():
                return float('inf') if b.turn == chess.BLACK else -float('inf')
            if b.is_stalemate():
                return 0.0
            return self.score_positions([b])[0]

        def search(b, d, player_max):
            key = (zobrist_hash(b), d, player_max)
            if key in self.tt:
                return self.tt[key]

            if d == 0 or b.is_game_over():
                val = score_terminal(b)
                self.tt[key] = val
                return val

            moves = list(b.legal_moves)
            if not moves:
                val = score_terminal(b)
                self.tt[key] = val
                return val

            children = []
            for mv in moves:
                b.push(mv)
                children.append((mv, b.copy(stack=False)))
                b.pop()

            child_scores = self.score_positions([cb for _, cb in children])

            k = min(beam_width, len(children))
            order = sorted(range(len(children)),
                           key=lambda i: child_scores[i],
                           reverse=player_max)[:k]

            best = None
            for i in order:
                _, cb = children[i]
                v = search(cb, d - 1, not player_max)
                if best is None or (player_max and v > best) or (not player_max and v < best):
                    best = v

            self.tt[key] = best
            return best

        # Root: choose move
        legal = list(board.legal_moves)
        if not legal:
            return None
        children = []
        for mv in legal:
            board.push(mv)
            children.append((mv, board.copy(stack=False)))
            board.pop()
        scores = self.score_positions([cb for _, cb in children])

        k = min(beam_width, len(children))
        order = sorted(range(len(children)),
                       key=lambda i: scores[i],
                       reverse=player_root)[:k]

        best_mv, best_val = None, None
        for i in order:
            mv, cb = children[i]
            v = search(cb, depth - 1, not player_root)
            if best_val is None or (player_root and v > best_val) or (not player_root and v < best_val):
                best_mv, best_val = mv, v
        return best_mv


def play_game(engine, solis_engine, solis_color, depth, beam_width,
              stockfish_elo=None, time_limit=0.05, pgn_save_path=None):
    """Play a single game between SOLIS and Stockfish.

    Args:
        engine: chess.engine.SimpleEngine instance (Stockfish).
        solis_engine: SolisEngine instance.
        solis_color: "white" or "black".
        depth: Search depth for SOLIS.
        beam_width: Beam width for SOLIS.
        stockfish_elo: ELO rating for Stockfish (used in PGN headers).
        time_limit: Time limit per Stockfish move.
        pgn_save_path: If provided, append PGN to this file.

    Returns:
        1 for SOLIS win, 0 for loss, 0.5 for draw.
    """
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers['Event'] = "SOLIS vs Stockfish"

    opponent = f"Stockfish_{stockfish_elo}" if stockfish_elo else "Stockfish"
    game.headers["White"] = "SOLIS" if solis_color == "white" else opponent
    game.headers["Black"] = opponent if solis_color == "white" else "SOLIS"

    node = game

    while not board.is_game_over():
        is_solis_turn = (
            (board.turn == chess.WHITE and solis_color == "white") or
            (board.turn == chess.BLACK and solis_color == "black")
        )

        if is_solis_turn:
            move = solis_engine.select_move(board, beam_width=beam_width, depth=depth)
        else:
            result = engine.play(board, chess.engine.Limit(time=time_limit))
            move = result.move

        board.push(move)
        node = node.add_variation(move)

    result = board.result()
    game.headers["Result"] = result

    if pgn_save_path:
        with open(pgn_save_path, "a") as pgn_file:
            print(game, file=pgn_file, flush=True)

    if result == "1-0":
        return 1 if solis_color == "white" else 0
    elif result == "0-1":
        return 1 if solis_color == "black" else 0
    else:
        return 0.5
