"""
Evaluate SOLIS against Stockfish at various ELO levels.

Plays N games (alternating colors) and reports win/draw/loss statistics.
Games are saved as PGN files for analysis.

Usage:
    python evaluate.py \\
        --checkpoint /path/to/model.pth \\
        --v_white /path/to/mean_white_checkmate.npy \\
        --v_black /path/to/mean_black_checkmate.npy \\
        --stockfish /path/to/stockfish \\
        --depth 4 --width 3 --elo 2200 --games 100
"""

import argparse

import chess.engine
from tqdm import tqdm

from solis.search import SolisEngine, play_game


def main():
    parser = argparse.ArgumentParser(description="Evaluate SOLIS vs Stockfish")
    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights (.pth or .ckpt)")
    parser.add_argument("--v_white", type=str, required=True, help="Path to mean white checkmate embedding (.npy)")
    parser.add_argument("--v_black", type=str, required=True, help="Path to mean black checkmate embedding (.npy)")
    parser.add_argument("--is_checkpoint", action="store_true", help="If set, load from Lightning .ckpt (expects state_dict key)")
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=6)
    # Search
    parser.add_argument("--depth", type=int, default=4, help="Search depth")
    parser.add_argument("--width", type=int, default=3, help="Beam width")
    # Stockfish
    parser.add_argument("--stockfish", type=str, required=True, help="Path to Stockfish binary")
    parser.add_argument("--elo", type=int, nargs="+", default=[2200], help="Stockfish ELO rating(s)")
    parser.add_argument("--time_limit", type=float, default=0.05, help="Stockfish time limit per move")
    # Evaluation
    parser.add_argument("--games", type=int, default=100, help="Number of games to play per ELO level")
    parser.add_argument("--pgn_dir", type=str, default="pgn/", help="Directory to save PGN files")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(args.pgn_dir, exist_ok=True)

    # Load SOLIS engine
    model_config = dict(
        embed_dim=args.embed_dim,
        ff_dim=args.ff_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    solis = SolisEngine(
        model_config=model_config,
        checkpoint_path=args.checkpoint,
        v_white_path=args.v_white,
        v_black_path=args.v_black,
        is_checkpoint=args.is_checkpoint,
    )
    print(f"SOLIS loaded (embed_dim={args.embed_dim}, depth={args.depth}, width={args.width})")

    # Play against Stockfish
    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as engine:
        for elo in args.elo:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            pgn_path = os.path.join(args.pgn_dir, f"solis_d{args.depth}_w{args.width}_elo{elo}.pgn")

            results = []
            for i in tqdm(range(args.games), desc=f"ELO {elo}"):
                solis_color = "white" if i % 2 == 0 else "black"
                result = play_game(
                    engine=engine,
                    solis_engine=solis,
                    solis_color=solis_color,
                    depth=args.depth,
                    beam_width=args.width,
                    stockfish_elo=elo,
                    time_limit=args.time_limit,
                    pgn_save_path=pgn_path,
                )
                results.append(result)
                wins = results.count(1)
                draws = results.count(0.5)
                losses = results.count(0)
                print(f"ELO {elo} | Game {i+1}/{args.games} | W:{wins} D:{draws} L:{losses}")

            wins = results.count(1)
            draws = results.count(0.5)
            losses = results.count(0)
            win_rate = sum(results) / len(results)
            print(f"\n=== ELO {elo} Final ===")
            print(f"Wins: {wins}  Draws: {draws}  Losses: {losses}  Win rate: {win_rate:.3f}")
            print(f"PGN saved to: {pgn_path}\n")


if __name__ == "__main__":
    main()
