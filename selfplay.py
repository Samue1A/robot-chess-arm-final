"""Run an unguarded timing-hybrid-aggressive mirror match and save its PGN."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn

from engine import TimingHybridEngine


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
GAMES_DIR = ROOT / "games"


def load_runtime_config() -> dict:
    defaults = {
        "search_seconds": 30.0,
        "minimum_depth": 4,
        "maximum_depth": 8,
        "use_opening_book": True,
    }
    if CONFIG_PATH.exists():
        defaults.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return defaults


def save_game(game: chess.pgn.Game, output_path: Path):
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    text = str(game) + "\n"
    output_path.write_text(text, encoding="utf-8")
    (GAMES_DIR / "latest_selfplay.pgn").write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play timing-hybrid-aggressive against an independent copy of itself."
    )
    parser.add_argument("--seconds", type=float, help="Nominal time per non-book move")
    parser.add_argument("--min-depth", type=int, help="Forced minimum search depth")
    parser.add_argument("--max-depth", type=int, help="Maximum iterative-deepening depth")
    parser.add_argument("--no-book", action="store_true", help="Disable the bundled opening book")
    parser.add_argument(
        "--max-plies",
        type=int,
        default=220,
        help="Emergency game-length cap matching N6 (default: 220 plies)",
    )
    parser.add_argument("--output", type=Path, help="PGN output path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_runtime_config()
    seconds = float(args.seconds if args.seconds is not None else config["search_seconds"])
    min_depth = int(args.min_depth if args.min_depth is not None else config["minimum_depth"])
    max_depth = int(args.max_depth if args.max_depth is not None else config["maximum_depth"])
    use_book = bool(config["use_opening_book"]) and not args.no_book
    max_plies = max(1, int(args.max_plies))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or (GAMES_DIR / f"selfplay_unguarded_{stamp}.pgn")
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading the exact N6 timing-hybrid-aggressive self-play engine...", flush=True)
    print("Stockfish tactical guard: DISABLED", flush=True)
    # N6's original self-play used one global player for both colors, retaining
    # its normal caches across the whole game. Use the same arrangement here.
    engine = TimingHybridEngine(
        search_seconds=seconds,
        min_depth=min_depth,
        max_depth=max_depth,
        use_opening_book=use_book,
        tactical_guard_enabled=False,
    )
    if engine.tactical_guard.enabled or engine.tactical_guard.available:
        raise RuntimeError("Stockfish tactical guard was not fully disabled")

    board = chess.Board()
    history: list[str] = []
    game = chess.pgn.Game()
    game.headers.update(
        {
            "Event": "N10 unguarded mirror self-play",
            "Site": "Local",
            "Date": datetime.now().strftime("%Y.%m.%d"),
            "Round": "1",
            "White": "timing-hybrid-aggressive (no Stockfish)",
            "Black": "timing-hybrid-aggressive (no Stockfish)",
            "Result": "*",
            "TimeControl": f"{seconds:g} seconds per move",
            "StockfishGuard": "disabled",
            "SearchSeconds": f"{seconds:g}",
            "MinimumDepth": str(min_depth),
            "MaximumDepth": str(max_depth),
            "OpeningBook": "enabled" if use_book else "disabled",
        }
    )
    node = game
    result_text = "*"
    termination = "interrupted"

    try:
        while board.ply() < max_plies:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                result_text = outcome.result()
                termination = outcome.termination.name.lower().replace("_", " ")
                break

            mover = "White" if board.turn == chess.WHITE else "Black"
            move_number = board.fullmove_number
            result = engine.choose_move(board, history)
            if result.guard is not None and result.guard.get("enabled"):
                raise RuntimeError("Self-play guard unexpectedly enabled")

            move = result.move
            san = board.san(move)
            eval_for_mover = result.score_cp if board.turn == chess.WHITE else -result.score_cp
            node = node.add_variation(move)
            node.comment = (
                f"source={result.source}; depth={result.depth}; "
                f"eval_white={result.score_cp:+.1f}cp; "
                f"eval_for_mover={eval_for_mover:+.1f}cp; "
                f"time={result.elapsed:.2f}s; nodes={result.nodes}"
            )
            board.push(move)
            history.append(san)
            game.headers["Result"] = "*"
            save_game(game, output_path)
            print(
                f"{move_number}{'.' if mover == 'White' else '...'} {san:<8} "
                f"source={result.source:<20} depth={result.depth} "
                f"eval={eval_for_mover:+.0f}cp time={result.elapsed:.1f}s nodes={result.nodes}",
                flush=True,
            )
        else:
            result_text = "*"
            termination = f"stopped at {max_plies}-ply safety cap"
    except KeyboardInterrupt:
        print("\nSelf-play interrupted; saving the completed position.", flush=True)
    finally:
        game.headers["Result"] = result_text
        game.headers["Termination"] = termination
        save_game(game, output_path)
        engine.close()

    print(f"Result: {result_text} ({termination})", flush=True)
    print(f"PGN saved to: {output_path}", flush=True)
    print(f"Latest self-play copy: {GAMES_DIR / 'latest_selfplay.pgn'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
