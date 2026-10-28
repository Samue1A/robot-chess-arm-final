"""Replay a saved chess game using the physical arm for both colors."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn

from physical_arm import ArmError, PhysicalChessArm


FOLDER = Path(__file__).resolve().parent
DEFAULT_PGN = FOLDER / "movement_test_game.pgn"
CHECKPOINT_PATH = FOLDER / "movement_replay_checkpoint.json"


def load_game(path: Path) -> chess.pgn.Game:
    if not path.exists():
        raise FileNotFoundError(f"PGN does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        raise ValueError(f"No chess game was found in {path}")
    if game.errors:
        raise ValueError(f"PGN parser reported errors: {game.errors}")
    return game


def validated_moves(game: chess.pgn.Game) -> tuple[list[chess.Move], dict[str, int]]:
    board = game.board()
    moves: list[chess.Move] = []
    stats = {"captures": 0, "castles": 0, "promotions": 0}
    for ply, move in enumerate(game.mainline_moves(), start=1):
        if move not in board.legal_moves:
            raise ValueError(
                f"Illegal PGN move at ply {ply}: {move.uci()} from {board.fen()}"
            )
        stats["captures"] += int(board.is_capture(move))
        stats["castles"] += int(board.is_castling(move))
        stats["promotions"] += int(bool(move.promotion))
        moves.append(move)
        board.push(move)
    if not moves:
        raise ValueError("The PGN contains no moves")
    return moves, stats


def save_checkpoint(
    *,
    pgn_path: Path,
    next_ply: int,
    board: chess.Board,
    last_san: str | None,
):
    payload = {
        "pgn": str(pgn_path.resolve()),
        "next_ply": next_ply,
        "last_completed_san": last_san,
        "fen_after_completed_moves": board.fen(),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    CHECKPOINT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute both sides of a PGN on the physical chess board"
    )
    parser.add_argument(
        "pgn",
        nargs="?",
        type=Path,
        default=DEFAULT_PGN,
        help=f"PGN to replay (default: {DEFAULT_PGN.name})",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Move the real RoArm; omission prints a complete dry run",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the single initial board/tray confirmation",
    )
    parser.add_argument(
        "--manual-captures",
        action="store_true",
        help="Ask for captured pieces to be removed by hand instead of using the tray",
    )
    parser.add_argument(
        "--start-ply",
        type=int,
        default=0,
        help="Skip this many completed half-moves; the real board must match the shown FEN",
    )
    parser.add_argument(
        "--end-ply",
        type=int,
        help="Stop before this zero-based ply index (useful for a shorter test)",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Optional pause after each completed ply",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pgn_path = args.pgn if args.pgn.is_absolute() else (Path.cwd() / args.pgn)
    game = load_game(pgn_path)
    moves, stats = validated_moves(game)
    start_ply = int(args.start_ply)
    end_ply = len(moves) if args.end_ply is None else int(args.end_ply)
    if not 0 <= start_ply <= len(moves):
        raise ValueError(f"--start-ply must be between 0 and {len(moves)}")
    if not start_ply <= end_ply <= len(moves):
        raise ValueError(f"--end-ply must be between {start_ply} and {len(moves)}")
    if args.delay_seconds < 0:
        raise ValueError("--delay-seconds must not be negative")

    board = game.board()
    for move in moves[:start_ply]:
        board.push(move)

    print("\nPhysical PGN movement replay")
    print("PGN:", pgn_path.resolve())
    print(
        f"Game: {game.headers.get('White', '?')} - {game.headers.get('Black', '?')} | "
        f"result {game.headers.get('Result', '*')}"
    )
    print(
        f"Moves: {len(moves)} plies | captures: {stats['captures']} | "
        f"castles: {stats['castles']} | promotions: {stats['promotions']}"
    )
    print(f"Executing plies {start_ply + 1} through {end_ply}.")
    print("Starting FEN:", board.fen())
    if args.live:
        print("LIVE MODE: the arm will move pieces for BOTH White and Black.")
        print("Place the real board in the exact starting position shown above.")
        print("Keep the board path clear and ensure the capture tray can hold 13 pieces.")
        if not args.yes:
            response = input("Type RUN to start the full physical replay: ").strip().upper()
            if response != "RUN":
                print("Cancelled; nothing moved.")
                return 0
    else:
        print("DRY RUN: no robot request will be made. Add --live for real movement.")

    capture_mode = "manual" if args.manual_captures else "arm"
    arm = PhysicalChessArm(FOLDER, dry_run=not args.live, capture_mode=capture_mode)
    arm.load_capture_drop(required=args.live and capture_mode == "arm")
    if not args.live and capture_mode == "arm" and arm.capture_drop is None:
        arm.capture_mode = "manual"
    arm.connect()
    arm.return_to_init()

    last_san: str | None = None
    for ply_index in range(start_ply, end_ply):
        move = moves[ply_index]
        if move not in board.legal_moves:
            raise RuntimeError(
                f"Replay state diverged before ply {ply_index + 1}: {move.uci()} is illegal"
            )
        san = board.san(move)
        side = "White" if board.turn == chess.WHITE else "Black"
        move_number = board.fullmove_number
        print(
            f"\n=== Ply {ply_index + 1}/{len(moves)} | "
            f"{side} {move_number}{'.' if board.turn == chess.WHITE else '...'} {san} "
            f"({move.uci()}) ==="
        )
        try:
            arm.execute_chess_move(board, move)
        except (ArmError, KeyboardInterrupt) as exc:
            save_checkpoint(
                pgn_path=pgn_path,
                next_ply=ply_index,
                board=board,
                last_san=last_san,
            )
            print(f"\nREPLAY STOPPED before logical completion of ply {ply_index + 1}: {exc}")
            print("Restore the physical board to this FEN before resuming:", board.fen())
            print(
                "Resume command: python replay_game.py "
                f'"{pgn_path}" --live --yes --start-ply {ply_index}'
            )
            return 1

        board.push(move)
        last_san = san
        save_checkpoint(
            pgn_path=pgn_path,
            next_ply=ply_index + 1,
            board=board,
            last_san=last_san,
        )
        print("Completed:", san, "| FEN:", board.fen())
        if args.delay_seconds:
            time.sleep(float(args.delay_seconds))

    print(
        f"\nMovement replay complete: {end_ply - start_ply} plies executed. "
        f"Final result: {game.headers.get('Result', '*')}"
    )
    print("Checkpoint:", CHECKPOINT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
