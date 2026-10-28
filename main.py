"""N10: complete chess game using the best engine and calibrated RoArm motion."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn

from engine import TimingHybridEngine
from physical_arm import ArmError, PhysicalChessArm


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
GAMES_DIR = ROOT / "games"

DEFAULT_CONFIG = {
    "arm_enabled": True,
    "arm_ip": "192.168.4.1",
    "capture_mode": "arm",
    "human_color": "black",
    "search_seconds": 30.0,
    "minimum_depth": 4,
    "maximum_depth": 8,
    "use_opening_book": True,
    "tactical_guard_enabled": False,
    "tactical_guard_depth": 16,
    "tactical_guard_override_cp": 100.0,
    "stockfish_path": "assets/stockfish/stockfish-windows-x86-64-avx2.exe",
    "confirm_each_engine_move": False,
}


def load_config() -> dict:
    config = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        config.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return config


def configured_stockfish_path(config: dict) -> Path | None:
    value = config.get("stockfish_path")
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def print_guard_status(engine: TimingHybridEngine):
    guard = engine.tactical_guard
    if not guard.enabled:
        print("Stockfish tactical guard: disabled")
    elif guard.available:
        print(
            f"Stockfish tactical guard ready: depth {guard.depth}, "
            f"override only above {guard.override_cp:.0f}cp"
        )
    else:
        print(f"WARNING: Stockfish tactical guard unavailable; neural engine will continue: {guard.unavailable_reason}")


def print_guard_decision(board: chess.Board, result):
    guard = getattr(result, "guard", None)
    if not guard or not guard.get("enabled"):
        return
    if not guard.get("available"):
        print(f"Tactical guard fallback: {guard.get('reason', 'Stockfish unavailable')}")
        return
    original = chess.Move.from_uci(guard["original_move"])
    best = chess.Move.from_uci(guard["best_move"])
    loss = float(guard.get("loss_cp", 0.0))
    elapsed = float(guard.get("elapsed_seconds", 0.0))
    if guard.get("overrode"):
        print(
            f"Tactical guard OVERRIDE: {board.san(original)} -> {board.san(best)} | "
            f"original was {loss:.0f}cp worse | Stockfish depth {guard['depth']} in {elapsed:.2f}s"
        )
    else:
        print(
            f"Tactical guard kept {board.san(original)} | best={board.san(best)} "
            f"loss={loss:.0f}cp | Stockfish depth {guard['depth']} in {elapsed:.2f}s"
        )


def parse_color(value: str) -> chess.Color:
    return chess.WHITE if value.lower() == "white" else chess.BLACK


def parse_human_move(board: chess.Board, text: str) -> chess.Move:
    text = text.strip()
    try:
        return board.parse_san(text)
    except ValueError:
        pass
    try:
        move = chess.Move.from_uci(text.lower())
    except ValueError as exc:
        raise ValueError("Use SAN (Nf3, O-O) or UCI notation (g1f3, e7e8q).") from exc
    if move not in board.legal_moves:
        raise ValueError(f"{text!r} is not legal in this position.")
    return move


def san_history(board: chess.Board) -> list[str]:
    replay = board.root()
    history = []
    for move in board.move_stack:
        history.append(replay.san(move))
        replay.push(move)
    return history


def print_board(board: chess.Board, human_color: chess.Color):
    try:
        print("\n" + board.unicode(borders=True, empty_square="·", orientation=human_color))
    except (TypeError, UnicodeEncodeError):
        print("\n" + str(board))
    print("FEN:", board.fen())


def game_names(human_color: chess.Color) -> tuple[str, str]:
    if human_color == chess.WHITE:
        return "Human", "N10 timing-hybrid-aggressive"
    return "N10 timing-hybrid-aggressive", "Human"


def write_game(board: chess.Board, human_color: chess.Color, session_path: Path, result: str = "*"):
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    game = chess.pgn.Game.from_board(board)
    white, black = game_names(human_color)
    game.headers.update(
        {
            "Event": "N10 physical chess game",
            "Site": "Physical board",
            "Date": datetime.now().strftime("%Y.%m.%d"),
            "Round": "1",
            "White": white,
            "Black": black,
            "Result": result,
        }
    )
    text = str(game) + "\n"
    session_path.write_text(text, encoding="utf-8")
    (GAMES_DIR / "latest.pgn").write_text(text, encoding="utf-8")


def load_resumed_board(path: str | Path) -> chess.Board:
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        raise ValueError(f"No PGN game found in {path}")
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board


def make_arm(config: dict, dry_run: bool = False) -> PhysicalChessArm:
    return PhysicalChessArm(
        ROOT,
        dry_run=dry_run,
        capture_mode=config["capture_mode"],
    )


def prepare_arm(config: dict, dry_run: bool = False) -> PhysicalChessArm:
    arm = make_arm(config, dry_run=dry_run)
    arm.connect()
    if dry_run:
        arm.load_capture_drop(required=False)
        if arm.capture_mode == "arm" and arm.capture_drop is None:
            print("No capture-drop point is saved; dry-run captures will use manual removal.")
            arm.capture_mode = "manual"
        arm.return_to_init()
        return arm
    arm.load_capture_drop(required=False)
    if arm.capture_mode == "arm" and arm.capture_drop is None:
        answer = input(
            "Capture tray is not calibrated. Press Enter to calibrate it now, "
            "or type MANUAL to remove captures yourself: "
        ).strip().lower()
        if answer == "manual":
            arm.capture_mode = "manual"
        else:
            arm.calibrate_capture_drop()
    arm.return_to_init()
    return arm


def human_turn(board: chess.Board) -> tuple[chess.Move | None, str | None]:
    while True:
        raw = input("\nYour move (SAN/UCI, help, claim, resign, or quit): ").strip()
        command = raw.lower()
        if command in {"help", "?"}:
            print("Examples: e4, Nf3, O-O, e2e4, e7e8q")
            print("Move the piece physically first, then enter the same move here.")
            continue
        if command in {"board", "b"}:
            print(board)
            continue
        if command in {"claim", "draw"}:
            if board.can_claim_draw():
                return None, "1/2-1/2"
            print("No draw can currently be claimed.")
            continue
        if command in {"resign", "r"}:
            return None, "0-1" if board.turn == chess.WHITE else "1-0"
        if command in {"quit", "q", "exit"}:
            return None, "*"
        try:
            return parse_human_move(board, raw), None
        except ValueError as exc:
            print(f"Invalid move: {exc}")
            print("If you already moved a piece physically, put it back before trying again.")


def play_game(args, config: dict) -> int:
    human_color = parse_color(args.human or config["human_color"])
    board = load_resumed_board(args.resume) if args.resume else chess.Board()
    history = san_history(board)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_path = GAMES_DIR / f"game_{timestamp}.pgn"

    arm = None
    arm_enabled = (bool(config["arm_enabled"]) or args.dry_arm) and not args.no_arm
    if arm_enabled:
        arm = prepare_arm(config, dry_run=args.dry_arm)
    else:
        print("Terminal-only mode: computer moves will not be sent to the arm.")

    print("\nLoading timing-hybrid-aggressive engine...")
    engine = TimingHybridEngine(
        search_seconds=args.seconds if args.seconds is not None else config["search_seconds"],
        min_depth=config["minimum_depth"],
        max_depth=config["maximum_depth"],
        use_opening_book=bool(config["use_opening_book"]) and not args.no_book,
        tactical_guard_enabled=bool(config["tactical_guard_enabled"]) and not args.no_tactical_guard,
        tactical_guard_depth=config["tactical_guard_depth"],
        tactical_guard_override_cp=config["tactical_guard_override_cp"],
        stockfish_path=configured_stockfish_path(config),
    )
    print(f"Engine ready on {engine.device}. You are {'White' if human_color else 'Black'}.")
    print_guard_status(engine)
    write_game(board, human_color, session_path)

    final_result = "*"
    try:
        while True:
            outcome = board.outcome(claim_draw=False)
            if outcome is not None:
                final_result = outcome.result()
                break

            print_board(board, human_color)
            if board.turn == human_color:
                move, manual_result = human_turn(board)
                if manual_result is not None:
                    final_result = manual_result
                    break
                assert move is not None
                san = board.san(move)
                board.push(move)
                history.append(san)
                print(f"You played {san} ({move.uci()}).")
            else:
                side = "White" if board.turn == chess.WHITE else "Black"
                print(f"\n{side} engine is thinking...")
                result = engine.choose_move(board, history)
                move = result.move
                san = board.san(move)
                side_score = result.score_cp if board.turn == chess.WHITE else -result.score_cp
                print(
                    f"Engine chose {san} ({move.uci()}) | source={result.source} "
                    f"depth={result.depth} eval={side_score:+.0f}cp time={result.elapsed:.1f}s"
                )
                print_guard_decision(board, result)
                if arm is not None:
                    if config.get("confirm_each_engine_move", True) and not args.dry_arm:
                        answer = input("Press Enter to execute, or type STOP to abort: ").strip().lower()
                        if answer == "stop":
                            raise ArmError("Stopped before physical execution")
                    arm.execute_chess_move(board, move)
                board.push(move)
                history.append(san)
            write_game(board, human_color, session_path)
    except (KeyboardInterrupt, EOFError):
        print("\nGame interrupted. The last completed position has been saved.")
    except ArmError as exc:
        print(f"\nARM STOP: {exc}")
        print("The logical board was not advanced for the failed computer move.")
    finally:
        write_game(board, human_color, session_path, final_result)
        engine.close()

    if final_result == "*":
        print(f"Game paused. Resume with: python main.py --resume \"{session_path}\"")
    else:
        print_board(board, human_color)
        print("Game over:", final_result)
    print("PGN saved to:", session_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play physical chess against N10.")
    parser.add_argument("--human", choices=["white", "black"], help="Your color")
    parser.add_argument("--seconds", type=float, help="Nominal engine time per non-book move")
    parser.add_argument("--no-arm", action="store_true", help="Play in the terminal without arm movement")
    parser.add_argument("--dry-arm", action="store_true", help="Print arm commands without sending them")
    parser.add_argument("--no-book", action="store_true", help="Disable the bundled opening book")
    parser.add_argument("--no-tactical-guard", action="store_true", help="Disable the bundled Stockfish blunder guard")
    parser.add_argument("--resume", help="Resume the logical position from a saved PGN")
    parser.add_argument("--calibrate-capture", action="store_true", help="Calibrate only the capture-tray release point")
    parser.add_argument("--open-gripper", action="store_true", help="Open only the jaws for recovery")
    parser.add_argument("--self-test", action="store_true", help="Load the model and run a fast engine smoke test")
    parser.add_argument("--guard-test", action="store_true", help="Replay the Rb8+/g6 tactical-guard regression")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config()
    if args.calibrate_capture:
        arm = make_arm(config)
        arm.connect()
        arm.calibrate_capture_drop()
        return 0
    if args.open_gripper:
        arm = make_arm(config)
        arm.open_gripper_now()
        return 0
    if args.self_test:
        print("Loading engine for self-test...")
        engine = TimingHybridEngine(
            search_seconds=1.0,
            use_opening_book=False,
            tactical_guard_enabled=bool(config["tactical_guard_enabled"]),
            tactical_guard_depth=config["tactical_guard_depth"],
            tactical_guard_override_cp=config["tactical_guard_override_cp"],
            stockfish_path=configured_stockfish_path(config),
        )
        try:
            result = engine.smoke_test()
            print(
                f"PASS: {result.move.uci()} source={result.source} "
                f"depth={result.depth} nodes={result.nodes} device={engine.device}"
            )
            print_guard_status(engine)
        finally:
            engine.close()
        return 0
    if args.guard_test:
        print("Loading the move-40 tactical-guard regression...")
        engine = TimingHybridEngine(
            search_seconds=1.0,
            use_opening_book=False,
            tactical_guard_enabled=True,
            tactical_guard_depth=config["tactical_guard_depth"],
            tactical_guard_override_cp=config["tactical_guard_override_cp"],
            stockfish_path=configured_stockfish_path(config),
        )
        try:
            guard = engine.tactical_guard_regression_test()
            print(
                "PASS: reproduced N6 Rb8+ was overridden with g6 | "
                f"loss={guard['loss_cp']:.0f}cp depth={guard['depth']} "
                f"time={guard['elapsed_seconds']:.2f}s"
            )
        finally:
            engine.close()
        return 0
    return play_game(args, config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArmError, FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
