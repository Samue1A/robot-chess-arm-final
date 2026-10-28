"""Chess rules adapter for the calibrated RoArm placement controller."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import requests

from roarm_controller import RoArmPlacementController


PIECE_NAMES = {
    chess.KING: "king",
    chess.QUEEN: "queen",
    chess.ROOK: "rook",
    chess.BISHOP: "bishop",
    chess.KNIGHT: "knight",
    chess.PAWN: "pawn",
}


class ArmError(RuntimeError):
    pass


class PhysicalChessArm:
    def __init__(
        self,
        folder: str | Path,
        *,
        dry_run: bool = False,
        capture_mode: str = "arm",
    ):
        if capture_mode not in {"arm", "manual"}:
            raise ValueError("capture_mode must be 'arm' or 'manual'")
        self.folder = Path(folder)
        self.dry_run = bool(dry_run)
        self.capture_mode = capture_mode
        self.capture_path = self.folder / "capture_drop.json"
        self.controller = RoArmPlacementController(
            self.folder,
            live=not self.dry_run,
            assume_yes=True,
        )
        self.capture_drop: tuple[float, float, float] | None = None

    def connect(self):
        self.controller.status()

    def return_to_init(self):
        self.controller.move_init("Game initial position")

    def open_gripper_now(self):
        self.controller.open_gripper_now()

    def load_capture_drop(self, *, required: bool = False):
        self.capture_drop = None
        if self.capture_path.exists():
            payload = json.loads(self.capture_path.read_text(encoding="utf-8"))
            point = payload.get("xyz")
            if isinstance(point, list) and len(point) == 3:
                self.capture_drop = tuple(float(value) for value in point)
        if required and self.capture_drop is None:
            raise ArmError("Capture tray is not calibrated")

    def calibrate_capture_drop(self):
        if self.dry_run:
            raise ArmError("Cannot calibrate the capture tray in dry-run mode")
        print("\nCAPTURE-TRAY CALIBRATION")
        print("Put a tray outside the board where the arm can release captured pieces safely.")
        self.controller.gripper(
            float(self.controller.settings["gripper"]["open_angle"]),
            label="open for capture calibration",
        )
        self.controller.set_torque_lock(False)
        try:
            input(
                "Support and move the arm by hand until the open jaws are over the tray at "
                "the desired release height, then press Enter..."
            )
            data = self.controller.feedback()
            assert data is not None
            point = [round(float(value), 6) for value in data["xyz"]]
        finally:
            self.controller.set_torque_lock(True)
        self.capture_path.write_text(json.dumps({"xyz": point}, indent=2) + "\n", encoding="utf-8")
        self.capture_drop = tuple(point)
        print("Saved capture release point:", tuple(round(value, 1) for value in point))
        self.controller.move_init("Capture calibration complete")

    def transfer_piece(self, source: str, destination: str, piece_name: str):
        self.controller.execute(source, destination, piece_name)

    def remove_captured_piece(self, square_name: str, piece_name: str):
        if self.capture_mode == "manual":
            if self.dry_run:
                print(f"[dry arm] manually remove {piece_name} from {square_name.upper()}")
            else:
                input(
                    f"Remove the captured {piece_name} from {square_name.upper()}, "
                    "then press Enter..."
                )
            return
        if self.capture_drop is None:
            raise ArmError("Capture tray is not calibrated")
        self.controller.remove_to_capture_drop(square_name, piece_name, self.capture_drop)

    def execute_chess_move(self, board: chess.Board, move: chess.Move):
        try:
            if move not in board.legal_moves:
                raise ArmError(f"Refusing to execute illegal move {move.uci()}")
            moving_piece = board.piece_at(move.from_square)
            if moving_piece is None:
                raise ArmError(f"No piece at {chess.square_name(move.from_square)}")
            piece_name = PIECE_NAMES[moving_piece.piece_type]
            source = chess.square_name(move.from_square)
            destination = chess.square_name(move.to_square)

            if board.is_capture(move):
                capture_square = move.to_square
                if board.is_en_passant(move):
                    capture_square += -8 if moving_piece.color == chess.WHITE else 8
                captured_piece = board.piece_at(capture_square)
                if captured_piece is None:
                    raise ArmError("Could not identify the captured piece")
                self.remove_captured_piece(
                    chess.square_name(capture_square),
                    PIECE_NAMES[captured_piece.piece_type],
                )

            if board.is_castling(move):
                self.transfer_piece(source, destination, "king")
                rank = chess.square_rank(move.from_square)
                if chess.square_file(move.to_square) > chess.square_file(move.from_square):
                    rook_source = chess.square_name(chess.square(7, rank))
                    rook_destination = chess.square_name(chess.square(5, rank))
                else:
                    rook_source = chess.square_name(chess.square(0, rank))
                    rook_destination = chess.square_name(chess.square(3, rank))
                self.transfer_piece(rook_source, rook_destination, "rook")
            else:
                self.transfer_piece(source, destination, piece_name)

            if move.promotion:
                promoted_name = PIECE_NAMES[move.promotion]
                message = (
                    f"Promotion: replace the pawn on {destination.upper()} with a "
                    f"{promoted_name}."
                )
                if self.dry_run:
                    print("[dry arm]", message)
                else:
                    input(message + " Press Enter when done...")
        except ArmError:
            raise
        except (RuntimeError, ValueError, requests.RequestException) as exc:
            raise ArmError(str(exc)) from exc
