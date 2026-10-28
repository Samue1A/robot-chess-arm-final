"""Standalone timing-hybrid-aggressive chess engine used by the N10 release.

The search core is carried over from n6/play/timing_hybrid_aggressive.ipynb.
The bundled Stockfish verifier is a narrow tactical guard around that primary
engine. Notebook UI, experiments, and alternate variants are omitted.
"""

from __future__ import annotations

import math
import random
import re
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import chess
import chess.engine
import chess.polyglot
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PACKAGE_DIR = Path(__file__).resolve().parent
ASSET_DIR = PACKAGE_DIR / "assets"
DEFAULT_WEIGHTS_PATH = ASSET_DIR / "timing_hybrid_aggressive.pth"
DEFAULT_OPENING_PATH = ASSET_DIR / "openings.npz"
DEFAULT_STOCKFISH_PATH = ASSET_DIR / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
DEFAULT_TACTICAL_GUARD_DEPTH = 16
DEFAULT_TACTICAL_GUARD_OVERRIDE_CP = 100.0
TACTICAL_GUARD_MATE_CP = 1_000_000

# Model/search values from the active N6 timing_hybrid_aggressive notebook.
VALUE_CP_SCALE = 400.0
DEFAULT_MODEL_TIME = 30.0
DEFAULT_MODEL_MAX_DEPTH = 8
DEFAULT_MODEL_MIN_DEPTH = 4
OPENING_BOOK_PLIES = 12

SEARCH_ROOT_TOP_N = 12
SEARCH_CHILD_TOP_N = 6
SEARCH_QUIESCENCE_DEPTH = 2
SEARCH_EXTENSION_LIMIT = 2
SEARCH_FULL_CLAIM_DRAW_CHECKS = False
EVAL_CACHE_MAX_ITEMS = 200_000
TT_MAX_ITEMS = 500_000

BATCHED_MINIMAX_ENABLED = False
BATCHED_MINIMAX_MAX_DEPTH = 2
BATCHED_EVAL_BATCH_SIZE = 128
BATCHED_FRONTIER_Q_DEPTH = 1
BATCHED_FRONTIER_Q_MAX_MOVES = 10
BATCHED_FRONTIER_MAX_NODES = 60_000
BATCHED_DIRECT_MIN_DEPTH = False
BATCHED_DIRECT_SHALLOW_DEPTHS = 2

STRICT_TIME_LIMIT = False
STRICT_TIME_AFTER_DEPTH = 2
STRICT_TIME_SAFETY_SECONDS = 0.25
SEARCH_GUARDS_USE_REMAINING_TIME = True

TIMING_GATE_ENABLED = True
TIMING_GATE_MIN_DEPTH = 4
TIMING_GATE_DEFAULT_MULTIPLIER = 3.5
TIMING_GATE_MIN_MULTIPLIER = 2.25
TIMING_GATE_MAX_MULTIPLIER = 7.5
TIMING_GATE_MAX_TOTAL_SECONDS = 55.0
TIMING_GATE_MAX_OVERRUN_RATIO = 1.8
TIMING_GATE_SAFETY_SECONDS = 2.5
TIMING_GATE_ESTIMATE_MARGIN = 1.08
TIMING_GATE_MIN_NEXT_DEPTH_SECONDS = 1.0
TIMING_GATE_LOPSIDED_CP = 700.0
TIMING_GATE_CRUSHING_CP = 1100.0
TIMING_GATE_HOPELESS_CP = 1600.0
TIMING_GATE_HOPELESS_MATERIAL_CP = 900.0
TIMING_GATE_HOPELESS_OWN_MATERIAL_CP = 120.0
TIMING_GATE_FAST_LOST_DEPTH = 2
TIMING_GATE_FAST_LOST_MAX_TIME = 4.0
TIMING_GATE_BALANCED_CP = 220.0
TIMING_GATE_BALANCED_MATERIAL_CP = 220.0
TIMING_GATE_SHARP_FORCING_MOVES = 7
TIMING_GATE_SHARP_MAX_TOTAL_SECONDS = 75.0
TIMING_GATE_SHARP_OVERRUN_RATIO = 2.4
TIMING_GATE_SHARP_ESTIMATE_MARGIN = 1.0

DRAW_CONTEMPT_CP = 140.0
DRAW_CONTEMPT_TRIGGER_CP = 90.0
CONVERSION_TRIGGER_CP = 180.0
CONVERSION_MARGIN_CP = 90.0
CONVERSION_MIN_MATERIAL_CP = 300.0
CONVERSION_ENDGAME_MIN_MATERIAL_CP = 180.0
CONVERSION_ENDGAME_MAX_NONKING = 8

HYBRID_GUARD_ENABLED = True
HYBRID_GUARD_DEPTH = 4
HYBRID_GUARD_SHARP_DEPTH = 5
HYBRID_GUARD_MAX_TIME = 4.0
HYBRID_GUARD_MIN_TIME = 1.0
HYBRID_GUARD_MARGIN_CP = 150.0
HYBRID_GUARD_QUIET_MARGIN_CP = 260.0
HYBRID_GUARD_NN_TOLERANCE_CP = 330.0

ENDGAME_SPECIALIST_ENABLED = True
ENDGAME_SPECIALIST_DEPTH = 10
ENDGAME_SPECIALIST_MAX_TIME = 10.0
ENDGAME_SPECIALIST_MIN_ADV_CP = 450.0

OBVIOUS_MOVE_FASTPATH_ENABLED = True
OBVIOUS_MOVE_SCRATCH_DEPTH = 3
OBVIOUS_MOVE_MARGIN_CP = 750.0

REPLY_AUDIT_ENABLED = True
REPLY_AUDIT_DEPTH = 3
REPLY_AUDIT_MAX_TIME = 4.0
REPLY_AUDIT_CANDIDATES = 4
REPLY_AUDIT_STATIC_DROP_CP = 110.0
REPLY_AUDIT_SHARP_STATIC_DROP_CP = 65.0
REPLY_AUDIT_OVERRIDE_MARGIN_CP = 90.0

ENABLE_SYZYGY = False
SYZYGY_MAX_PIECES = 5

RANDOM_SEED = 7
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ResBlock(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class ChessNet(nn.Module):
    def __init__(self, in_ch: int = 19, filters: int = 128, num_blocks: int = 10, policy_dim: int = 4672):
        super().__init__()
        self.input_conv = nn.Conv2d(in_ch, filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(filters)
        self.res_blocks = nn.Sequential(*[ResBlock(filters) for _ in range(num_blocks)])
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 64)
        self.value_fc2 = nn.Linear(64, 1)
        self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, policy_dim)

    def forward_features(self, x):
        x = F.relu(self.input_bn(self.input_conv(x)), inplace=True)
        return self.res_blocks(x)

    def forward_raw(self, x, legal_mask=None):
        x = self.forward_features(x)
        v = F.relu(self.value_bn(self.value_conv(x)), inplace=True)
        v = v.view(v.size(0), -1)
        v = self.value_fc2(F.relu(self.value_fc1(v), inplace=True))
        p = F.relu(self.policy_bn(self.policy_conv(x)), inplace=True)
        p = self.policy_fc(p.view(p.size(0), -1))
        if legal_mask is not None:
            p = p.masked_fill(~legal_mask, float("-inf"))
        return v, p

    def forward(self, x, legal_mask=None):
        v, p = self.forward_raw(x, legal_mask)
        return torch.sigmoid(v), F.softmax(p, dim=-1)


def load_model(path: str | Path = DEFAULT_WEIGHTS_PATH, device: torch.device | None = None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint is missing: {path}")
    device = device or best_device()
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state") or checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state is None:
        raise KeyError("Checkpoint does not contain model_state")
    in_ch = int(checkpoint.get("input_channels", checkpoint.get("in_ch", 19)))
    filters = int(checkpoint.get("filters", 128))
    blocks = int(checkpoint.get("num_blocks", 10))
    policy_dim = int(checkpoint.get("policy_dim", 4672))
    model = ChessNet(in_ch=in_ch, filters=filters, num_blocks=blocks, policy_dim=policy_dim).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, device


PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
POLICY_DIM = 4672


def board_to_icpram_bitmap_plus(board: chess.Board, channels_first: bool = True) -> np.ndarray:
    planes: list[np.ndarray] = []
    for color, sign in ((chess.WHITE, +1.0), (chess.BLACK, -1.0)):
        for piece_type in PIECE_ORDER:
            plane = np.zeros((8, 8), dtype=np.float32)
            for square in board.pieces(piece_type, color):
                plane[7 - chess.square_rank(square), chess.square_file(square)] = sign
            planes.append(plane)
    planes.append(np.full((8, 8), float(board.has_kingside_castling_rights(chess.WHITE)), dtype=np.float32))
    planes.append(np.full((8, 8), float(board.has_queenside_castling_rights(chess.WHITE)), dtype=np.float32))
    planes.append(np.full((8, 8), float(board.has_kingside_castling_rights(chess.BLACK)), dtype=np.float32))
    planes.append(np.full((8, 8), float(board.has_queenside_castling_rights(chess.BLACK)), dtype=np.float32))
    planes.append(np.full((8, 8), 1.0 if board.turn == chess.WHITE else -1.0, dtype=np.float32))
    ep = np.zeros((8, 8), dtype=np.float32)
    if board.ep_square is not None:
        ep[7 - chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1.0
    planes.append(ep)
    planes.append(np.full((8, 8), min(board.halfmove_clock, 100) / 100.0, dtype=np.float32))
    encoded = np.stack(planes, axis=0)
    return encoded if channels_first else np.transpose(encoded, (1, 2, 0))


def move_to_index(move: chess.Move) -> int:
    if move.promotion and move.promotion != chess.QUEEN:
        offset = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
        return 4096 + move.from_square * 3 + offset[move.promotion]
    return move.from_square * 64 + move.to_square


def get_legal_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(POLICY_DIM, dtype=np.bool_)
    for move in board.legal_moves:
        index = move_to_index(move)
        if 0 <= index < POLICY_DIM:
            mask[index] = True
    return mask


class OpeningBook:
    """Small runtime index for the bundled Games.npz opening lines."""

    def __init__(self, path: str | Path = DEFAULT_OPENING_PATH, max_plies: int = OPENING_BOOK_PLIES):
        self.path = Path(path)
        self.max_plies = int(max_plies)
        self.next_by_prefix: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        if self.path.exists():
            self._load()

    @property
    def enabled(self) -> bool:
        return bool(self.next_by_prefix)

    def _load(self):
        with np.load(self.path, allow_pickle=True) as data:
            games = data["positions"]
            for tokens in games:
                board = chess.Board()
                prefix: list[str] = []
                for token in list(tokens)[: self.max_plies]:
                    try:
                        move = board.parse_san(str(token))
                    except ValueError:
                        try:
                            move = chess.Move.from_uci(str(token).lower())
                        except ValueError:
                            break
                        if move not in board.legal_moves:
                            break
                    san = board.san(move)
                    self.next_by_prefix[tuple(prefix)][san] += 1
                    board.push(move)
                    prefix.append(san)

    def choose(self, board: chess.Board, san_history: list[str]) -> Optional[chess.Move]:
        choices = self.next_by_prefix.get(tuple(san_history))
        if not choices:
            return None
        for san, _count in choices.most_common():
            try:
                move = board.parse_san(san)
            except ValueError:
                continue
            if move in board.legal_moves:
                return move
        return None


# Improved neural player:
# - timed iterative deepening
# - eval and move-order caches
# - policy-pruned alpha-beta
# - quiescence and tactical/passed-pawn extensions
# - draw contempt and root conversion bias
# - optional Syzygy probing
@dataclass
class SearchResult:
    move: chess.Move
    score_cp: float
    source: str = "nn"
    depth: int = 0
    nodes: int = 0
    elapsed: float = 0.0
    candidates: list[tuple[float, chess.Move]] | None = None
    guard: dict | None = None


class StockfishTacticalGuard:
    """Persistent Stockfish verifier for the primary neural engine's move.

    Stockfish does not replace the timing-hybrid-aggressive search. It evaluates
    that search's selected move and only intervenes when the selected move is
    more than ``override_cp`` worse than Stockfish's best move.
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_STOCKFISH_PATH,
        depth: int = DEFAULT_TACTICAL_GUARD_DEPTH,
        override_cp: float = DEFAULT_TACTICAL_GUARD_OVERRIDE_CP,
        enabled: bool = True,
    ):
        self.path = Path(path).expanduser().resolve()
        self.depth = max(1, int(depth))
        self.override_cp = max(0.0, float(override_cp))
        self.enabled = bool(enabled)
        self.engine: chess.engine.SimpleEngine | None = None
        self.unavailable_reason: str | None = None
        if self.enabled:
            self._start()

    @property
    def available(self) -> bool:
        return self.engine is not None

    def _start(self) -> bool:
        if not self.enabled:
            self.unavailable_reason = "disabled"
            return False
        if not self.path.is_file():
            self.unavailable_reason = f"Stockfish executable not found: {self.path}"
            return False
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(str(self.path))
        except (OSError, RuntimeError, chess.engine.EngineError) as exc:
            self.engine = None
            self.unavailable_reason = f"could not start Stockfish: {exc}"
            return False
        self.unavailable_reason = None
        return True

    def _discard_engine(self):
        engine, self.engine = self.engine, None
        if engine is None:
            return
        try:
            engine.close()
        except Exception:
            pass

    @staticmethod
    def _score_for_side(info: dict, side: chess.Color) -> tuple[int, int | None]:
        score = info["score"].pov(side)
        numeric = score.score(mate_score=TACTICAL_GUARD_MATE_CP)
        if numeric is None:
            raise ValueError("Stockfish returned a score without centipawn or mate value")
        return int(numeric), score.mate()

    def _analyse_once(self, board: chess.Board, chosen: chess.Move) -> dict:
        if self.engine is None:
            raise RuntimeError(self.unavailable_reason or "Stockfish is unavailable")

        started = time.perf_counter()
        limit = chess.engine.Limit(depth=self.depth)
        best_info = self.engine.analyse(board, limit)
        pv = best_info.get("pv") or []
        if not pv:
            raise ValueError("Stockfish returned no principal variation")
        best_move = pv[0]
        best_score, best_mate = self._score_for_side(best_info, board.turn)

        if best_move == chosen:
            chosen_score, chosen_mate = best_score, best_mate
        else:
            chosen_info = self.engine.analyse(board, limit, root_moves=[chosen])
            chosen_score, chosen_mate = self._score_for_side(chosen_info, board.turn)

        loss_cp = max(0, best_score - chosen_score)
        overrode = best_move != chosen and loss_cp > self.override_cp
        selected = best_move if overrode else chosen
        return {
            "enabled": True,
            "available": True,
            "depth": self.depth,
            "override_threshold_cp": self.override_cp,
            "original_move": chosen.uci(),
            "best_move": best_move.uci(),
            "selected_move": selected.uci(),
            "best_score_for_side_cp": best_score,
            "original_score_for_side_cp": chosen_score,
            "best_mate_for_side": best_mate,
            "original_mate_for_side": chosen_mate,
            "loss_cp": loss_cp,
            "overrode": overrode,
            "elapsed_seconds": time.perf_counter() - started,
            "source": str(self.path),
        }

    def choose(self, board: chess.Board, chosen: chess.Move) -> dict:
        if not self.enabled:
            return {"enabled": False, "available": False, "reason": "disabled", "selected_move": chosen.uci()}
        if chosen not in board.legal_moves:
            return {
                "enabled": True,
                "available": self.available,
                "reason": f"primary engine supplied illegal move {chosen.uci()}",
                "selected_move": chosen.uci(),
            }
        if self.engine is None and not self._start():
            return {
                "enabled": True,
                "available": False,
                "reason": self.unavailable_reason,
                "selected_move": chosen.uci(),
            }

        # A terminated subprocess gets one automatic restart. If that also
        # fails, play continues with the primary neural move rather than
        # stopping a physical game.
        for attempt in range(2):
            try:
                return self._analyse_once(board, chosen)
            except (OSError, RuntimeError, TimeoutError, ValueError, chess.engine.EngineError) as exc:
                self.unavailable_reason = f"Stockfish analysis failed: {exc}"
                self._discard_engine()
                if attempt == 0 and self._start():
                    continue
        return {
            "enabled": True,
            "available": False,
            "reason": self.unavailable_reason,
            "selected_move": chosen.uci(),
        }

    def close(self):
        engine, self.engine = self.engine, None
        if engine is None:
            return
        try:
            engine.quit()
        except Exception:
            try:
                engine.close()
            except Exception:
                pass


class SearchTimeout(Exception):
    pass


class BatchedSearchTooLarge(Exception):
    pass


PIECE_VALUE_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def material_cp_white(board: chess.Board) -> int:
    total = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUE_CP.get(piece.piece_type, 0)
        total += value if piece.color == chess.WHITE else -value
    return total


def nonking_piece_count(board: chess.Board) -> int:
    return sum(1 for piece in board.piece_map().values() if piece.piece_type != chess.KING)


def is_passed_pawn(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    piece = board.piece_at(square)
    if piece is None or piece.piece_type != chess.PAWN or piece.color != color:
        return False
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    enemy = not color
    files = [f for f in (file_idx - 1, file_idx, file_idx + 1) if 0 <= f <= 7]
    ranks = range(rank_idx + 1, 8) if color == chess.WHITE else range(rank_idx - 1, -1, -1)
    for f in files:
        for r in ranks:
            p = board.piece_at(chess.square(f, r))
            if p is not None and p.color == enemy and p.piece_type == chess.PAWN:
                return False
    return True


def passed_pawn_danger(board: chess.Board, color: chess.Color) -> float:
    danger = 0.0
    for sq in board.pieces(chess.PAWN, color):
        if not is_passed_pawn(board, sq, color):
            continue
        rank = chess.square_rank(sq)
        progress = rank if color == chess.WHITE else 7 - rank
        if progress >= 4:
            danger += (progress - 3) ** 2
            file_idx = chess.square_file(sq)
            if file_idx in (3, 4):
                danger += 0.75
    return danger


def move_is_dangerous_passer_push(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False
    to_rank = chess.square_rank(move.to_square)
    if piece.color == chess.WHITE and to_rank >= 5:
        return True
    if piece.color == chess.BLACK and to_rank <= 2:
        return True
    return False


def piece_trade_class(piece_type: int) -> str | None:
    if piece_type == chess.QUEEN:
        return "queen"
    if piece_type == chess.ROOK:
        return "rook"
    if piece_type in (chess.BISHOP, chess.KNIGHT):
        return "minor"
    return None


def side_material_cp(board: chess.Board, side: chess.Color) -> int:
    material = material_cp_white(board)
    return material if side == chess.WHITE else -material


def nonpawn_trade_bonus(board: chess.Board, move: chess.Move, root_turn: chess.Color, side_score: float) -> float:
    """Prefer real simplification only when already safely better.

    Real simplification means queen-for-queen, rook-for-rook, or
    bishop/knight-for-bishop/knight. Pawn swaps deliberately get no bonus.
    The bonus is conservative for small edges and larger for clearly won games.
    """
    mover = board.piece_at(move.from_square)
    captured = board.piece_at(move.to_square)
    if mover is None or captured is None:
        return 0.0
    if mover.piece_type == chess.PAWN or captured.piece_type == chess.PAWN:
        return 0.0

    mover_class = piece_trade_class(mover.piece_type)
    captured_class = piece_trade_class(captured.piece_type)
    if mover_class is None or captured_class is None:
        return 0.0

    mover_value = PIECE_VALUE_CP.get(mover.piece_type, 0)
    captured_value = PIECE_VALUE_CP.get(captured.piece_type, 0)
    material_before = side_material_cp(board, root_turn)
    material_after = material_before + captured_value

    # Avoid queen-trade "conversion" when the edge is mostly attacking rather
    # than material. If we are not materially ahead, do not bonus queen trades.
    if captured_class == "queen" and material_before < 250:
        return 0.0

    same_trade = mover_class == captured_class
    wins_material = captured_value > mover_value
    equal_or_better_trade = same_trade or captured_value >= mover_value

    if equal_or_better_trade:
        if side_score < 300:
            return 0.0
        if material_before < 100 and side_score < 700:
            return 0.0

        base = {"queen": 150.0, "rook": 85.0, "minor": 50.0}.get(captured_class, 0.0)
        if side_score < 500:
            scale = 0.30
        elif side_score < 900:
            scale = 0.65
        else:
            scale = 1.00
        if wins_material:
            scale += 0.25
        return base * scale

    # Deliberately losing material to simplify: only when the game is already
    # overwhelmingly won, e.g. two rooks vs a knight type liquidation.
    material_loss = mover_value - captured_value
    if captured_class == "minor" and mover_class == "rook":
        if side_score >= 1000 and material_before >= 800 and material_after >= 500:
            return 35.0
        if side_score >= 1500 and material_before >= 1200 and material_after >= 800:
            return 70.0

    return -40.0 if material_loss > 0 else 0.0


class ClassicalTimeout(Exception):
    pass


CLASSICAL_MATE_SCORE = 100_000.0
CLASSICAL_INF = 1_000_000.0


CLASSICAL_PIECE_VALUE_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def cp_for_color(color: chess.Color, cp_white: float) -> float:
    return float(cp_white) if color == chess.WHITE else -float(cp_white)


def cp_from_color(color: chess.Color, cp_for_side: float) -> float:
    return float(cp_for_side) if color == chess.WHITE else -float(cp_for_side)


def square_chebyshev(a: chess.Square, b: chess.Square) -> int:
    return max(abs(chess.square_file(a) - chess.square_file(b)), abs(chess.square_rank(a) - chess.square_rank(b)))


def square_manhattan(a: chess.Square, b: chess.Square) -> int:
    return abs(chess.square_file(a) - chess.square_file(b)) + abs(chess.square_rank(a) - chess.square_rank(b))


def edge_pressure(square: chess.Square) -> float:
    f = chess.square_file(square)
    r = chess.square_rank(square)
    return max(abs(f - 3.5), abs(r - 3.5)) / 3.5


def center_pressure(square: chess.Square) -> float:
    f = chess.square_file(square)
    r = chess.square_rank(square)
    return 1.0 - (abs(f - 3.5) + abs(r - 3.5)) / 7.0


def square_color(square: chess.Square) -> int:
    return (chess.square_file(square) + chess.square_rank(square)) & 1


def pieces_without_kings(board: chess.Board) -> list[chess.Piece]:
    return [p for p in board.piece_map().values() if p.piece_type != chess.KING]


def has_any_pawns(board: chess.Board) -> bool:
    return bool(board.pieces(chess.PAWN, chess.WHITE) or board.pieces(chess.PAWN, chess.BLACK))


def material_cp_for_color(board: chess.Board, color: chess.Color) -> int:
    return side_material_cp(board, color)


def side_piece_counts(board: chess.Board, color: chess.Color) -> Counter:
    counts = Counter()
    for piece in board.piece_map().values():
        if piece.color == color and piece.piece_type != chess.KING:
            counts[piece.piece_type] += 1
    return counts


def nonking_material_for_color(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for piece in board.piece_map().values():
        if piece.color == color and piece.piece_type != chess.KING:
            total += CLASSICAL_PIECE_VALUE_CP.get(piece.piece_type, 0)
    return total


def mating_material_kind(board: chess.Board, color: chess.Color, min_adv_cp: float | None = None) -> str | None:
    if has_any_pawns(board):
        return None
    if min_adv_cp is None:
        min_adv_cp = globals().get("ENDGAME_SPECIALIST_MIN_ADV_CP", 450.0)
    counts = side_piece_counts(board, color)
    enemy_counts = side_piece_counts(board, not color)
    own_mat = nonking_material_for_color(board, color)
    enemy_mat = nonking_material_for_color(board, not color)
    adv = own_mat - enemy_mat

    if enemy_mat == 0:
        if counts[chess.QUEEN] >= 1:
            return "queen"
        if counts[chess.ROOK] >= 1:
            return "rook"
        if counts[chess.BISHOP] >= 1 and counts[chess.KNIGHT] >= 1:
            return "bishop-knight"
        if counts[chess.BISHOP] >= 2:
            return "two-bishops"
        if counts[chess.QUEEN] + counts[chess.ROOK] >= 2:
            return "major"

    # Converted no-pawn positions where the stronger side still has to win/mate.
    if adv >= float(min_adv_cp) and sum(counts.values()) + sum(enemy_counts.values()) <= 6:
        if counts[chess.QUEEN] or counts[chess.ROOK] or (counts[chess.BISHOP] and counts[chess.KNIGHT]) or counts[chess.BISHOP] >= 2:
            return "converted"
    return None


def classical_terminal_value_white(board: chess.Board) -> float | None:
    outcome = board.outcome(claim_draw=False)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    score = CLASSICAL_MATE_SCORE - min(board.ply(), 10_000)
    return score if outcome.winner == chess.WHITE else -score


def legal_count_for_turn(board: chess.Board, turn: chess.Color) -> int:
    b = board.copy(stack=False)
    b.turn = turn
    try:
        return b.legal_moves.count()
    except Exception:
        return len(list(b.legal_moves))


def classical_static_eval_white(board: chess.Board) -> float:
    terminal = classical_terminal_value_white(board)
    if terminal is not None:
        return terminal

    score = 0.0
    piece_map = board.piece_map()
    endgameish = len(piece_map) <= 12 or not (
        board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK)
    )

    for sq, piece in piece_map.items():
        if piece.piece_type == chess.KING:
            continue
        sign = 1.0 if piece.color == chess.WHITE else -1.0
        value = CLASSICAL_PIECE_VALUE_CP.get(piece.piece_type, 0)
        score += sign * value

        center = center_pressure(sq)
        if piece.piece_type == chess.KNIGHT:
            score += sign * (28.0 * center)
        elif piece.piece_type == chess.BISHOP:
            score += sign * (18.0 * center)
        elif piece.piece_type == chess.ROOK:
            file_idx = chess.square_file(sq)
            same_file_pawns = len(board.pieces(chess.PAWN, chess.WHITE) & chess.BB_FILES[file_idx]) + len(board.pieces(chess.PAWN, chess.BLACK) & chess.BB_FILES[file_idx])
            if same_file_pawns == 0:
                score += sign * 18.0
        elif piece.piece_type == chess.PAWN:
            rank = chess.square_rank(sq) if piece.color == chess.WHITE else 7 - chess.square_rank(sq)
            score += sign * (rank * 7.0)
            if is_passed_pawn(board, sq, piece.color):
                score += sign * (25.0 + rank * rank * 5.0)

        enemy = not piece.color
        attackers = len(board.attackers(enemy, sq))
        defenders = len(board.attackers(piece.color, sq))
        if attackers and piece.piece_type != chess.PAWN:
            if defenders == 0:
                score -= sign * min(260.0, 0.38 * value)
            elif attackers > defenders:
                score -= sign * min(180.0, 0.22 * value)

    for color in (chess.WHITE, chess.BLACK):
        sign = 1.0 if color == chess.WHITE else -1.0
        counts = side_piece_counts(board, color)
        if counts[chess.BISHOP] >= 2:
            score += sign * 35.0
        king_sq = board.king(color)
        if king_sq is not None and endgameish:
            enemy_king = board.king(not color)
            if enemy_king is not None:
                score += sign * (7 - square_chebyshev(king_sq, enemy_king)) * 7.0
        elif king_sq is not None:
            score -= sign * edge_pressure(king_sq) * 18.0

    try:
        white_mob = legal_count_for_turn(board, chess.WHITE)
        black_mob = legal_count_for_turn(board, chess.BLACK)
        score += 1.5 * (white_mob - black_mob)
    except Exception:
        pass

    if board.is_check():
        score += -45.0 if board.turn == chess.WHITE else 45.0

    return float(score)


def classical_move_order_score(board: chess.Board, move: chess.Move) -> float:
    score = 0.0
    mover = board.piece_at(move.from_square)
    victim = None
    if board.is_en_passant(move):
        victim = chess.Piece(chess.PAWN, not board.turn)
    elif board.is_capture(move):
        victim = board.piece_at(move.to_square)
    if victim is not None:
        victim_value = CLASSICAL_PIECE_VALUE_CP.get(victim.piece_type, 0)
        mover_value = CLASSICAL_PIECE_VALUE_CP.get(mover.piece_type, 0) if mover else 0
        score += 10_000.0 + 10.0 * victim_value - mover_value
    if move.promotion:
        score += 8_000.0 + CLASSICAL_PIECE_VALUE_CP.get(move.promotion, 0)
    if board.gives_check(move):
        score += 2_000.0
    if mover is not None and mover.piece_type == chess.KING:
        score -= 20.0
    return score


def classical_ordered_moves(board: chess.Board, moves: Iterable[chess.Move] | None = None) -> list[chess.Move]:
    ordered = list(board.legal_moves if moves is None else moves)
    ordered.sort(key=lambda mv: classical_move_order_score(board, mv), reverse=True)
    return ordered


def classical_forcing_moves(board: chess.Board) -> list[chess.Move]:
    moves = []
    for mv in board.legal_moves:
        if board.is_capture(mv) or board.gives_check(mv) or mv.promotion:
            moves.append(mv)
    return classical_ordered_moves(board, moves)


def classical_check_deadline(deadline: float | None):
    if deadline is not None and time.perf_counter() >= deadline:
        raise ClassicalTimeout()


def classical_quiescence(board: chess.Board, alpha: float, beta: float, q_depth: int, deadline: float | None) -> float:
    classical_check_deadline(deadline)
    terminal = classical_terminal_value_white(board)
    if terminal is not None:
        return terminal

    stand_pat = classical_static_eval_white(board)
    if q_depth <= 0:
        return stand_pat

    if board.turn == chess.WHITE:
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)
        best = stand_pat
        for mv in classical_forcing_moves(board):
            board.push(mv)
            try:
                score = classical_quiescence(board, alpha, beta, q_depth - 1, deadline)
            finally:
                board.pop()
            best = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return float(best)

    if stand_pat <= alpha:
        return stand_pat
    beta = min(beta, stand_pat)
    best = stand_pat
    for mv in classical_forcing_moves(board):
        board.push(mv)
        try:
            score = classical_quiescence(board, alpha, beta, q_depth - 1, deadline)
        finally:
            board.pop()
        best = min(best, score)
        beta = min(beta, best)
        if beta <= alpha:
            break
    return float(best)


def classical_alphabeta(board: chess.Board, depth: int, alpha: float, beta: float, deadline: float | None, q_depth: int = 2) -> float:
    classical_check_deadline(deadline)
    terminal = classical_terminal_value_white(board)
    if terminal is not None:
        return terminal
    if depth <= 0:
        return classical_quiescence(board, alpha, beta, q_depth, deadline)

    moves = classical_ordered_moves(board)
    if board.turn == chess.WHITE:
        best = -CLASSICAL_INF
        for mv in moves:
            board.push(mv)
            try:
                score = classical_alphabeta(board, depth - 1, alpha, beta, deadline, q_depth=q_depth)
            finally:
                board.pop()
            best = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return float(best)

    best = CLASSICAL_INF
    for mv in moves:
        board.push(mv)
        try:
            score = classical_alphabeta(board, depth - 1, alpha, beta, deadline, q_depth=q_depth)
        finally:
            board.pop()
        best = min(best, score)
        beta = min(beta, best)
        if beta <= alpha:
            break
    return float(best)


def classical_root_scores(
    board: chess.Board,
    depth: int = 4,
    candidate_moves: Iterable[chess.Move] | None = None,
    time_limit: float | None = None,
    q_depth: int = 2,
) -> list[tuple[float, chess.Move]]:
    deadline = None if not time_limit or time_limit <= 0 else time.perf_counter() + float(time_limit)
    moves = classical_ordered_moves(board, candidate_moves)
    scored: list[tuple[float, chess.Move]] = []
    alpha, beta = -CLASSICAL_INF, CLASSICAL_INF

    for mv in moves:
        classical_check_deadline(deadline)
        board.push(mv)
        try:
            score = classical_alphabeta(board, max(0, depth - 1), alpha, beta, deadline, q_depth=q_depth)
        finally:
            board.pop()
        scored.append((float(score), mv))
        if board.turn == chess.WHITE:
            alpha = max(alpha, score)
        else:
            beta = min(beta, score)

    scored.sort(key=lambda item: item[0], reverse=(board.turn == chess.WHITE))
    return scored


def endgame_mate_pressure_for_side(board: chess.Board, winning_color: chess.Color, kind: str) -> float:
    enemy = not winning_color
    enemy_king = board.king(enemy)
    own_king = board.king(winning_color)
    if enemy_king is None or own_king is None:
        return 0.0

    pressure = 0.0
    pressure += edge_pressure(enemy_king) * 260.0
    pressure += (7 - square_chebyshev(own_king, enemy_king)) * 28.0

    enemy_moves = legal_count_for_turn(board, enemy)
    pressure += max(0, 8 - enemy_moves) * 22.0

    b = board.copy(stack=False)
    b.turn = enemy
    if b.is_check():
        pressure += 45.0

    if kind == "bishop-knight":
        bishops = list(board.pieces(chess.BISHOP, winning_color))
        if bishops:
            bishop_color = square_color(bishops[0])
            good_corners = [sq for sq in (chess.A1, chess.A8, chess.H1, chess.H8) if square_color(sq) == bishop_color]
            best_corner_dist = min(square_manhattan(enemy_king, corner) for corner in good_corners)
            pressure += (14 - best_corner_dist) * 38.0
            pressure -= min(square_manhattan(enemy_king, corner) for corner in set((chess.A1, chess.A8, chess.H1, chess.H8)) - set(good_corners)) * 3.0
    else:
        best_corner_dist = min(square_manhattan(enemy_king, corner) for corner in (chess.A1, chess.A8, chess.H1, chess.H8))
        pressure += (14 - best_corner_dist) * 12.0

    # Keep the stronger king active and discourage aimless piece shuffling.
    for sq, piece in board.piece_map().items():
        if piece.color != winning_color or piece.piece_type == chess.KING:
            continue
        pressure += max(0, 7 - square_chebyshev(sq, enemy_king)) * 5.0

    return float(pressure)


def endgame_static_eval_white(board: chess.Board, winning_color: chess.Color, kind: str) -> float:
    terminal = classical_terminal_value_white(board)
    if terminal is not None:
        return terminal

    side_material = material_cp_for_color(board, winning_color)
    pressure = endgame_mate_pressure_for_side(board, winning_color, kind)
    halfmove_penalty = min(board.halfmove_clock, 90) * 3.0
    side_score = side_material + pressure - halfmove_penalty
    return cp_from_color(winning_color, side_score)


def endgame_alphabeta(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    deadline: float | None,
    winning_color: chess.Color,
    kind: str,
) -> float:
    classical_check_deadline(deadline)
    terminal = classical_terminal_value_white(board)
    if terminal is not None:
        return terminal
    if depth <= 0:
        return endgame_static_eval_white(board, winning_color, kind)

    moves = classical_ordered_moves(board)
    if board.turn == chess.WHITE:
        best = -CLASSICAL_INF
        for mv in moves:
            board.push(mv)
            try:
                score = endgame_alphabeta(board, depth - 1, alpha, beta, deadline, winning_color, kind)
            finally:
                board.pop()
            best = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return float(best)

    best = CLASSICAL_INF
    for mv in moves:
        board.push(mv)
        try:
            score = endgame_alphabeta(board, depth - 1, alpha, beta, deadline, winning_color, kind)
        finally:
            board.pop()
        best = min(best, score)
        beta = min(beta, best)
        if beta <= alpha:
            break
    return float(best)


def endgame_root_scores(
    board: chess.Board,
    winning_color: chess.Color,
    kind: str,
    depth: int = 10,
    time_limit: float | None = None,
) -> list[tuple[float, chess.Move]]:
    deadline = None if not time_limit or time_limit <= 0 else time.perf_counter() + float(time_limit)
    moves = classical_ordered_moves(board)
    scored: list[tuple[float, chess.Move]] = []
    alpha, beta = -CLASSICAL_INF, CLASSICAL_INF
    for mv in moves:
        classical_check_deadline(deadline)
        board.push(mv)
        try:
            score = endgame_alphabeta(board, max(0, depth - 1), alpha, beta, deadline, winning_color, kind)
        finally:
            board.pop()
        scored.append((float(score), mv))
        if board.turn == chess.WHITE:
            alpha = max(alpha, score)
        else:
            beta = min(beta, score)
    scored.sort(key=lambda item: item[0], reverse=(board.turn == chess.WHITE))
    return scored


def position_sharpness(board: chess.Board) -> float:
    forcing = 0
    hanging = 0
    for mv in board.legal_moves:
        if board.is_capture(mv) or board.gives_check(mv) or mv.promotion:
            forcing += 1
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        if board.is_attacked_by(not piece.color, sq):
            value = CLASSICAL_PIECE_VALUE_CP.get(piece.piece_type, 0)
            if value >= 300:
                hanging += 1
    return float(forcing * 18 + hanging * 35 + (80 if board.is_check() else 0))



class NeuralChessPlayer:
    def __init__(
        self,
        model: ChessNet,
        device: torch.device,
        cp_scale: float = 400.0,
        mate_score: float = 100_000.0,
        tt_max_items: int = TT_MAX_ITEMS,
        eval_cache_max_items: int = EVAL_CACHE_MAX_ITEMS,
        root_top_n: int = SEARCH_ROOT_TOP_N,
        child_top_n: int = SEARCH_CHILD_TOP_N,
        q_depth: int = SEARCH_QUIESCENCE_DEPTH,
        extension_limit: int = SEARCH_EXTENSION_LIMIT,
        batched_minimax_enabled: bool = BATCHED_MINIMAX_ENABLED,
        batched_minimax_max_depth: int = BATCHED_MINIMAX_MAX_DEPTH,
        batched_eval_batch_size: int = BATCHED_EVAL_BATCH_SIZE,
        batched_frontier_q_depth: int = BATCHED_FRONTIER_Q_DEPTH,
        batched_frontier_q_max_moves: int = BATCHED_FRONTIER_Q_MAX_MOVES,
        batched_frontier_max_nodes: int = BATCHED_FRONTIER_MAX_NODES,
        batched_direct_min_depth: bool = BATCHED_DIRECT_MIN_DEPTH,
        batched_direct_shallow_depths: int = BATCHED_DIRECT_SHALLOW_DEPTHS,
        draw_contempt_cp: float = DRAW_CONTEMPT_CP,
        draw_contempt_trigger_cp: float = DRAW_CONTEMPT_TRIGGER_CP,
        conversion_trigger_cp: float = CONVERSION_TRIGGER_CP,
        conversion_margin_cp: float = CONVERSION_MARGIN_CP,
        conversion_min_material_cp: float = CONVERSION_MIN_MATERIAL_CP,
        conversion_endgame_min_material_cp: float = CONVERSION_ENDGAME_MIN_MATERIAL_CP,
        conversion_endgame_max_nonking: int = CONVERSION_ENDGAME_MAX_NONKING,
        syzygy_path: str | Path | None = None,
        syzygy_max_pieces: int = SYZYGY_MAX_PIECES,
        variant_name: str = "hybrid",
        hybrid_guard_enabled: bool = HYBRID_GUARD_ENABLED,
        hybrid_guard_depth: int = HYBRID_GUARD_DEPTH,
        hybrid_guard_sharp_depth: int = HYBRID_GUARD_SHARP_DEPTH,
        hybrid_guard_max_time: float = HYBRID_GUARD_MAX_TIME,
        hybrid_guard_min_time: float = HYBRID_GUARD_MIN_TIME,
        hybrid_guard_margin_cp: float = HYBRID_GUARD_MARGIN_CP,
        hybrid_guard_quiet_margin_cp: float = HYBRID_GUARD_QUIET_MARGIN_CP,
        hybrid_guard_nn_tolerance_cp: float = HYBRID_GUARD_NN_TOLERANCE_CP,
        endgame_specialist_enabled: bool = ENDGAME_SPECIALIST_ENABLED,
        endgame_specialist_depth: int = ENDGAME_SPECIALIST_DEPTH,
        endgame_specialist_max_time: float = ENDGAME_SPECIALIST_MAX_TIME,
        endgame_specialist_min_adv_cp: float = ENDGAME_SPECIALIST_MIN_ADV_CP,
        obvious_move_fastpath_enabled: bool = OBVIOUS_MOVE_FASTPATH_ENABLED,
        obvious_move_scratch_depth: int = OBVIOUS_MOVE_SCRATCH_DEPTH,
        obvious_move_margin_cp: float = OBVIOUS_MOVE_MARGIN_CP,
        reply_audit_enabled: bool = REPLY_AUDIT_ENABLED,
        reply_audit_depth: int = REPLY_AUDIT_DEPTH,
        reply_audit_max_time: float = REPLY_AUDIT_MAX_TIME,
        reply_audit_candidates: int = REPLY_AUDIT_CANDIDATES,
        reply_audit_static_drop_cp: float = REPLY_AUDIT_STATIC_DROP_CP,
        reply_audit_sharp_static_drop_cp: float = REPLY_AUDIT_SHARP_STATIC_DROP_CP,
        reply_audit_override_margin_cp: float = REPLY_AUDIT_OVERRIDE_MARGIN_CP,
        timing_gate_enabled: bool = TIMING_GATE_ENABLED,
        timing_gate_min_depth: int = TIMING_GATE_MIN_DEPTH,
        timing_gate_default_multiplier: float = TIMING_GATE_DEFAULT_MULTIPLIER,
        timing_gate_min_multiplier: float = TIMING_GATE_MIN_MULTIPLIER,
        timing_gate_max_multiplier: float = TIMING_GATE_MAX_MULTIPLIER,
        timing_gate_max_total_seconds: float = TIMING_GATE_MAX_TOTAL_SECONDS,
        timing_gate_max_overrun_ratio: float = TIMING_GATE_MAX_OVERRUN_RATIO,
        timing_gate_safety_seconds: float = TIMING_GATE_SAFETY_SECONDS,
        timing_gate_estimate_margin: float = TIMING_GATE_ESTIMATE_MARGIN,
        timing_gate_min_next_depth_seconds: float = TIMING_GATE_MIN_NEXT_DEPTH_SECONDS,
        timing_gate_lopsided_cp: float = TIMING_GATE_LOPSIDED_CP,
        timing_gate_crushing_cp: float = TIMING_GATE_CRUSHING_CP,
        timing_gate_hopeless_cp: float = TIMING_GATE_HOPELESS_CP,
        timing_gate_hopeless_material_cp: float = TIMING_GATE_HOPELESS_MATERIAL_CP,
        timing_gate_hopeless_own_material_cp: float = TIMING_GATE_HOPELESS_OWN_MATERIAL_CP,
        timing_gate_fast_lost_depth: int = TIMING_GATE_FAST_LOST_DEPTH,
        timing_gate_fast_lost_max_time: float = TIMING_GATE_FAST_LOST_MAX_TIME,
        timing_gate_balanced_cp: float = TIMING_GATE_BALANCED_CP,
        timing_gate_balanced_material_cp: float = TIMING_GATE_BALANCED_MATERIAL_CP,
        timing_gate_sharp_forcing_moves: int = TIMING_GATE_SHARP_FORCING_MOVES,
        timing_gate_sharp_max_total_seconds: float = TIMING_GATE_SHARP_MAX_TOTAL_SECONDS,
        timing_gate_sharp_overrun_ratio: float = TIMING_GATE_SHARP_OVERRUN_RATIO,
        timing_gate_sharp_estimate_margin: float = TIMING_GATE_SHARP_ESTIMATE_MARGIN,
    ):
        self.model = model
        self.device = device
        self.cp_scale = float(cp_scale)
        self.mate_score = float(mate_score)
        self.tt: OrderedDict[tuple[int, int], tuple[int, str, float, list[chess.Move]]] = OrderedDict()
        self.tt_max_items = int(tt_max_items)
        self.eval_cache: OrderedDict[tuple[int, int], tuple[float, np.ndarray]] = OrderedDict()
        self.move_cache: OrderedDict[tuple[int, int], list[chess.Move]] = OrderedDict()
        self.forcing_cache: OrderedDict[tuple[int, int], list[chess.Move]] = OrderedDict()
        self.eval_cache_max_items = int(eval_cache_max_items)
        self.root_top_n = int(root_top_n)
        self.child_top_n = int(child_top_n)
        self.q_depth = int(q_depth)
        self.extension_limit = int(extension_limit)
        self.batched_minimax_enabled = bool(batched_minimax_enabled)
        self.batched_minimax_max_depth = int(batched_minimax_max_depth)
        self.batched_eval_batch_size = int(batched_eval_batch_size)
        self.batched_frontier_q_depth = int(batched_frontier_q_depth)
        self.batched_frontier_q_max_moves = int(batched_frontier_q_max_moves)
        self.batched_frontier_max_nodes = int(batched_frontier_max_nodes)
        self.batched_direct_min_depth = bool(batched_direct_min_depth)
        self.batched_direct_shallow_depths = int(batched_direct_shallow_depths)
        self.draw_contempt_cp = float(draw_contempt_cp)
        self.draw_contempt_trigger_cp = float(draw_contempt_trigger_cp)
        self.conversion_trigger_cp = float(conversion_trigger_cp)
        self.conversion_margin_cp = float(conversion_margin_cp)
        self.conversion_min_material_cp = float(conversion_min_material_cp)
        self.conversion_endgame_min_material_cp = float(conversion_endgame_min_material_cp)
        self.conversion_endgame_max_nonking = int(conversion_endgame_max_nonking)
        self.variant_name = str(variant_name)
        self.hybrid_guard_enabled = bool(hybrid_guard_enabled)
        self.hybrid_guard_depth = int(hybrid_guard_depth)
        self.hybrid_guard_sharp_depth = int(hybrid_guard_sharp_depth)
        self.hybrid_guard_max_time = float(hybrid_guard_max_time)
        self.hybrid_guard_min_time = float(hybrid_guard_min_time)
        self.hybrid_guard_margin_cp = float(hybrid_guard_margin_cp)
        self.hybrid_guard_quiet_margin_cp = float(hybrid_guard_quiet_margin_cp)
        self.hybrid_guard_nn_tolerance_cp = float(hybrid_guard_nn_tolerance_cp)
        self.endgame_specialist_enabled = bool(endgame_specialist_enabled)
        self.endgame_specialist_depth = int(endgame_specialist_depth)
        self.endgame_specialist_max_time = float(endgame_specialist_max_time)
        self.endgame_specialist_min_adv_cp = float(endgame_specialist_min_adv_cp)
        self.obvious_move_fastpath_enabled = bool(obvious_move_fastpath_enabled)
        self.obvious_move_scratch_depth = int(obvious_move_scratch_depth)
        self.obvious_move_margin_cp = float(obvious_move_margin_cp)
        self.reply_audit_enabled = bool(reply_audit_enabled)
        self.reply_audit_depth = int(reply_audit_depth)
        self.reply_audit_max_time = float(reply_audit_max_time)
        self.reply_audit_candidates = int(reply_audit_candidates)
        self.reply_audit_static_drop_cp = float(reply_audit_static_drop_cp)
        self.reply_audit_sharp_static_drop_cp = float(reply_audit_sharp_static_drop_cp)
        self.reply_audit_override_margin_cp = float(reply_audit_override_margin_cp)
        self.timing_gate_enabled = bool(timing_gate_enabled)
        self.timing_gate_min_depth = int(timing_gate_min_depth)
        self.timing_gate_default_multiplier = float(timing_gate_default_multiplier)
        self.timing_gate_min_multiplier = float(timing_gate_min_multiplier)
        self.timing_gate_max_multiplier = float(timing_gate_max_multiplier)
        self.timing_gate_max_total_seconds = float(timing_gate_max_total_seconds)
        self.timing_gate_max_overrun_ratio = float(timing_gate_max_overrun_ratio)
        self.timing_gate_safety_seconds = float(timing_gate_safety_seconds)
        self.timing_gate_estimate_margin = float(timing_gate_estimate_margin)
        self.timing_gate_min_next_depth_seconds = float(timing_gate_min_next_depth_seconds)
        self.timing_gate_lopsided_cp = float(timing_gate_lopsided_cp)
        self.timing_gate_crushing_cp = float(timing_gate_crushing_cp)
        self.timing_gate_hopeless_cp = float(timing_gate_hopeless_cp)
        self.timing_gate_hopeless_material_cp = float(timing_gate_hopeless_material_cp)
        self.timing_gate_hopeless_own_material_cp = float(timing_gate_hopeless_own_material_cp)
        self.timing_gate_fast_lost_depth = int(timing_gate_fast_lost_depth)
        self.timing_gate_fast_lost_max_time = float(timing_gate_fast_lost_max_time)
        self.timing_gate_balanced_cp = float(timing_gate_balanced_cp)
        self.timing_gate_balanced_material_cp = float(timing_gate_balanced_material_cp)
        self.timing_gate_sharp_forcing_moves = int(timing_gate_sharp_forcing_moves)
        self.timing_gate_sharp_max_total_seconds = float(timing_gate_sharp_max_total_seconds)
        self.timing_gate_sharp_overrun_ratio = float(timing_gate_sharp_overrun_ratio)
        self.timing_gate_sharp_estimate_margin = float(timing_gate_sharp_estimate_margin)
        self.nodes = 0
        self.deadline: float | None = None
        self.root_static_value = 0.0
        self.root_turn = chess.WHITE
        self.syzygy = None
        self.syzygy_max_pieces = int(syzygy_max_pieces)
        self._load_syzygy(syzygy_path)

    def _load_syzygy(self, syzygy_path: str | Path | None):
        if not syzygy_path or not ENABLE_SYZYGY:
            return
        path = Path(syzygy_path)
        if not path.exists():
            print("Syzygy path not found:", path)
            return
        try:
            import chess.syzygy

            self.syzygy = chess.syzygy.open_tablebase(str(path))
            print("Syzygy tablebase loaded:", path)
        except Exception as exc:
            print("Could not load Syzygy tablebase:", exc)
            self.syzygy = None

    def close(self):
        if self.syzygy is not None:
            self.syzygy.close()
            self.syzygy = None

    def clear_tt(self):
        self.tt.clear()
        self.move_cache.clear()
        self.forcing_cache.clear()

    def clear_caches(self):
        self.clear_tt()
        self.eval_cache.clear()

    def _position_key(self, board: chess.Board) -> tuple[int, int]:
        return (chess.polyglot.zobrist_hash(board), min(int(board.halfmove_clock), 100))

    def _check_time(self):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise SearchTimeout()

    def _ordered_put(self, cache: OrderedDict, key, value, max_items: int):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > max_items:
            cache.popitem(last=False)

    def _tt_get(self, key):
        entry = self.tt.get(key)
        if entry is not None:
            self.tt.move_to_end(key)
        return entry

    def _tt_put(self, key, entry):
        old = self.tt.get(key)
        if old is not None and entry[0] < old[0]:
            self.tt.move_to_end(key)
            return
        self.tt[key] = entry
        self.tt.move_to_end(key)
        if len(self.tt) > self.tt_max_items:
            self.tt.popitem(last=False)

    def draw_value(self) -> float:
        if self.root_static_value > self.draw_contempt_trigger_cp:
            return -self.draw_contempt_cp
        if self.root_static_value < -self.draw_contempt_trigger_cp:
            return self.draw_contempt_cp
        return 0.0

    def tablebase_value(self, board: chess.Board) -> Optional[float]:
        if self.syzygy is None:
            return None
        if chess.popcount(board.occupied) > self.syzygy_max_pieces:
            return None
        try:
            wdl = self.syzygy.probe_wdl(board)
        except Exception:
            return None
        if wdl == 0:
            return self.draw_value()
        winner = board.turn if wdl > 0 else not board.turn
        sign = 1.0 if winner == chess.WHITE else -1.0
        try:
            dtz = abs(int(self.syzygy.probe_dtz(board)))
        except Exception:
            dtz = 100
        return sign * (self.mate_score * 0.5 - min(dtz, 1000))

    def terminal_value(self, board: chess.Board) -> Optional[float]:
        tb = self.tablebase_value(board)
        if tb is not None:
            return tb

        if board.is_checkmate():
            winner = not board.turn
            score = self.mate_score - min(board.ply(), 10_000)
            return score if winner == chess.WHITE else -score

        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
            or board.is_fivefold_repetition()
            or board.can_claim_fifty_moves()
            or board.is_repetition(3)
        ):
            return self.draw_value()

        if SEARCH_FULL_CLAIM_DRAW_CHECKS and board.can_claim_threefold_repetition():
            return self.draw_value()

        return None

    @torch.inference_mode()
    def evaluate(self, board: chess.Board) -> tuple[float, np.ndarray]:
        self._check_time()
        key = self._position_key(board)
        cached = self.eval_cache.get(key)
        if cached is not None:
            self.eval_cache.move_to_end(key)
            return cached

        x_np = board_to_icpram_bitmap_plus(board)
        x = torch.from_numpy(x_np).unsqueeze(0).to(self.device, dtype=torch.float32, non_blocking=True)
        legal_mask_np = get_legal_mask(board)
        legal_mask = torch.from_numpy(legal_mask_np).unsqueeze(0).to(self.device, dtype=torch.bool, non_blocking=True)
        v_raw, p_logits = self.model.forward_raw(x, legal_mask)
        value_cp = float(v_raw.squeeze().detach().cpu().item()) * self.cp_scale
        logits = p_logits.squeeze(0).detach().float().cpu().numpy()
        self._ordered_put(self.eval_cache, key, (value_cp, logits), self.eval_cache_max_items)
        return value_cp, logits

    @torch.inference_mode()
    def evaluate_many(self, boards: list[chess.Board], batch_size: int = 128) -> list[tuple[float, np.ndarray]]:
        if not boards:
            return []
        results: list[tuple[float, np.ndarray] | None] = [None] * len(boards)
        missing: list[tuple[list[int], chess.Board, tuple[int, int]]] = []
        missing_by_key: dict[tuple[int, int], tuple[list[int], chess.Board, tuple[int, int]]] = {}
        for i, board in enumerate(boards):
            self._check_time()
            terminal = self.terminal_value(board)
            if terminal is not None:
                dummy_logits = np.full(POLICY_DIM, -np.inf, dtype=np.float32)
                results[i] = (float(terminal), dummy_logits)
                continue
            key = self._position_key(board)
            cached = self.eval_cache.get(key)
            if cached is not None:
                self.eval_cache.move_to_end(key)
                results[i] = cached
            else:
                pending = missing_by_key.get(key)
                if pending is None:
                    pending = ([i], board, key)
                    missing_by_key[key] = pending
                    missing.append(pending)
                else:
                    pending[0].append(i)

        for start in range(0, len(missing), batch_size):
            self._check_time()
            chunk = missing[start : start + batch_size]
            xs = np.stack([board_to_icpram_bitmap_plus(board) for _indices, board, _ in chunk], axis=0)
            masks = np.stack([get_legal_mask(board) for _indices, board, _ in chunk], axis=0)
            x = torch.from_numpy(xs).to(self.device, dtype=torch.float32, non_blocking=True)
            legal_mask = torch.from_numpy(masks).to(self.device, dtype=torch.bool, non_blocking=True)
            v_raw, p_logits = self.model.forward_raw(x, legal_mask)
            values = (v_raw.squeeze(-1).detach().float().cpu().numpy() * self.cp_scale).astype(np.float32)
            logits_batch = p_logits.detach().float().cpu().numpy()
            for row, (indices, _board, key) in enumerate(chunk):
                item = (float(values[row]), logits_batch[row])
                self._ordered_put(self.eval_cache, key, item, self.eval_cache_max_items)
                for i in indices:
                    results[i] = item

        return [item for item in results if item is not None]

    def ordered_moves_from_logits(self, board: chess.Board, logits: np.ndarray) -> list[chess.Move]:
        moves = list(board.legal_moves)
        if not moves:
            return []
        scored = []
        for mv in moves:
            idx = move_to_index(mv)
            score = float(logits[idx]) if 0 <= idx < len(logits) and np.isfinite(logits[idx]) else -1e9
            if mv.promotion:
                score += 0.45
            if board.is_capture(mv):
                score += 0.28
            if board.gives_check(mv):
                score += 0.25
            if move_is_dangerous_passer_push(board, mv):
                score += 0.20
            scored.append((score, mv))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [mv for _, mv in scored]

    def ordered_moves(self, board: chess.Board) -> list[chess.Move]:
        key = self._position_key(board)
        cached = self.move_cache.get(key)
        if cached is not None:
            self.move_cache.move_to_end(key)
            return list(cached)

        _, logits = self.evaluate(board)
        ordered = self.ordered_moves_from_logits(board, logits)
        self._ordered_put(self.move_cache, key, ordered, self.eval_cache_max_items)
        return list(ordered)

    def forcing_moves(self, board: chess.Board, ordered: list[chess.Move]) -> list[chess.Move]:
        key = self._position_key(board)
        cached = self.forcing_cache.get(key)
        if cached is not None:
            self.forcing_cache.move_to_end(key)
            return list(cached)
        forcing = []
        for mv in ordered:
            if mv.promotion or board.is_capture(mv) or board.gives_check(mv) or move_is_dangerous_passer_push(board, mv):
                forcing.append(mv)
        self._ordered_put(self.forcing_cache, key, forcing, self.eval_cache_max_items)
        return forcing

    def candidate_moves(self, board: chess.Board, depth: int, root: bool = False, quiescence: bool = False) -> list[chess.Move]:
        ordered = self.ordered_moves(board)
        return self.select_candidate_moves(board, ordered, depth, root=root, quiescence=quiescence)

    def select_candidate_moves(
        self,
        board: chess.Board,
        ordered: list[chess.Move],
        depth: int,
        root: bool = False,
        quiescence: bool = False,
    ) -> list[chess.Move]:
        if not ordered:
            return []
        if quiescence:
            return self.forcing_moves(board, ordered)
        if root:
            cap = self.root_top_n
        else:
            cap = self.child_top_n
        if depth <= 1 and not root:
            cap = max(4, cap - 2)
        selected = list(ordered[: min(cap, len(ordered))])
        seen = set(selected)
        for mv in self.forcing_moves(board, ordered):
            if mv not in seen:
                selected.append(mv)
                seen.add(mv)
        return selected

    def candidate_moves_from_logits(self, board: chess.Board, logits: np.ndarray, depth: int, root: bool = False) -> list[chess.Move]:
        ordered = self.ordered_moves_from_logits(board, logits)
        return self.select_candidate_moves(board, ordered, depth, root=root, quiescence=False)

    def extension_for(self, board: chess.Board, move: chess.Move, extensions_left: int) -> int:
        if extensions_left <= 0:
            return 0
        if move.promotion:
            return 1
        if board.gives_check(move):
            return 1
        if move_is_dangerous_passer_push(board, move):
            return 1
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            mover = board.piece_at(move.from_square)
            if captured is not None and mover is not None:
                if PIECE_VALUE_CP.get(captured.piece_type, 0) >= PIECE_VALUE_CP.get(mover.piece_type, 0):
                    return 1
        return 0

    def quiescence(self, board: chess.Board, alpha: float, beta: float, q_depth: int) -> float:
        self.nodes += 1
        if self.nodes % 128 == 0:
            self._check_time()

        terminal = self.terminal_value(board)
        if terminal is not None:
            return terminal

        stand_pat, _ = self.evaluate(board)
        if q_depth <= 0:
            return stand_pat

        if board.turn == chess.WHITE:
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            best = stand_pat
            for mv in self.candidate_moves(board, 0, quiescence=True):
                board.push(mv)
                try:
                    score = self.quiescence(board, alpha, beta, q_depth - 1)
                finally:
                    board.pop()
                best = max(best, score)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return float(best)
        else:
            if stand_pat <= alpha:
                return stand_pat
            beta = min(beta, stand_pat)
            best = stand_pat
            for mv in self.candidate_moves(board, 0, quiescence=True):
                board.push(mv)
                try:
                    score = self.quiescence(board, alpha, beta, q_depth - 1)
                finally:
                    board.pop()
                best = min(best, score)
                beta = min(beta, best)
                if beta <= alpha:
                    break
            return float(best)

    def alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float, extensions_left: int) -> float:
        self.nodes += 1
        if self.nodes % 128 == 0:
            self._check_time()

        terminal = self.terminal_value(board)
        if terminal is not None:
            return terminal
        if depth <= 0:
            return self.quiescence(board, alpha, beta, self.q_depth)

        key = self._position_key(board)
        entry = self._tt_get(key)
        moves = None
        if entry is not None:
            stored_depth, flag, value, stored_moves = entry
            if stored_depth >= depth:
                if flag == "EXACT":
                    return value
                if flag == "LOWER":
                    alpha = max(alpha, value)
                elif flag == "UPPER":
                    beta = min(beta, value)
                if beta <= alpha:
                    return value
                moves = list(stored_moves)
        if moves is None:
            moves = self.candidate_moves(board, depth, root=False)

        alpha0, beta0 = alpha, beta
        if board.turn == chess.WHITE:
            best = -math.inf
            for mv in moves:
                ext = self.extension_for(board, mv, extensions_left)
                board.push(mv)
                try:
                    score = self.alphabeta(board, depth - 1 + ext, alpha, beta, extensions_left - ext)
                finally:
                    board.pop()
                if score > best:
                    best = score
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
        else:
            best = math.inf
            for mv in moves:
                ext = self.extension_for(board, mv, extensions_left)
                board.push(mv)
                try:
                    score = self.alphabeta(board, depth - 1 + ext, alpha, beta, extensions_left - ext)
                finally:
                    board.pop()
                if score < best:
                    best = score
                beta = min(beta, best)
                if beta <= alpha:
                    break

        flag = "EXACT"
        if best <= alpha0:
            flag = "UPPER"
        elif best >= beta0:
            flag = "LOWER"
        self._tt_put(key, (depth, flag, float(best), moves))
        return float(best)

    def policy_move(self, board: chess.Board) -> Optional[chess.Move]:
        moves = self.ordered_moves(board)
        return moves[0] if moves else None

    def find_mate_in_one(self, board: chess.Board) -> Optional[chess.Move]:
        # Mate beats every eval, every conversion heuristic, every policy prior.
        for mv in board.legal_moves:
            board.push(mv)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return mv
        return None

    def obvious_move_result(self, board: chess.Board, start: float, time_limit: float | None = None) -> Optional[SearchResult]:
        if not self.obvious_move_fastpath_enabled:
            return None
        legal = list(board.legal_moves)
        if len(legal) == 1:
            mv = legal[0]
            board.push(mv)
            try:
                score = self.terminal_value(board)
                if score is None:
                    score, _ = self.evaluate(board)
            finally:
                board.pop()
            return SearchResult(mv, float(score), "obvious-only-move", depth=1, nodes=1, elapsed=time.perf_counter() - start)

        if len(legal) > 10 and not board.is_check():
            return None

        tactical = board.is_check()
        last_to = board.peek().to_square if board.move_stack else None
        recaptures = []
        if last_to is not None:
            target_piece = board.piece_at(last_to)
            if target_piece is not None and target_piece.color != board.turn:
                recaptures = [mv for mv in legal if board.is_capture(mv) and mv.to_square == last_to]
                if target_piece.piece_type in (chess.QUEEN, chess.ROOK):
                    tactical = True

        if not tactical and len(recaptures) != 1:
            return None

        depth = int(self.obvious_move_scratch_depth)
        try:
            scored = classical_root_scores(board, depth=depth, candidate_moves=legal, time_limit=0.35, q_depth=1)
        except ClassicalTimeout:
            return None
        if not scored:
            return None

        best_score, best_move = scored[0]
        if len(scored) == 1:
            margin = float("inf")
        else:
            margin = cp_for_color(board.turn, scored[0][0]) - cp_for_color(board.turn, scored[1][0])

        if board.is_check() and len(legal) <= 2 and margin >= 350:
            return SearchResult(best_move, best_score, "obvious-check", depth=depth, nodes=len(scored), elapsed=time.perf_counter() - start, candidates=scored)

        if recaptures and best_move in recaptures and margin >= self.obvious_move_margin_cp:
            return SearchResult(best_move, best_score, "obvious-recapture", depth=depth, nodes=len(scored), elapsed=time.perf_counter() - start, candidates=scored)

        return None

    def endgame_specialist_result(
        self,
        board: chess.Board,
        start: float,
        time_limit: float | None = None,
        max_depth: int = DEFAULT_MODEL_MAX_DEPTH,
    ) -> Optional[SearchResult]:
        if not self.endgame_specialist_enabled:
            return None
        if has_any_pawns(board):
            return None

        own_kind = mating_material_kind(board, board.turn, self.endgame_specialist_min_adv_cp)
        own_adv = material_cp_for_color(board, board.turn)
        if own_kind is None or own_adv < self.endgame_specialist_min_adv_cp:
            return None

        base_depth = int(self.endgame_specialist_depth)
        if own_kind == "bishop-knight":
            depth = max(base_depth, 12)
        elif own_kind in ("queen", "rook", "major"):
            depth = max(base_depth, 10)
        else:
            depth = max(8, min(base_depth, 10))

        budget = self.endgame_specialist_max_time
        if time_limit and time_limit > 0:
            budget = min(float(budget), max(2.0, float(time_limit)))

        try:
            scored = endgame_root_scores(board, board.turn, own_kind, depth=depth, time_limit=budget)
        except ClassicalTimeout as exc:
            scored = getattr(exc, "partial", [])
        if not scored:
            return None

        best_score, best_move = scored[0]
        return SearchResult(best_move, float(best_score), f"endgame-{own_kind}", depth=depth, nodes=len(scored), elapsed=time.perf_counter() - start, candidates=scored)

    def guard_candidate_moves(self, board: chess.Board, result: SearchResult) -> list[chess.Move]:
        candidates = []
        seen = set()

        def add(mv):
            if mv is not None and mv in board.legal_moves and mv not in seen:
                candidates.append(mv)
                seen.add(mv)

        add(result.move)
        if result.candidates:
            ordered = sorted(result.candidates, key=lambda item: cp_for_color(board.turn, item[0]), reverse=True)
            for _score, mv in ordered[:18]:
                add(mv)
        try:
            for mv in self.ordered_moves(board)[:18]:
                add(mv)
        except Exception:
            pass
        for mv in board.legal_moves:
            if board.is_capture(mv) or board.gives_check(mv) or mv.promotion:
                add(mv)

        if board.legal_moves.count() <= 28 or position_sharpness(board) >= 180:
            for mv in board.legal_moves:
                add(mv)
        return candidates

    def hybrid_guard_result(
        self,
        board: chess.Board,
        result: SearchResult,
        start: float,
        time_limit: float | None = None,
    ) -> SearchResult:
        if not self.hybrid_guard_enabled:
            return result
        if result is None or result.move is None or result.source in ("book", "mate-in-1") or result.source.startswith("endgame"):
            return result

        sharpness = position_sharpness(board)
        sharp = sharpness >= 120 or board.is_check()
        depth = int(self.hybrid_guard_sharp_depth if sharp else self.hybrid_guard_depth)
        budget = float(self.hybrid_guard_max_time)
        if time_limit and time_limit > 0:
            budget = min(budget, max(float(self.hybrid_guard_min_time), float(time_limit) * 0.18))
            if SEARCH_GUARDS_USE_REMAINING_TIME:
                remaining = start + float(time_limit) - time.perf_counter()
                if remaining <= float(self.hybrid_guard_min_time):
                    return result
                budget = min(budget, remaining)

        moves = self.guard_candidate_moves(board, result)
        if not moves:
            return result

        try:
            scratch = classical_root_scores(board, depth=depth, candidate_moves=moves, time_limit=budget, q_depth=2)
        except ClassicalTimeout:
            return result
        if not scratch:
            return result

        scratch_best_score, scratch_best_move = scratch[0]
        scratch_by_move = {mv: score for score, mv in scratch}
        chosen_scratch = scratch_by_move.get(result.move)
        if chosen_scratch is None:
            return result

        best_side = cp_for_color(board.turn, scratch_best_score)
        chosen_side = cp_for_color(board.turn, chosen_scratch)
        scratch_loss = best_side - chosen_side
        margin = float(self.hybrid_guard_margin_cp if sharp else self.hybrid_guard_quiet_margin_cp)
        if scratch_loss < margin:
            return result

        nn_scores = {}
        if result.candidates:
            for score, mv in result.candidates:
                nn_scores[mv] = float(score)
        chosen_nn_side = cp_for_color(board.turn, nn_scores.get(result.move, result.score_cp))
        guard_nn_side = cp_for_color(board.turn, nn_scores.get(scratch_best_move, result.score_cp))
        nn_tolerance = float(self.hybrid_guard_nn_tolerance_cp)

        # Only override when the classical search screams blunder and the NN is
        # not strongly certain that the guard move is positionally awful.
        if guard_nn_side + nn_tolerance < chosen_nn_side and scratch_loss < margin * 1.75:
            return result

        score_cp = nn_scores.get(scratch_best_move, scratch_best_score)
        merged_candidates = list(result.candidates or [])
        seen_moves = {mv for _score, mv in merged_candidates}
        for scratch_score, scratch_move in scratch:
            if scratch_move not in seen_moves:
                merged_candidates.append((float(scratch_score), scratch_move))
                seen_moves.add(scratch_move)

        guarded = SearchResult(
            scratch_best_move,
            float(score_cp),
            f"{result.source}+guard",
            depth=result.depth,
            nodes=result.nodes + len(scratch),
            elapsed=time.perf_counter() - start,
            candidates=merged_candidates,
        )
        return guarded


    def reply_audit_result(
        self,
        board: chess.Board,
        result: SearchResult,
        start: float,
        time_limit: float | None = None,
    ) -> SearchResult:
        """Re-root suspicious candidate moves as opponent-to-move positions.

        This catches the failure mode where a root move looks fine because the
        opponent reply was pruned as a child, but the same position looks bad
        when the opponent is searched as a fresh root.
        """
        if not self.reply_audit_enabled:
            return result
        if not time_limit or time_limit <= 0:
            return result
        if result is None or result.move is None:
            return result
        if result.source in ("book", "mate-in-1") or result.source.startswith("endgame"):
            return result
        if result.depth < 3:
            return result

        root_turn = board.turn
        try:
            root_static_cp = float(self.root_static_value)
            root_static_side = cp_for_color(root_turn, root_static_cp)
            child = board.copy(stack=True)
            child.push(result.move)
            child_static_cp, _ = self.evaluate(child)
            child_static_side = cp_for_color(root_turn, float(child_static_cp))
        except Exception:
            return result

        static_drop = root_static_side - child_static_side
        sharp = position_sharpness(board) >= 120 or board.is_check()
        drop_trigger = self.reply_audit_sharp_static_drop_cp if sharp else self.reply_audit_static_drop_cp

        tied_or_sharp = sharp
        if result.candidates:
            ordered = sorted(result.candidates, key=lambda item: cp_for_color(root_turn, item[0]), reverse=True)
            if ordered:
                best_side = cp_for_color(root_turn, ordered[0][0])
                chosen_side = cp_for_color(root_turn, result.score_cp)
                tied_or_sharp = tied_or_sharp or (best_side - chosen_side <= 55.0)
        if static_drop < drop_trigger and not tied_or_sharp:
            return result

        audit_moves = []
        seen = set()

        def add(mv):
            if mv is not None and mv in board.legal_moves and mv not in seen and len(audit_moves) < max(1, self.reply_audit_candidates):
                audit_moves.append(mv)
                seen.add(mv)

        add(result.move)
        if result.candidates:
            ordered = sorted(result.candidates, key=lambda item: cp_for_color(root_turn, item[0]), reverse=True)
            for _score, mv in ordered:
                add(mv)
        try:
            for mv in self.ordered_moves(board):
                add(mv)
        except Exception:
            pass
        if len(audit_moves) <= 1:
            return result

        old_deadline = self.deadline
        old_root_turn = self.root_turn
        old_root_static = self.root_static_value
        old_nodes = self.nodes
        audit_rows = []
        audit_budget = min(float(self.reply_audit_max_time), max(1.0, float(time_limit) * 0.35))
        if SEARCH_GUARDS_USE_REMAINING_TIME:
            remaining = start + float(time_limit) - time.perf_counter()
            if remaining <= 1.0:
                return result
            audit_budget = min(audit_budget, remaining)
        self.deadline = time.perf_counter() + audit_budget
        audit_depth = max(1, min(int(self.reply_audit_depth), int(result.depth) - 1))

        try:
            for mv in audit_moves:
                if self.deadline is not None and time.perf_counter() >= self.deadline:
                    break
                child = board.copy(stack=True)
                child.push(mv)
                child_static, _ = self.evaluate(child)
                self.root_turn = child.turn
                self.root_static_value = float(child_static)
                try:
                    child_result = self.search_depth_once(child, depth=audit_depth, top_k=1, temperature_cp=0.0, verbose=False)
                    audited_cp = float(child_result.score_cp)
                    reply_san = child.san(child_result.move) if child_result and child_result.move in child.legal_moves else None
                    source = child_result.source if child_result else None
                    nodes = child_result.nodes if child_result else None
                except SearchTimeout:
                    break
                audit_rows.append({
                    "move": mv,
                    "move_san": board.san(mv),
                    "move_uci": mv.uci(),
                    "child_static_cp_white": float(child_static),
                    "child_static_for_root_side": cp_for_color(root_turn, float(child_static)),
                    "audited_cp_white": audited_cp,
                    "audited_for_root_side": cp_for_color(root_turn, audited_cp),
                    "opponent_reply_san": reply_san,
                    "opponent_reply_uci": None if child_result is None else child_result.move.uci(),
                    "reply_source": source,
                    "reply_nodes": nodes,
                })
        finally:
            self.deadline = old_deadline
            self.root_turn = old_root_turn
            self.root_static_value = old_root_static
            self.nodes = max(self.nodes, old_nodes)

        if not audit_rows:
            return result

        audit_rows.sort(key=lambda row: row["audited_for_root_side"], reverse=True)
        chosen_row = next((row for row in audit_rows if row["move"] == result.move), None)
        best_row = audit_rows[0]
        if chosen_row is None:
            return result

        audit_loss = best_row["audited_for_root_side"] - chosen_row["audited_for_root_side"]
        setattr(result, "reply_audit", [{k: v for k, v in row.items() if k != "move"} for row in audit_rows])
        setattr(result, "reply_audit_loss_cp", float(audit_loss))

        if audit_loss < float(self.reply_audit_override_margin_cp):
            return result

        merged_candidates = list(result.candidates or [])
        seen_candidates = {mv for _score, mv in merged_candidates}
        if best_row["move"] not in seen_candidates:
            merged_candidates.append((float(best_row["audited_cp_white"]), best_row["move"]))

        audited = SearchResult(
            best_row["move"],
            float(best_row["audited_cp_white"]),
            f"{result.source}+replyaudit",
            depth=result.depth,
            nodes=max(result.nodes, self.nodes),
            elapsed=time.perf_counter() - start,
            candidates=merged_candidates,
        )
        setattr(audited, "reply_audit", [{k: v for k, v in row.items() if k != "move"} for row in audit_rows])
        setattr(audited, "reply_audit_loss_cp", float(audit_loss))
        return audited


    def conversion_gate_info(self, board: chess.Board, root_turn: chess.Color, side_best: float | None = None) -> dict:
        """Return whether conversion tie-breaks are allowed in this position.

        A search/eval edge can be an attack, initiative, or temporary pressure.
        Conversion should only override raw search once that edge has already
        become a real material advantage on the board.
        """
        material_for_side = float(side_material_cp(board, root_turn))
        queens_on = bool(board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK))
        nonking = int(nonking_piece_count(board))
        side_best_val = None if side_best is None else float(side_best)

        base = {
            "side_best_cp": side_best_val,
            "material_for_side_cp": material_for_side,
            "min_material_cp": float(self.conversion_min_material_cp),
            "endgame_min_material_cp": float(self.conversion_endgame_min_material_cp),
            "endgame_max_nonking": int(self.conversion_endgame_max_nonking),
            "nonking_count": nonking,
            "queens_on": queens_on,
        }

        if side_best_val is not None and side_best_val < float(self.conversion_trigger_cp):
            return {**base, "allowed": False, "reason": "search edge below conversion trigger"}

        if material_for_side >= float(self.conversion_min_material_cp):
            return {**base, "allowed": True, "reason": "material advantage is already converted"}

        if (
            not queens_on
            and nonking <= int(self.conversion_endgame_max_nonking)
            and material_for_side >= float(self.conversion_endgame_min_material_cp)
        ):
            return {**base, "allowed": True, "reason": "low-material queenless endgame with exchange-level material edge"}

        return {**base, "allowed": False, "reason": "eval edge has not materialised into enough material advantage"}

    def conversion_allowed(self, board: chess.Board, root_turn: chess.Color, side_best: float) -> bool:
        return bool(self.conversion_gate_info(board, root_turn, side_best).get("allowed"))

    def conversion_margin_for(self, side_best: float) -> float:
        # Small edge: be very careful. Huge edge: can prefer cleaner wins.
        if side_best < 300:
            return min(self.conversion_margin_cp, 25.0)
        if side_best < 600:
            return min(self.conversion_margin_cp, 50.0)
        if side_best < 1000:
            return min(self.conversion_margin_cp, 80.0)
        return min(max(self.conversion_margin_cp, 120.0), 160.0)

    def conversion_score(self, board: chess.Board, move: chess.Move, searched_score: float, root_turn: chess.Color) -> float:
        side_mult = 1.0 if root_turn == chess.WHITE else -1.0
        side_score = searched_score * side_mult
        before_enemy_passer = passed_pawn_danger(board, not root_turn)
        before_material = material_cp_white(board) * side_mult
        gives_check = board.gives_check(move)

        board.push(move)
        after_enemy_passer = passed_pawn_danger(board, not root_turn)
        after_material = material_cp_white(board) * side_mult
        repeat_penalty = 1.0 if board.is_repetition(2) else 0.0
        opponent_forcing = 0
        for reply in board.legal_moves:
            if board.is_capture(reply) or board.gives_check(reply) or reply.promotion or move_is_dangerous_passer_push(board, reply):
                opponent_forcing += 1
        board.pop()

        material_gain = after_material - before_material
        trade_bonus = nonpawn_trade_bonus(board, move, root_turn, side_score)

        score = 0.0
        score += 0.24 * side_score
        score += trade_bonus
        score += 0.04 * material_gain
        score += 28.0 * (before_enemy_passer - after_enemy_passer)
        score -= 5.0 * opponent_forcing
        score -= 180.0 * repeat_penalty

        # A check is not simplification. Do not let conversion mode prefer random
        # checking moves unless they also score best by search.
        if gives_check and trade_bonus <= 0:
            score -= 25.0
        return float(score)

    def choose_from_scored(
        self,
        board: chess.Board,
        scored: list[tuple[float, chess.Move]],
        top_k: int = 1,
        temperature_cp: float = 0.0,
    ) -> tuple[float, chess.Move, str]:
        root_turn = board.turn
        reverse = root_turn == chess.WHITE
        scored.sort(key=lambda item: item[0], reverse=reverse)
        best_score, best_move = scored[0]

        mate = self.find_mate_in_one(board)
        if mate is not None:
            mate_score = self.mate_score - min(board.ply(), 10_000)
            return (mate_score if root_turn == chess.WHITE else -mate_score), mate, "mate-in-1"

        def side_score(score: float) -> float:
            return score if root_turn == chess.WHITE else -score

        side_best = side_score(best_score)
        use_conversion = self.conversion_allowed(board, root_turn, side_best)
        if use_conversion:
            margin = self.conversion_margin_for(side_best)
            candidates = []
            for score, mv in scored:
                if side_best - side_score(score) <= margin:
                    candidates.append((score, mv))
            if len(candidates) > 1:
                best_conv = max(
                    candidates,
                    key=lambda item: self.conversion_score(board, item[1], item[0], root_turn),
                )
                if best_conv[1] != best_move:
                    return best_conv[0], best_conv[1], "nn+conversion"

        if top_k and top_k > 1 and temperature_cp and temperature_cp > 0:
            top = scored[: min(int(top_k), len(scored))]
            side_scores = np.array([side_score(s) for s, _ in top], dtype=np.float64)
            side_scores = side_scores - side_scores.max()
            weights = np.exp(side_scores / max(float(temperature_cp), 1e-6))
            weights = weights / weights.sum()
            idx = int(np.random.choice(len(top), p=weights))
            return top[idx][0], top[idx][1], "nn+sample"

        return best_score, best_move, "nn"

    def board_after(self, board: chess.Board, move: chess.Move) -> chess.Board:
        child = board.copy(stack=True)
        child.push(move)
        return child

    def _cache_ordered_moves_from_logits(self, board: chess.Board, logits: np.ndarray) -> list[chess.Move]:
        ordered = self.ordered_moves_from_logits(board, logits)
        self._ordered_put(self.move_cache, self._position_key(board), ordered, self.eval_cache_max_items)
        return ordered

    def _batched_frontier_values(
        self,
        boards: list[chess.Board],
        evals: list[tuple[float, np.ndarray]],
        q_depth: int,
    ) -> list[float]:
        values = [float(value) for value, _logits in evals]
        if q_depth <= 0 or not boards:
            return values

        forcing_boards: list[chess.Board] = []
        forcing_owner: list[int] = []
        max_forcing = max(0, int(self.batched_frontier_q_max_moves))
        for i, (leaf, (_value, logits)) in enumerate(zip(boards, evals)):
            terminal = self.terminal_value(leaf)
            if terminal is not None:
                values[i] = float(terminal)
                continue
            ordered = self._cache_ordered_moves_from_logits(leaf, logits)
            forcing = self.forcing_moves(leaf, ordered)
            if max_forcing > 0:
                forcing = forcing[:max_forcing]
            for mv in forcing:
                forcing_boards.append(self.board_after(leaf, mv))
                forcing_owner.append(i)

        if not forcing_boards:
            return values

        child_evals = self.evaluate_many(forcing_boards, batch_size=self.batched_eval_batch_size)
        self.nodes += len(forcing_boards)
        grouped: dict[int, list[float]] = defaultdict(list)
        for owner, (child_value, _child_logits) in zip(forcing_owner, child_evals):
            grouped[owner].append(float(child_value))

        for owner, child_values in grouped.items():
            if boards[owner].turn == chess.WHITE:
                values[owner] = max(values[owner], max(child_values))
            else:
                values[owner] = min(values[owner], min(child_values))
        return values

    def search_depth_frontier_batched_once(
        self,
        board: chess.Board,
        depth: int,
        top_k: int = 1,
        temperature_cp: float = 0.0,
        verbose: bool = False,
    ) -> SearchResult:
        root_turn = board.turn
        nodes: list[dict] = [{"board": board, "move": None, "parent": None, "children": [], "value": None, "remaining": int(depth), "extensions_left": self.extension_limit}]
        frontier = [0]
        max_nodes = max(0, int(self.batched_frontier_max_nodes))

        while frontier:
            self._check_time()
            expand_indices = []
            for idx in frontier:
                node = nodes[idx]
                if node["value"] is not None or int(node["remaining"]) <= 0:
                    continue
                terminal = self.terminal_value(node["board"])
                if terminal is not None:
                    node["value"] = float(terminal)
                else:
                    expand_indices.append(idx)
            if not expand_indices:
                break

            expand_boards = [nodes[idx]["board"] for idx in expand_indices]
            expand_evals = self.evaluate_many(expand_boards, batch_size=self.batched_eval_batch_size)
            self.nodes += len(expand_boards)
            next_frontier: list[int] = []

            for idx, (static_value, logits) in zip(expand_indices, expand_evals):
                node = nodes[idx]
                node_board = node["board"]
                remaining = int(node["remaining"])
                ordered = self._cache_ordered_moves_from_logits(node_board, logits)
                moves = self.select_candidate_moves(node_board, ordered, remaining, root=(idx == 0), quiescence=False)
                if not moves:
                    node["value"] = float(static_value)
                    continue

                for mv in moves:
                    ext = self.extension_for(node_board, mv, int(node["extensions_left"]))
                    child = self.board_after(node_board, mv)
                    child_idx = len(nodes)
                    nodes.append({
                        "board": child,
                        "move": mv,
                        "parent": idx,
                        "children": [],
                        "value": None,
                        "remaining": remaining - 1 + ext,
                        "extensions_left": max(0, int(node["extensions_left"]) - ext),
                    })
                    node["children"].append(child_idx)
                    if remaining - 1 + ext > 0:
                        next_frontier.append(child_idx)
                    if max_nodes and len(nodes) > max_nodes:
                        raise BatchedSearchTooLarge(f"batched frontier exceeded {max_nodes} nodes")
            frontier = next_frontier

        leaf_indices = [idx for idx, node in enumerate(nodes) if not node["children"] and node["value"] is None]
        if leaf_indices:
            leaf_boards = [nodes[idx]["board"] for idx in leaf_indices]
            leaf_evals = self.evaluate_many(leaf_boards, batch_size=self.batched_eval_batch_size)
            self.nodes += len(leaf_boards)
            q_depth = max(0, min(int(self.batched_frontier_q_depth), int(self.q_depth), 1))
            leaf_values = self._batched_frontier_values(leaf_boards, leaf_evals, q_depth=q_depth)
            for idx, value in zip(leaf_indices, leaf_values):
                nodes[idx]["value"] = float(value)

        for idx in range(len(nodes) - 1, -1, -1):
            node = nodes[idx]
            if node["children"]:
                child_values = [float(nodes[child_idx]["value"]) for child_idx in node["children"] if nodes[child_idx]["value"] is not None]
                if not child_values:
                    node["value"], _ = self.evaluate(node["board"])
                elif node["board"].turn == chess.WHITE:
                    node["value"] = max(child_values)
                else:
                    node["value"] = min(child_values)

        root_children = nodes[0]["children"]
        if not root_children:
            raise ValueError("No legal moves to search.")
        scored = [(float(nodes[child_idx]["value"]), nodes[child_idx]["move"]) for child_idx in root_children]

        chosen_score, chosen_move, source = self.choose_from_scored(
            board,
            scored,
            top_k=top_k,
            temperature_cp=temperature_cp,
        )

        if verbose:
            reverse = root_turn == chess.WHITE
            shown = sorted(scored, key=lambda item: item[0], reverse=reverse)
            print(f"frontier-batched depth={depth}, tree_nodes={len(nodes)}, nodes={self.nodes}, tt={len(self.tt)}, eval_cache={len(self.eval_cache)}")
            for rank, (score, mv) in enumerate(shown[:8], start=1):
                mark = "*" if mv == chosen_move else " "
                print(f"{rank:>2}{mark} {board.san(mv):>8} {score:+.1f} cp")
            if source != "nn":
                print("choice mode:", source)

        return SearchResult(chosen_move, float(chosen_score), source, depth=depth, nodes=self.nodes, candidates=scored)

    def search_depth_batched_once(
        self,
        board: chess.Board,
        depth: int,
        top_k: int = 1,
        temperature_cp: float = 0.0,
        verbose: bool = False,
    ) -> SearchResult:
        root_turn = board.turn
        moves = self.candidate_moves(board, depth, root=True)
        if not moves:
            raise ValueError("No legal moves to search.")

        child_boards = [self.board_after(board, mv) for mv in moves]
        child_evals = self.evaluate_many(child_boards)
        self.nodes += len(child_boards)

        if depth <= 1:
            scored = [(child_evals[i][0], mv) for i, mv in enumerate(moves)]
        else:
            leaf_boards: list[chess.Board] = []
            leaf_owner: list[int] = []
            scored: list[tuple[float, chess.Move] | None] = [None] * len(moves)
            for i, child in enumerate(child_boards):
                terminal = self.terminal_value(child)
                if terminal is not None:
                    scored[i] = (float(terminal), moves[i])
                    continue
                replies = self.candidate_moves_from_logits(child, child_evals[i][1], depth - 1, root=False)
                if not replies:
                    scored[i] = (child_evals[i][0], moves[i])
                    continue
                for reply in replies:
                    leaf_boards.append(self.board_after(child, reply))
                    leaf_owner.append(i)

            leaf_evals = self.evaluate_many(leaf_boards)
            self.nodes += len(leaf_boards)
            grouped: dict[int, list[float]] = defaultdict(list)
            for owner, (value, _logits) in zip(leaf_owner, leaf_evals):
                grouped[owner].append(float(value))

            for i, mv in enumerate(moves):
                if scored[i] is not None:
                    continue
                values = grouped.get(i, [])
                if not values:
                    scored[i] = (child_evals[i][0], mv)
                elif child_boards[i].turn == chess.WHITE:
                    scored[i] = (max(values), mv)
                else:
                    scored[i] = (min(values), mv)

            scored = [item for item in scored if item is not None]

        chosen_score, chosen_move, source = self.choose_from_scored(
            board,
            scored,
            top_k=top_k,
            temperature_cp=temperature_cp,
        )

        if verbose:
            reverse = root_turn == chess.WHITE
            shown = sorted(scored, key=lambda item: item[0], reverse=reverse)
            print(f"batched depth={depth}, nodes={self.nodes}, tt={len(self.tt)}, eval_cache={len(self.eval_cache)}")
            for rank, (score, mv) in enumerate(shown[:8], start=1):
                mark = "*" if mv == chosen_move else " "
                print(f"{rank:>2}{mark} {board.san(mv):>8} {score:+.1f} cp")
            if source != "nn":
                print("choice mode:", source)

        return SearchResult(chosen_move, float(chosen_score), source, depth=depth, nodes=self.nodes, candidates=scored)

    def search_depth_once(
        self,
        board: chess.Board,
        depth: int,
        top_k: int = 1,
        temperature_cp: float = 0.0,
        verbose: bool = False,
    ) -> SearchResult:
        if self.batched_minimax_enabled and depth <= self.batched_minimax_max_depth:
            try:
                return self.search_depth_frontier_batched_once(
                    board,
                    depth=depth,
                    top_k=top_k,
                    temperature_cp=temperature_cp,
                    verbose=verbose,
                )
            except BatchedSearchTooLarge:
                pass

        if depth <= 2:
            return self.search_depth_batched_once(
                board,
                depth=depth,
                top_k=top_k,
                temperature_cp=temperature_cp,
                verbose=verbose,
            )

        root_turn = board.turn
        moves = self.candidate_moves(board, depth, root=True)
        if not moves:
            raise ValueError("No legal moves to search.")

        scored: list[tuple[float, chess.Move]] = []
        alpha, beta = -math.inf, math.inf
        for mv in moves:
            self._check_time()
            ext = self.extension_for(board, mv, self.extension_limit)
            board.push(mv)
            try:
                score = self.alphabeta(board, depth - 1 + ext, alpha, beta, self.extension_limit - ext)
            finally:
                board.pop()
            scored.append((score, mv))
            if root_turn == chess.WHITE:
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)

        chosen_score, chosen_move, source = self.choose_from_scored(
            board,
            scored,
            top_k=top_k,
            temperature_cp=temperature_cp,
        )

        if verbose:
            reverse = root_turn == chess.WHITE
            shown = sorted(scored, key=lambda item: item[0], reverse=reverse)
            print(f"searched depth={depth}, nodes={self.nodes}, tt={len(self.tt)}, eval_cache={len(self.eval_cache)}")
            for rank, (score, mv) in enumerate(shown[:8], start=1):
                mark = "*" if mv == chosen_move else " "
                print(f"{rank:>2}{mark} {board.san(mv):>8} {score:+.1f} cp")
            if source != "nn":
                print("choice mode:", source)

        return SearchResult(chosen_move, float(chosen_score), source, depth=depth, nodes=self.nodes, candidates=scored)


    def _timing_material_context(self, board: chess.Board) -> dict:
        own_material = 0
        enemy_material = 0
        own_nonking = 0
        enemy_nonking = 0
        own_pawns = 0
        enemy_pawns = 0
        for piece in board.piece_map().values():
            if piece.piece_type == chess.KING:
                continue
            value = PIECE_VALUE_CP.get(piece.piece_type, 0)
            if piece.color == board.turn:
                own_material += value
                own_nonking += 1
                own_pawns += int(piece.piece_type == chess.PAWN)
            else:
                enemy_material += value
                enemy_nonking += 1
                enemy_pawns += int(piece.piece_type == chess.PAWN)
        return {
            "own_material_cp": int(own_material),
            "enemy_material_cp": int(enemy_material),
            "material_for_side_cp": int(own_material - enemy_material),
            "own_nonking_count": int(own_nonking),
            "enemy_nonking_count": int(enemy_nonking),
            "own_pawns": int(own_pawns),
            "enemy_pawns": int(enemy_pawns),
        }

    def _timing_forcing_count(self, board: chess.Board) -> int:
        count = 0
        for mv in board.legal_moves:
            if board.is_capture(mv) or board.gives_check(mv) or mv.promotion or move_is_dangerous_passer_push(board, mv):
                count += 1
        return int(count)

    def _timing_position_policy(self, board: chess.Board, root_cp_white: float) -> dict:
        side_static = cp_for_color(board.turn, root_cp_white)
        material = self._timing_material_context(board)
        material_side = float(material["material_for_side_cp"])
        forcing_count = self._timing_forcing_count(board)
        legal_count = board.legal_moves.count()

        hopeless_material = (
            material_side <= -float(self.timing_gate_hopeless_material_cp)
            or (
                material["own_material_cp"] <= float(self.timing_gate_hopeless_own_material_cp)
                and material["enemy_material_cp"] >= 500
            )
        )
        hopeless = side_static <= -float(self.timing_gate_hopeless_cp) or hopeless_material
        lopsided = abs(side_static) >= float(self.timing_gate_lopsided_cp) or abs(material_side) >= float(self.timing_gate_lopsided_cp)
        crushing = abs(side_static) >= float(self.timing_gate_crushing_cp) or abs(material_side) >= float(self.timing_gate_crushing_cp)
        balanced = abs(side_static) <= float(self.timing_gate_balanced_cp) and abs(material_side) <= float(self.timing_gate_balanced_material_cp)
        sharp = board.is_check() or forcing_count >= int(self.timing_gate_sharp_forcing_moves)

        mode = "normal"
        skip_optional = False
        fast_depth = None
        fast_time = None
        max_total_seconds = None
        estimate_margin = float(self.timing_gate_estimate_margin)
        reason = "normal position"

        if hopeless:
            mode = "hopeless_fast"
            skip_optional = True
            fast_depth = max(1, int(self.timing_gate_fast_lost_depth))
            fast_time = max(0.25, float(self.timing_gate_fast_lost_max_time))
            reason = "hopeless by material/eval; do not make the opponent wait"
        elif lopsided:
            mode = "crushing_lopsided" if crushing else "lopsided"
            skip_optional = True
            reason = "position is already lopsided; optional deeper search is low value"
        elif balanced and sharp:
            mode = "balanced_sharp"
            max_total_seconds = float(self.timing_gate_sharp_max_total_seconds)
            estimate_margin = float(self.timing_gate_sharp_estimate_margin)
            reason = "balanced and forcing; spending extra time can matter"

        policy = {
            "mode": mode,
            "reason": reason,
            "side_static_cp": float(side_static),
            "material_for_side_cp": float(material_side),
            "forcing_moves": int(forcing_count),
            "legal_moves": int(legal_count),
            "skip_optional_depths": bool(skip_optional),
            "fast_depth": fast_depth,
            "fast_time_seconds": fast_time,
            "max_total_seconds": max_total_seconds,
            "estimate_margin": float(estimate_margin),
        }
        policy.update(material)
        return policy

    def _timing_gate_allowed_total(self, time_limit: float | None, timing_policy: dict | None = None) -> float:
        target = float(time_limit) if time_limit and time_limit > 0 else 30.0
        target = max(0.1, target)
        timing_policy = timing_policy or {}
        if timing_policy.get("mode") == "balanced_sharp":
            overrun_cap = target * max(1.0, float(self.timing_gate_sharp_overrun_ratio))
            max_total = float(timing_policy.get("max_total_seconds") or self.timing_gate_sharp_max_total_seconds)
            return max(target, min(max_total, overrun_cap))
        overrun_cap = target * max(1.0, float(self.timing_gate_max_overrun_ratio))
        if target >= 20.0:
            return max(target, min(float(self.timing_gate_max_total_seconds), overrun_cap))
        # For short games, do not suddenly turn a 5s move into a 55s move.
        return max(target, overrun_cap + float(self.timing_gate_safety_seconds))

    def _timing_gate_estimate_next(self, depth_timings: list[dict]) -> tuple[float, float] | tuple[None, None]:
        if not depth_timings:
            return None, None
        last = depth_timings[-1]
        last_seconds = max(float(self.timing_gate_min_next_depth_seconds), float(last.get("delta_seconds", 0.0)))
        completed_depth = int(last.get("depth", 0))

        # For the first optional depth after the forced minimum, use the measured d4 time
        # with the laptop-observed d4->d5 multiplier. Earlier depth ratios are often noisier.
        if completed_depth >= int(self.timing_gate_min_depth):
            multiplier = float(self.timing_gate_default_multiplier)
        elif len(depth_timings) >= 2:
            prev_seconds = max(float(self.timing_gate_min_next_depth_seconds), float(depth_timings[-2].get("delta_seconds", 0.0)))
            multiplier = last_seconds / prev_seconds if prev_seconds > 0 else float(self.timing_gate_default_multiplier)
        else:
            multiplier = float(self.timing_gate_default_multiplier)

        multiplier = max(float(self.timing_gate_min_multiplier), min(float(self.timing_gate_max_multiplier), float(multiplier)))
        return last_seconds * multiplier, multiplier

    def _timing_gate_next_decision(
        self,
        depth_timings: list[dict],
        elapsed_so_far: float,
        time_limit: float | None,
        next_depth: int,
        timing_policy: dict | None = None,
    ) -> dict:
        timing_policy = timing_policy or {}
        estimated_next, multiplier = self._timing_gate_estimate_next(depth_timings)
        allowed_total = self._timing_gate_allowed_total(time_limit, timing_policy=timing_policy)
        if estimated_next is None:
            return {"attempt": False, "decision": "skip_next_depth", "reason": "no timing estimate", "timing_policy": timing_policy}

        projected_total = float(elapsed_so_far) + float(estimated_next)
        estimate_margin = float(timing_policy.get("estimate_margin", self.timing_gate_estimate_margin))
        projected_with_margin = float(elapsed_so_far) + float(estimated_next) * estimate_margin + float(self.timing_gate_safety_seconds)
        skip_optional = bool(timing_policy.get("skip_optional_depths"))
        attempt = (not skip_optional) and projected_with_margin <= allowed_total
        reason = timing_policy.get("reason") if skip_optional else ("worth attempting" if attempt else "estimated time too high")
        return {
            "attempt": bool(attempt),
            "decision": "attempt_next_depth" if attempt else "skip_next_depth",
            "reason": str(reason),
            "completed_depth": int(depth_timings[-1].get("depth", 0)),
            "next_depth": int(next_depth),
            "elapsed_so_far_seconds": float(elapsed_so_far),
            "last_depth_seconds": float(depth_timings[-1].get("delta_seconds", 0.0)),
            "estimated_multiplier": float(multiplier),
            "estimated_next_depth_seconds": float(estimated_next),
            "projected_total_seconds": float(projected_total),
            "projected_total_with_margin_seconds": float(projected_with_margin),
            "allowed_total_seconds": float(allowed_total),
            "deadline_total_seconds": float(min(allowed_total, projected_with_margin)),
            "timing_policy": timing_policy,
        }

    def choose_move(
        self,
        board: chess.Board,
        depth: int | None = None,
        time_limit: float | None = DEFAULT_MODEL_TIME,
        max_depth: int = DEFAULT_MODEL_MAX_DEPTH,
        min_depth: int = DEFAULT_MODEL_MIN_DEPTH,
        top_k: int = 1,
        temperature_cp: float = 0.0,
        verbose: bool = False,
    ) -> Optional[SearchResult]:
        if board.is_game_over(claim_draw=False):
            return None
        if depth is not None:
            max_depth = int(depth)
            min_depth = int(depth)
            time_limit = None

        max_depth = int(max_depth)
        min_depth = max(1, int(min_depth))
        if time_limit and time_limit > 0 and depth is None:
            max_depth = max(max_depth, min_depth)

        self.nodes = 0
        self.root_turn = board.turn
        self.deadline = None
        self.root_static_value, _ = self.evaluate(board)
        start = time.perf_counter()
        timing_policy = self._timing_position_policy(board, self.root_static_value) if (self.timing_gate_enabled and depth is None) else {"mode": "disabled"}

        mate = self.find_mate_in_one(board)
        if mate is not None:
            mate_score = self.mate_score - min(board.ply(), 10_000)
            score = mate_score if board.turn == chess.WHITE else -mate_score
            result = SearchResult(mate, float(score), "mate-in-1", depth=1, nodes=1, elapsed=time.perf_counter() - start)
            result.timing_policy = timing_policy
            return result

        obvious = self.obvious_move_result(board, start=start, time_limit=time_limit)
        if obvious is not None:
            obvious.timing_policy = timing_policy
            return obvious

        endgame = self.endgame_specialist_result(board, start=start, time_limit=time_limit, max_depth=max_depth)
        if endgame is not None:
            endgame.timing_policy = timing_policy
            return endgame

        if timing_policy.get("mode") == "hopeless_fast":
            fast_depth = max(1, int(timing_policy.get("fast_depth") or self.timing_gate_fast_lost_depth))
            max_depth = min(max_depth, fast_depth)
            min_depth = min(min_depth, max_depth)
            if time_limit and time_limit > 0:
                time_limit = min(float(time_limit), float(timing_policy.get("fast_time_seconds") or self.timing_gate_fast_lost_max_time))

        fallback = self.policy_move(board)
        if fallback is None:
            return None
        best = SearchResult(fallback, self.root_static_value, "policy", depth=0, nodes=0, elapsed=0.0)
        raw_best_history: list[chess.Move] = []
        depth_timings: list[dict] = []
        last_depth_finished = start
        timing_gate_next_deadline: float | None = None
        if depth is None and self.batched_direct_min_depth and self.batched_minimax_enabled and min_depth > 2:
            shallow = max(0, min(int(self.batched_direct_shallow_depths), min_depth - 1))
            depth_sequence = list(range(1, shallow + 1))
            if min_depth not in depth_sequence:
                depth_sequence.append(min_depth)
            depth_sequence.extend(range(min_depth + 1, int(max_depth) + 1))
        else:
            depth_sequence = list(range(1, int(max_depth) + 1))

        try:
            for d in depth_sequence:
                if time_limit and time_limit > 0:
                    if self.timing_gate_enabled and depth is None and d > min_depth and timing_gate_next_deadline is not None:
                        self.deadline = timing_gate_next_deadline
                    elif STRICT_TIME_LIMIT and d > int(STRICT_TIME_AFTER_DEPTH):
                        self.deadline = start + max(0.01, float(time_limit) - float(STRICT_TIME_SAFETY_SECONDS))
                    elif d > min_depth:
                        self.deadline = start + float(time_limit)
                    else:
                        self.deadline = None
                    if self.deadline is not None and time.perf_counter() >= self.deadline:
                        break
                else:
                    self.deadline = None

                result = self.search_depth_once(
                    board,
                    depth=d,
                    top_k=top_k if d == max_depth else 1,
                    temperature_cp=temperature_cp if d == max_depth else 0.0,
                    verbose=verbose,
                )
                now = time.perf_counter()
                result.elapsed = now - start
                depth_delta = now - last_depth_finished
                last_depth_finished = now
                depth_timings.append({
                    "depth": int(d),
                    "elapsed_seconds": float(result.elapsed),
                    "delta_seconds": float(depth_delta),
                    "nodes": int(self.nodes),
                })
                result.timing_history = list(depth_timings)

                if result.candidates:
                    ordered = sorted(result.candidates, key=lambda item: cp_for_color(board.turn, item[0]), reverse=True)
                    if ordered:
                        raw_best_history.append(ordered[0][1])
                        # If conversion wants to overrule a raw search move that
                        # has stayed best across depths, trust the stable search.
                        if result.source == "nn+conversion" and len(raw_best_history) >= 2:
                            stable_raw = raw_best_history[-1] == raw_best_history[-2]
                            if stable_raw and result.move != raw_best_history[-1]:
                                raw_score = ordered[0][0]
                                result = SearchResult(
                                    raw_best_history[-1],
                                    float(raw_score),
                                    "nn+stable-best",
                                    depth=result.depth,
                                    nodes=result.nodes,
                                    elapsed=result.elapsed,
                                    candidates=result.candidates,
                                )

                best = result
                if self.timing_gate_enabled and depth is None and d >= max(min_depth, int(self.timing_gate_min_depth)) and d < int(max_depth):
                    decision = self._timing_gate_next_decision(
                        depth_timings,
                        elapsed_so_far=result.elapsed,
                        time_limit=time_limit,
                        next_depth=d + 1,
                        timing_policy=timing_policy,
                    )
                    result.timing_gate = decision
                    if not decision.get("attempt", False):
                        if verbose:
                            print(
                                f"timing gate: skipping depth {d + 1}; "
                                f"estimate {decision.get('projected_total_with_margin_seconds', 0.0):.1f}s "
                                f"> allowed {decision.get('allowed_total_seconds', 0.0):.1f}s"
                            )
                        break
                    timing_gate_next_deadline = start + float(decision["deadline_total_seconds"])
                if self.deadline is not None and time.perf_counter() >= self.deadline:
                    break
        except SearchTimeout:
            if self.timing_gate_enabled:
                note = getattr(best, "timing_gate", None)
                if isinstance(note, dict):
                    note["timed_out_after_attempt"] = True
            if verbose:
                print(f"time limit hit; using last complete depth {best.depth}")
        finally:
            self.deadline = None

        best.nodes = self.nodes
        best.elapsed = time.perf_counter() - start
        timing_gate_note = getattr(best, "timing_gate", None)
        timing_history_note = getattr(best, "timing_history", None)
        timing_policy_note = timing_policy
        best = self.hybrid_guard_result(board, best, start=start, time_limit=time_limit)
        best = self.reply_audit_result(board, best, start=start, time_limit=time_limit)
        if timing_gate_note is not None:
            best.timing_gate = timing_gate_note
        if timing_history_note is not None:
            best.timing_history = timing_history_note
        best.timing_policy = timing_policy_note
        best.nodes = max(best.nodes, self.nodes)
        best.elapsed = time.perf_counter() - start
        return best


def create_timing_hybrid_player(model: ChessNet, device: torch.device) -> NeuralChessPlayer:
    return NeuralChessPlayer(
        model,
        device=device,
        cp_scale=VALUE_CP_SCALE,
        tt_max_items=TT_MAX_ITEMS,
        eval_cache_max_items=EVAL_CACHE_MAX_ITEMS,
        root_top_n=SEARCH_ROOT_TOP_N,
        child_top_n=SEARCH_CHILD_TOP_N,
        q_depth=SEARCH_QUIESCENCE_DEPTH,
        extension_limit=SEARCH_EXTENSION_LIMIT,
        batched_minimax_enabled=BATCHED_MINIMAX_ENABLED,
        batched_minimax_max_depth=BATCHED_MINIMAX_MAX_DEPTH,
        batched_eval_batch_size=BATCHED_EVAL_BATCH_SIZE,
        batched_frontier_q_depth=BATCHED_FRONTIER_Q_DEPTH,
        batched_frontier_q_max_moves=BATCHED_FRONTIER_Q_MAX_MOVES,
        batched_frontier_max_nodes=BATCHED_FRONTIER_MAX_NODES,
        batched_direct_min_depth=BATCHED_DIRECT_MIN_DEPTH,
        batched_direct_shallow_depths=BATCHED_DIRECT_SHALLOW_DEPTHS,
        draw_contempt_cp=DRAW_CONTEMPT_CP,
        draw_contempt_trigger_cp=DRAW_CONTEMPT_TRIGGER_CP,
        conversion_trigger_cp=CONVERSION_TRIGGER_CP,
        conversion_margin_cp=CONVERSION_MARGIN_CP,
        conversion_min_material_cp=CONVERSION_MIN_MATERIAL_CP,
        conversion_endgame_min_material_cp=CONVERSION_ENDGAME_MIN_MATERIAL_CP,
        conversion_endgame_max_nonking=CONVERSION_ENDGAME_MAX_NONKING,
        syzygy_path=None,
        syzygy_max_pieces=SYZYGY_MAX_PIECES,
        variant_name="timing_hybrid_aggressive",
        hybrid_guard_enabled=HYBRID_GUARD_ENABLED,
        hybrid_guard_depth=HYBRID_GUARD_DEPTH,
        hybrid_guard_sharp_depth=HYBRID_GUARD_SHARP_DEPTH,
        hybrid_guard_max_time=HYBRID_GUARD_MAX_TIME,
        hybrid_guard_min_time=HYBRID_GUARD_MIN_TIME,
        hybrid_guard_margin_cp=HYBRID_GUARD_MARGIN_CP,
        hybrid_guard_quiet_margin_cp=HYBRID_GUARD_QUIET_MARGIN_CP,
        hybrid_guard_nn_tolerance_cp=HYBRID_GUARD_NN_TOLERANCE_CP,
        endgame_specialist_enabled=ENDGAME_SPECIALIST_ENABLED,
        endgame_specialist_depth=ENDGAME_SPECIALIST_DEPTH,
        endgame_specialist_max_time=ENDGAME_SPECIALIST_MAX_TIME,
        endgame_specialist_min_adv_cp=ENDGAME_SPECIALIST_MIN_ADV_CP,
        obvious_move_fastpath_enabled=OBVIOUS_MOVE_FASTPATH_ENABLED,
        obvious_move_scratch_depth=OBVIOUS_MOVE_SCRATCH_DEPTH,
        obvious_move_margin_cp=OBVIOUS_MOVE_MARGIN_CP,
        reply_audit_enabled=REPLY_AUDIT_ENABLED,
        reply_audit_depth=REPLY_AUDIT_DEPTH,
        reply_audit_max_time=REPLY_AUDIT_MAX_TIME,
        reply_audit_candidates=REPLY_AUDIT_CANDIDATES,
        reply_audit_static_drop_cp=REPLY_AUDIT_STATIC_DROP_CP,
        reply_audit_sharp_static_drop_cp=REPLY_AUDIT_SHARP_STATIC_DROP_CP,
        reply_audit_override_margin_cp=REPLY_AUDIT_OVERRIDE_MARGIN_CP,
        timing_gate_enabled=TIMING_GATE_ENABLED,
        timing_gate_min_depth=TIMING_GATE_MIN_DEPTH,
        timing_gate_default_multiplier=TIMING_GATE_DEFAULT_MULTIPLIER,
        timing_gate_min_multiplier=TIMING_GATE_MIN_MULTIPLIER,
        timing_gate_max_multiplier=TIMING_GATE_MAX_MULTIPLIER,
        timing_gate_max_total_seconds=TIMING_GATE_MAX_TOTAL_SECONDS,
        timing_gate_max_overrun_ratio=TIMING_GATE_MAX_OVERRUN_RATIO,
        timing_gate_safety_seconds=TIMING_GATE_SAFETY_SECONDS,
        timing_gate_estimate_margin=TIMING_GATE_ESTIMATE_MARGIN,
        timing_gate_min_next_depth_seconds=TIMING_GATE_MIN_NEXT_DEPTH_SECONDS,
        timing_gate_lopsided_cp=TIMING_GATE_LOPSIDED_CP,
        timing_gate_crushing_cp=TIMING_GATE_CRUSHING_CP,
        timing_gate_hopeless_cp=TIMING_GATE_HOPELESS_CP,
        timing_gate_hopeless_material_cp=TIMING_GATE_HOPELESS_MATERIAL_CP,
        timing_gate_hopeless_own_material_cp=TIMING_GATE_HOPELESS_OWN_MATERIAL_CP,
        timing_gate_fast_lost_depth=TIMING_GATE_FAST_LOST_DEPTH,
        timing_gate_fast_lost_max_time=TIMING_GATE_FAST_LOST_MAX_TIME,
        timing_gate_balanced_cp=TIMING_GATE_BALANCED_CP,
        timing_gate_balanced_material_cp=TIMING_GATE_BALANCED_MATERIAL_CP,
        timing_gate_sharp_forcing_moves=TIMING_GATE_SHARP_FORCING_MOVES,
        timing_gate_sharp_max_total_seconds=TIMING_GATE_SHARP_MAX_TOTAL_SECONDS,
        timing_gate_sharp_overrun_ratio=TIMING_GATE_SHARP_OVERRUN_RATIO,
        timing_gate_sharp_estimate_margin=TIMING_GATE_SHARP_ESTIMATE_MARGIN,
    )


class TimingHybridEngine:
    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
        opening_path: str | Path = DEFAULT_OPENING_PATH,
        search_seconds: float = DEFAULT_MODEL_TIME,
        min_depth: int = DEFAULT_MODEL_MIN_DEPTH,
        max_depth: int = DEFAULT_MODEL_MAX_DEPTH,
        use_opening_book: bool = True,
        tactical_guard_enabled: bool = False,
        tactical_guard_depth: int = DEFAULT_TACTICAL_GUARD_DEPTH,
        tactical_guard_override_cp: float = DEFAULT_TACTICAL_GUARD_OVERRIDE_CP,
        stockfish_path: str | Path | None = None,
    ):
        self.model, self.device = load_model(weights_path)
        self.player = create_timing_hybrid_player(self.model, self.device)
        self.search_seconds = float(search_seconds)
        self.min_depth = int(min_depth)
        self.max_depth = int(max_depth)
        self.book = OpeningBook(opening_path) if use_opening_book else None
        self.tactical_guard = StockfishTacticalGuard(
            path=stockfish_path or DEFAULT_STOCKFISH_PATH,
            depth=tactical_guard_depth,
            override_cp=tactical_guard_override_cp,
            enabled=tactical_guard_enabled,
        )

    def choose_move(self, board: chess.Board, san_history: list[str]) -> SearchResult:
        result = None
        if self.book is not None and self.book.enabled and board.ply() < OPENING_BOOK_PLIES:
            move = self.book.choose(board, san_history)
            if move is not None:
                result = SearchResult(move, 0.0, "book", depth=0, nodes=0, elapsed=0.0)
        if result is None:
            result = self.player.choose_move(
                board,
                time_limit=self.search_seconds,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
            )
        if result is None or result.move not in board.legal_moves:
            raise RuntimeError(f"Engine returned no legal move for {board.fen()}")

        original_source = result.source
        original_move = result.move
        guard = self.tactical_guard.choose(board, original_move)
        guard["original_source"] = original_source
        result.guard = guard
        result.elapsed += float(guard.get("elapsed_seconds", 0.0) or 0.0)
        selected_uci = guard.get("selected_move", original_move.uci())
        try:
            selected_move = chess.Move.from_uci(selected_uci)
        except (TypeError, ValueError):
            selected_move = original_move
        if guard.get("overrode") and selected_move in board.legal_moves:
            result.move = selected_move
            result.source = f"{original_source}+stockfish-guard"
            score_for_side = float(guard["best_score_for_side_cp"])
            result.score_cp = score_for_side if board.turn == chess.WHITE else -score_for_side
        return result

    def tactical_guard_regression_test(self) -> dict:
        board = chess.Board("6k1/p4pp1/P3p2p/4P1PP/5P2/5b2/1R1R4/r1rBK3 w - - 3 40")
        reproduced_n6_blunder = chess.Move.from_uci("b2b8")
        expected_saving_move = chess.Move.from_uci("g5g6")
        guard = self.tactical_guard.choose(board, reproduced_n6_blunder)
        if not guard.get("available"):
            raise RuntimeError(guard.get("reason") or "Stockfish tactical guard is unavailable")
        selected = chess.Move.from_uci(guard["selected_move"])
        if not guard.get("overrode") or selected != expected_saving_move:
            raise AssertionError(
                "Tactical guard regression failed: expected Rb8+ -> g6, "
                f"got {board.san(reproduced_n6_blunder)} -> {board.san(selected)}"
            )
        return guard

    def smoke_test(self) -> SearchResult:
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        result = self.player.choose_move(board, depth=1)
        if result is None or result.move not in board.legal_moves:
            raise RuntimeError("Engine smoke test failed")
        return result

    def close(self):
        try:
            self.tactical_guard.close()
        finally:
            self.player.close()

