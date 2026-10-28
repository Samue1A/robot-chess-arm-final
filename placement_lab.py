"""Repeatable experiments for tuning RoArm chess-piece placement."""

from __future__ import annotations

import argparse
import json
import shutil
import math
from datetime import datetime
from pathlib import Path

from roarm_controller import PIECES, RoArmPlacementController


FOLDER = Path(__file__).resolve().parent
CORRECTIONS_PATH = FOLDER / "placement_corrections.json"
SETTINGS_PATH = FOLDER / "settings.json"
RESULTS_PATH = FOLDER / "experiment_results.jsonl"


RELEASE_PROFILES = [
    (
        "ten_mm_hover_firm_grip",
        {
            "gripper": {"settle_torque": 180, "release_steps": 3, "release_step_delay_seconds": 0.10},
            "placement": {"hover_height_mm": 10.0, "hover_release_fraction": 0.20, "hover_release_steps": 2},
        },
    ),
    (
        "medium_hover",
        {
            "gripper": {"settle_torque": 120, "release_steps": 4, "release_step_delay_seconds": 0.12},
            "placement": {"hover_height_mm": 14.0, "hover_release_fraction": 0.28, "hover_release_steps": 3},
        },
    ),
    (
        "eighteen_mm_soft_hover",
        {
            "gripper": {"settle_torque": 90, "release_steps": 4, "release_step_delay_seconds": 0.12},
            "placement": {"hover_height_mm": 18.0, "hover_release_fraction": 0.35, "hover_release_steps": 3},
        },
    ),
    (
        "twenty_two_mm_very_soft_hover",
        {
            "gripper": {"settle_torque": 70, "release_steps": 5, "release_step_delay_seconds": 0.15},
            "placement": {"hover_height_mm": 22.0, "hover_release_fraction": 0.45, "hover_release_steps": 4},
        },
    ),
]


def valid_square(value: str) -> str:
    value = value.lower()
    if len(value) != 2 or value[0] not in "abcdefgh" or value[1] not in "12345678":
        raise argparse.ArgumentTypeError("Expected a square such as d5")
    return value


def valid_piece(value: str) -> str:
    value = value.lower()
    if value not in PIECES:
        raise argparse.ArgumentTypeError(f"Expected one of: {', '.join(PIECES)}")
    return value


def save_correction(args):
    data = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8"))
    if args.piece:
        table = data.setdefault("piece_square", {}).setdefault(args.piece, {})
    else:
        table = data.setdefault("squares", {})
    if args.clear:
        table.pop(args.square, None)
        print("Cleared", f"{args.piece} " if args.piece else "", args.square)
    else:
        current = table.setdefault(args.square, {"file_mm": 0.0, "rank_mm": 0.0, "z_mm": 0.0})
        for key in ("file_mm", "rank_mm", "z_mm"):
            value = getattr(args, key)
            if value is not None:
                current[key] = float(value)
        print("Saved correction:", current)
    CORRECTIONS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def show_coordinates(args):
    controller = RoArmPlacementController(live=False)
    raw = controller.square_xyz(args.square, piece=args.piece)
    corrected = controller.square_xyz(args.square, destination=True, piece=args.piece)
    correction = controller._placement_adjustment(args.square, args.piece)
    reach_correction = controller._destination_reach_file_correction_mm(args.square)
    print("Square:", args.square.upper(), "piece:", args.piece)
    print("Raw XYZ:", tuple(round(value, 3) for value in raw))
    print("Corrected XYZ:", tuple(round(value, 3) for value in corrected))
    print("Board-relative correction:", correction)
    print(
        "Reach-profile file correction:",
        f"{reach_correction:+.3f} mm",
        "(positive = toward h-file; negative = toward a-file)",
    )


def _file_direction(value: float) -> str:
    if abs(value) <= 1e-9:
        return "none"
    return "toward h-file" if value > 0.0 else "toward a-file"


def show_landing_profile(args):
    controller = RoArmPlacementController(live=False)
    profile = controller.corrections.get("destination_reach_profile", {})
    samples = profile.get("samples", []) if profile else []
    print("Destination reach calibration (display only; the arm will not move)")
    print("Sign convention: positive command = toward h-file; negative = toward a-file.")
    print("The profile is destination-only and is added after the existing gripper/padding calibration.")
    if not profile or not bool(profile.get("enabled", True)):
        print("Profile: disabled")
    elif not samples:
        print("Profile: enabled, with no samples")
    else:
        print("Saved samples:")
        for sample in samples:
            square = str(sample["square"]).lower()
            squares = float(sample.get("correction_file_squares", 0.0))
            millimetres = squares * controller._file_square_size_mm(square)
            raw = controller._raw_square_xyz(square)
            reach = (raw[0] * raw[0] + raw[1] * raw[1]) ** 0.5
            print(
                f"  {square.upper()}: reach={reach:.1f} mm, "
                f"command={squares:+.3f} square ({millimetres:+.2f} mm, {_file_direction(millimetres)})"
            )
    print("Interpolated additional correction at the centre of each rank:")
    for rank in "12345678":
        square = f"d{rank}"
        millimetres = controller._destination_reach_file_correction_mm(square)
        square_fraction = millimetres / controller._file_square_size_mm(square)
        print(
            f"  rank {rank}: {square_fraction:+.3f} square = {millimetres:+.2f} mm "
            f"({_file_direction(millimetres)})"
        )


def record_landing_error(args):
    """Accumulate the opposite of an observed destination error; never move the arm."""
    if not math.isfinite(args.amount) or args.amount <= 0.0:
        raise ValueError("amount must be greater than zero")
    controller = RoArmPlacementController(live=False)
    local_square_mm = controller._file_square_size_mm(args.square)
    error_squares = args.amount / local_square_mm if args.unit == "mm" else args.amount
    if error_squares > 1.0:
        raise ValueError("Refusing an observed error larger than one full square")

    data = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8"))
    profile = data.setdefault(
        "destination_reach_profile",
        {"enabled": True, "max_abs_file_squares": 1.0, "samples": []},
    )
    profile["enabled"] = True
    samples = profile.setdefault("samples", [])
    max_abs_squares = float(profile.get("max_abs_file_squares", 1.0))
    # A new sample begins at the correction currently interpolated at this reach,
    # so adding feedback does not accidentally discard the existing curve.
    previous = controller._destination_reach_file_correction_mm(args.square) / local_square_mm
    # The command must move opposite the observed residual landing error.
    correction_delta = -error_squares if args.direction == "toward-h" else error_squares
    updated = previous + correction_delta
    if abs(updated) > max_abs_squares:
        raise ValueError(
            f"This would set {args.square.upper()} to {updated:+.3f} square(s), beyond the "
            f"configured +/-{max_abs_squares:g} safety limit"
        )
    samples[:] = [
        item for item in samples if str(item.get("square", "")).lower() != args.square
    ]
    samples.append({"square": args.square, "correction_file_squares": round(updated, 5)})

    # Keep the file easy to inspect: nearest rank/reach samples first.
    samples.sort(
        key=lambda item: (
            int(str(item["square"])[1]),
            str(item["square"])[0],
        )
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = CORRECTIONS_PATH.with_name(f"placement_corrections_before_reach_{timestamp}.json")
    shutil.copy2(CORRECTIONS_PATH, backup)
    CORRECTIONS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    delta_mm = correction_delta * local_square_mm
    updated_mm = updated * local_square_mm
    print("No arm movement was sent.")
    print(
        f"Observed residual at {args.square.upper()}: {error_squares:.3f} square(s) "
        f"{args.direction.replace('-', ' ')}-file."
    )
    print(
        f"Added the opposite command correction: {correction_delta:+.3f} square "
        f"({delta_mm:+.2f} mm, {_file_direction(delta_mm)})."
    )
    print(
        f"New total reach-profile sample at {args.square.upper()}: {updated:+.3f} square "
        f"({updated_mm:+.2f} mm, {_file_direction(updated_mm)})."
    )
    print("Backup:", backup.name)
    print("Run `python placement_lab.py landing-profile` to inspect the full interpolation.")


def nudge_alignment(args):
    if args.amount_mm <= 0:
        raise ValueError("amount-mm must be greater than zero")
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    alignment = settings.setdefault("gripper_alignment", {"radial_mm": 0.0, "sideways_mm": 0.0})
    prefix = "pickup_" if args.pickup_only else ""
    changes = {
        "toward-base": (f"{prefix}radial_extra_mm" if prefix else "radial_mm", -args.amount_mm),
        "away-from-base": (f"{prefix}radial_extra_mm" if prefix else "radial_mm", args.amount_mm),
        "left": (f"{prefix}sideways_extra_mm" if prefix else "sideways_mm", args.amount_mm),
        "right": (f"{prefix}sideways_extra_mm" if prefix else "sideways_mm", -args.amount_mm),
    }
    key, change = changes[args.direction]
    alignment[key] = round(float(alignment.get(key, 0.0)) + change, 3)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print("Saved gripper alignment:", alignment)
    print("Adjusted pickup only." if args.pickup_only else "Adjusted pickup and landing.")
    print("Directions are viewed from the robot base looking outward along the arm.")


def temporary_alignment_overrides(left_delta_mm: float, *, rigid_grip: bool = False) -> dict | None:
    overrides: dict = {}
    if rigid_grip:
        overrides.setdefault("gripper", {})["floating_carry_enabled"] = False
    if left_delta_mm:
        settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        alignment = settings.get("gripper_alignment", {})
        current = float(alignment.get("sideways_mm", 0.0))
        trial = current + float(left_delta_mm)
        overrides.setdefault("gripper_alignment", {})["sideways_mm"] = trial
        print(
            f"TEMPORARY left alignment trial: {current:.3f} -> {trial:.3f} mm "
            "(settings.json is not changed)"
        )
    return overrides or None


def append_result(payload: dict):
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def release_sweep(args):
    print("Release-profile sweep")
    print("Source and destination must be physically clear before every trial.")
    if not args.live:
        print("DRY RUN: showing all four motion profiles without moving hardware.")
    for index, (name, overrides) in enumerate(RELEASE_PROFILES):
        print("\n" + "=" * 60)
        print(f"PROFILE {index + 1}/{len(RELEASE_PROFILES)}: {name}")
        print(json.dumps(overrides, indent=2))
        controller = RoArmPlacementController(live=args.live, overrides=overrides)
        controller.execute(args.source, args.destination, args.piece)
        if args.live:
            while True:
                raw_score = input("Placement score 1 (bad) to 5 (perfect): ").strip()
                if raw_score in {"1", "2", "3", "4", "5"}:
                    break
                print("Enter 1, 2, 3, 4, or 5.")
            note = input("Short note (tilt, offset, wobble, etc.): ").strip()
            append_result(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "test": "release_sweep",
                    "profile": name,
                    "overrides": overrides,
                    "source": args.source,
                    "destination": args.destination,
                    "piece": args.piece,
                    "score": int(raw_score),
                    "note": note,
                }
            )
            if index < len(RELEASE_PROFILES) - 1:
                input("Clear the path, then press Enter to return the piece for the next trial...")
                controller.execute(args.destination, args.source, args.piece)
        elif index < len(RELEASE_PROFILES) - 1:
            print("[DRY] Return motion for the next profile:")
            controller.execute(args.destination, args.source, args.piece)
    if args.live:
        print("Results appended to", RESULTS_PATH)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoArm piece-placement experiment lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Read live robot XYZ, angle, and gripper load")
    status.add_argument("--live", action="store_true")

    init = subparsers.add_parser("init", help="Run the same T:100 INIT command as the web interface")
    init.add_argument("--live", action="store_true")

    open_gripper = subparsers.add_parser(
        "open-gripper",
        help="Open the gripper without moving the arm, for recovery after a stopped run",
    )
    open_gripper.add_argument("--live", action="store_true")

    preview = subparsers.add_parser("preview", help="Show raw and corrected coordinates for a square")
    preview.add_argument("square", type=valid_square)
    preview.add_argument("piece", nargs="?", type=valid_piece, default="pawn")

    profile = subparsers.add_parser(
        "landing-profile",
        help="Show the destination-only reach correction for ranks 1 through 8",
    )

    landing_error = subparsers.add_parser(
        "landing-error",
        help="Record where a placed piece landed; saves the opposite correction without moving the arm",
    )
    landing_error.add_argument("square", type=valid_square)
    landing_error.add_argument(
        "direction",
        choices=("toward-a", "toward-h"),
        help="The board-file direction in which the piece actually missed",
    )
    landing_error.add_argument("amount", type=float)
    landing_error.add_argument(
        "--unit",
        choices=("squares", "mm"),
        default="squares",
        help="Unit for amount (default: squares, so 0.5 means half a square)",
    )

    move = subparsers.add_parser("move", help="Run one square-to-square placement")
    move.add_argument("source", type=valid_square)
    move.add_argument("destination", type=valid_square)
    move.add_argument("piece", nargs="?", type=valid_piece, default="pawn")
    move.add_argument("--live", action="store_true")
    move.add_argument("--yes", action="store_true", help="Run the full live sequence without confirmation prompts")
    move.add_argument(
        "--rigid-grip",
        action="store_true",
        help="Disable the default low-torque floating carry grip",
    )
    move.add_argument(
        "--left-offset-delta-mm",
        type=float,
        default=0.0,
        help="Temporarily add this many mm left for this test only; negative moves right",
    )

    pickup = subparsers.add_parser("pickup-test", help="Pick up and replace one piece to tune the jaws")
    pickup.add_argument("square", type=valid_square)
    pickup.add_argument("piece", nargs="?", type=valid_piece, default="pawn")
    pickup.add_argument("--live", action="store_true")
    pickup.add_argument("--yes", action="store_true", help="Run without confirmation prompts")
    pickup.add_argument(
        "--left-offset-delta-mm",
        type=float,
        default=0.0,
        help="Temporarily add this many mm left for this test only; negative moves right",
    )

    correction = subparsers.add_parser("correction", help="Set a destination landing correction")
    correction.add_argument("square", type=valid_square)
    correction.add_argument("--piece", type=valid_piece)
    correction.add_argument("--file-mm", type=float, help="Positive moves the commanded drop toward the h-file")
    correction.add_argument("--rank-mm", type=float, help="Positive moves the commanded drop toward rank 8")
    correction.add_argument("--z-mm", type=float, help="Positive raises the destination")
    correction.add_argument("--clear", action="store_true")

    nudge = subparsers.add_parser(
        "alignment-nudge",
        help="Fine-tune the calibrated padded-gripper centre",
    )
    nudge.add_argument(
        "direction",
        choices=("toward-base", "away-from-base", "left", "right"),
    )
    nudge.add_argument("amount_mm", type=float)
    nudge.add_argument(
        "--pickup-only",
        action="store_true",
        help="Adjust open-jaw pickup clearance without changing landing coordinates",
    )

    sweep = subparsers.add_parser("release-sweep", help="Compare four hover/release profiles")
    sweep.add_argument("source", type=valid_square)
    sweep.add_argument("destination", type=valid_square)
    sweep.add_argument("piece", nargs="?", type=valid_piece, default="pawn")
    sweep.add_argument("--live", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "status":
        RoArmPlacementController(live=args.live).status()
    elif args.command == "init":
        RoArmPlacementController(live=args.live).move_init()
    elif args.command == "open-gripper":
        RoArmPlacementController(live=args.live).open_gripper_now()
    elif args.command == "preview":
        show_coordinates(args)
    elif args.command == "landing-profile":
        show_landing_profile(args)
    elif args.command == "landing-error":
        record_landing_error(args)
    elif args.command == "move":
        overrides = temporary_alignment_overrides(
            args.left_offset_delta_mm,
            rigid_grip=args.rigid_grip,
        )
        RoArmPlacementController(live=args.live, assume_yes=args.yes, overrides=overrides).execute(
            args.source, args.destination, args.piece
        )
    elif args.command == "pickup-test":
        overrides = temporary_alignment_overrides(args.left_offset_delta_mm)
        RoArmPlacementController(
            live=args.live,
            assume_yes=args.yes,
            overrides=overrides,
        ).pickup_test(args.square, args.piece)
    elif args.command == "correction":
        save_correction(args)
    elif args.command == "alignment-nudge":
        nudge_alignment(args)
    elif args.command == "release-sweep":
        release_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

