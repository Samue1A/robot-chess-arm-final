"""Calibrate the padded gripper's effective centre relative to robot feedback."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
from datetime import datetime
from pathlib import Path

from roarm_controller import PIECES, RoArmPlacementController


FOLDER = Path(__file__).resolve().parent
SETTINGS_PATH = FOLDER / "settings.json"


def valid_square(value: str) -> str:
    value = value.lower()
    if len(value) != 2 or value[0] not in "abcdefgh" or value[1] not in "12345678":
        raise argparse.ArgumentTypeError("Expected a square such as d4")
    return value


def valid_piece(value: str) -> str:
    value = value.lower()
    if value not in PIECES:
        raise argparse.ArgumentTypeError(f"Expected one of: {', '.join(PIECES)}")
    return value


def local_components(
    nominal: tuple[float, float, float],
    measured: tuple[float, float, float],
) -> tuple[float, float]:
    x, y, _ = nominal
    radius = math.hypot(x, y)
    if radius <= 1e-9:
        raise ValueError("Cannot calibrate the gripper at the robot origin")
    radial_x, radial_y = x / radius, y / radius
    sideways_x, sideways_y = -radial_y, radial_x
    delta_x = measured[0] - x
    delta_y = measured[1] - y
    return (
        delta_x * radial_x + delta_y * radial_y,
        delta_x * sideways_x + delta_y * sideways_y,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure the effective centre of asymmetric gripper padding"
    )
    parser.add_argument(
        "squares",
        nargs="*",
        type=valid_square,
        default=["d4", "e4", "d5"],
        help="Clear calibration squares (default: d4 e4 d5)",
    )
    parser.add_argument("--piece", type=valid_piece, default="pawn")
    parser.add_argument("--live", action="store_true", help="Required to read and move the real arm")
    args = parser.parse_args()
    if not args.live:
        print("Nothing moved. Re-run with --live to calibrate the real padded gripper.")
        return 0

    controller = RoArmPlacementController(FOLDER, live=True)
    controller.status()
    height = float(controller.settings["pieces"]["heights_mm"][args.piece])
    below_top = float(controller.settings["pieces"]["grip_below_top_mm"][args.piece])
    grip_height = height - below_top
    samples: list[tuple[float, float]] = []

    print("\nPadded-gripper centre calibration")
    print("Use the same piece and do not move the board during these samples.")
    print("The surrounding squares must be empty.")
    for square in args.squares:
        surface = controller.square_xyz(square, piece=args.piece)
        nominal = (surface[0], surface[1], surface[2] + grip_height)
        print(f"\nPlace the {args.piece} upright in the centre of {square.upper()}.")
        controller.gripper(float(controller.settings["gripper"]["open_angle"]), label="open")
        controller.set_torque_lock(False)
        try:
            input(
                "By hand, position the jaws at normal gripping height with the piece exactly "
                "centred between the padded faces, then press Enter..."
            )
            data = controller.feedback()
            assert data is not None
            measured = data["xyz"]
        finally:
            controller.set_torque_lock(True)
        radial, sideways = local_components(nominal, measured)
        samples.append((radial, sideways))
        print(
            f"  sample {square.upper()}: radial={radial:+.2f} mm, "
            f"sideways={sideways:+.2f} mm"
        )

    radial_values = [sample[0] for sample in samples]
    sideways_values = [sample[1] for sample in samples]
    radial = statistics.median(radial_values)
    sideways = statistics.median(sideways_values)
    spread = max(
        max(radial_values) - min(radial_values),
        max(sideways_values) - min(sideways_values),
    )

    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    alignment = settings.setdefault("gripper_alignment", {})
    alignment["radial_mm"] = round(radial, 3)
    alignment["sideways_mm"] = round(sideways, 3)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = SETTINGS_PATH.with_name(f"settings_before_gripper_alignment_{timestamp}.json")
    shutil.copy2(SETTINGS_PATH, backup)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    print("\nSaved padded-gripper alignment:")
    print(f"  radial_mm={radial:+.3f}")
    print(f"  sideways_mm={sideways:+.3f}")
    print(f"  sample spread={spread:.2f} mm")
    if spread > 4.0:
        print("WARNING: samples varied by more than 4 mm; repeat with more careful centring.")
    print("Backup:", backup.name)
    controller.move_init("Calibration complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
