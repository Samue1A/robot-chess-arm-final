"""Record board corners with validation and backup."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import requests


FOLDER = Path(__file__).resolve().parent
SETTINGS_PATH = FOLDER / "settings.json"
OUTPUT_PATH = FOLDER / "board_corners.json"


def send(url: str, command: dict):
    response = requests.get(
        url,
        params={"json": json.dumps(command, separators=(",", ":"))},
        timeout=10,
    )
    response.raise_for_status()
    try:
        return response.json()
    except ValueError:
        return json.loads(response.text.strip())


def read_position(url: str) -> tuple[float, float, float]:
    data = send(url, {"T": 105})
    if isinstance(data, list):
        data = next(
            (
                item
                for item in data
                if isinstance(item, dict) and all(axis in item for axis in ("x", "y", "z"))
            ),
            None,
        )
    if not isinstance(data, dict) or not all(axis in data for axis in ("x", "y", "z")):
        raise RuntimeError(f"Invalid robot feedback: {data!r}")
    point = tuple(float(data[axis]) for axis in ("x", "y", "z"))
    if not all(math.isfinite(value) for value in point):
        raise RuntimeError("Robot returned non-finite coordinates")
    return point


def record_point(url: str, name: str, instruction: str) -> list[float]:
    print(f"\nCALIBRATE {name}: {instruction}")
    send(url, {"T": 210, "cmd": 0})
    input("Move the arm by hand, let it settle, then press Enter...")
    point = read_position(url)
    send(url, {"T": 210, "cmd": 1})
    time.sleep(0.3)
    rounded = [round(value, 6) for value in point]
    print(name, "=", rounded)
    return rounded


def validate_geometry(points: dict):
    distance = lambda first, second: math.dist(points[first], points[second])
    file_near = distance("a1", "h1")
    file_far = distance("a8", "h8")
    rank_left = distance("a1", "a8")
    rank_right = distance("h1", "h8")
    diagonal_one = distance("a1", "h8")
    diagonal_two = distance("h1", "a8")
    print("\nMeasured board geometry (mm):")
    print(f"  a1-h1={file_near:.1f}, a8-h8={file_far:.1f}")
    print(f"  a1-a8={rank_left:.1f}, h1-h8={rank_right:.1f}")
    print(f"  diagonals={diagonal_one:.1f}, {diagonal_two:.1f}")
    warnings = []
    if abs(file_near - file_far) > 15:
        warnings.append("opposite file edges differ by more than 15 mm")
    if abs(rank_left - rank_right) > 15:
        warnings.append("opposite rank edges differ by more than 15 mm")
    if abs(diagonal_one - diagonal_two) > 20:
        warnings.append("board diagonals differ by more than 20 mm")
    if min(file_near, file_far, rank_left, rank_right) < 100:
        warnings.append("one board edge is implausibly short")
    if warnings:
        print("WARNING:")
        for warning in warnings:
            print(" -", warning)
        if input("Save this calibration anyway? [y/N] ").strip().lower() not in {"y", "yes"}:
            raise RuntimeError("Calibration discarded")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate RoArm board corners")
    parser.add_argument("--live", action="store_true", help="Required to communicate with the arm")
    args = parser.parse_args()
    if not args.live:
        print("Calibration was not started. Re-run with --live when the arm area is clear.")
        return 0
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    url = f"http://{settings['robot_ip']}/js"
    print("Connecting to", url)
    read_position(url)
    print("Keep the gripper configuration and jaw angle consistent for all four corners.")
    points = {
        "a1": record_point(url, "a1", "place the jaw reference point on the centre of a1 at board level"),
        "h1": record_point(url, "h1", "place the same reference point on the centre of h1"),
        "a8": record_point(url, "a8", "place the same reference point on the centre of a8"),
        "h8": record_point(url, "h8", "place the same reference point on the centre of h8"),
    }
    validate_geometry(points)
    if OUTPUT_PATH.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = OUTPUT_PATH.with_name(f"board_corners_before_{timestamp}.json")
        shutil.copy2(OUTPUT_PATH, backup)
        print("Backed up previous calibration to", backup.name)
    OUTPUT_PATH.write_text(json.dumps(points, indent=2) + "\n", encoding="utf-8")
    print("Saved", OUTPUT_PATH)
    send(url, {"T": 100})
    print("Returned arm to firmware INIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

