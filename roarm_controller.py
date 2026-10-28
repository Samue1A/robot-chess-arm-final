"""Safer RoArm-M2-S chess-piece placement primitives.

The arm's clamp configuration has no independent pitch wrist. A piece held
rigidly can therefore arrive tilted at long reach. This controller loosens only
after reaching the destination, lets the suspended piece align under gravity,
and opens fully while its base is still above the board for a short free drop.
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import requests


PIECES = ("pawn", "knight", "bishop", "rook", "queen", "king")


def _deep_update(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 1e-9:
        raise ValueError("Calibration produced a zero-length board axis")
    return tuple(component / length for component in vector)


class RoArmPlacementController:
    def __init__(
        self,
        folder: str | Path | None = None,
        *,
        live: bool = False,
        assume_yes: bool = False,
        overrides: dict | None = None,
    ):
        self.folder = Path(folder or Path(__file__).resolve().parent)
        self.settings_path = self.folder / "settings.json"
        self.calibration_path = self.folder / "board_corners.json"
        self.corrections_path = self.folder / "placement_corrections.json"
        self.live = bool(live)
        self.assume_yes = bool(assume_yes)
        self._overrides = copy.deepcopy(overrides or {})
        self.settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
        if self._overrides:
            _deep_update(self.settings, copy.deepcopy(self._overrides))
        self.calibration = self._load_calibration()
        self.corrections = self._load_corrections()
        self.url = f"http://{self.settings['robot_ip']}/js"
        self.current_grip = float(self.settings["gripper"]["open_angle"])
        self.current_gripper_torque = None
        self._init_reference: tuple[float, float, float] | None = None

    def reload_tuning(self):
        """Reload physical tuning so edits made during a game apply to the next move."""
        self.settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
        if self._overrides:
            _deep_update(self.settings, copy.deepcopy(self._overrides))
        self.corrections = self._load_corrections()
        self.url = f"http://{self.settings['robot_ip']}/js"

    @property
    def dry_run(self) -> bool:
        return not self.live

    def clone_with_overrides(self, overrides: dict) -> "RoArmPlacementController":
        return RoArmPlacementController(
            self.folder,
            live=self.live,
            assume_yes=self.assume_yes,
            overrides=overrides,
        )

    def _load_calibration(self) -> dict:
        if not self.calibration_path.exists():
            raise FileNotFoundError(f"Missing calibration: {self.calibration_path}")
        data = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        for key in ("a1", "h1", "a8", "h8"):
            point = data.get(key)
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError(f"Calibration {key} must be [x, y, z]")
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in point):
                raise ValueError(f"Calibration {key} contains a non-finite value")
        return data

    def _load_corrections(self) -> dict:
        if not self.corrections_path.exists():
            return {"global": {}, "squares": {}, "piece_square": {}}
        return json.loads(self.corrections_path.read_text(encoding="utf-8"))

    def send(self, command: dict):
        if self.settings["debug"].get("show_raw_commands") or self.dry_run:
            prefix = "[DRY] " if self.dry_run else ""
            print(prefix + "> " + json.dumps(command, separators=(",", ":")))
        if self.dry_run:
            return {}
        network = self.settings.get("network", {})
        connect_timeout = float(network.get("connect_timeout_seconds", 2.0))
        read_timeout = float(network.get("read_timeout_seconds", 10.0))
        retry_seconds = float(network.get("retry_seconds", 1.0))
        report_every = max(1, int(network.get("report_every_attempts", 5)))
        retry_forever = bool(network.get("retry_forever", True))
        attempt = 0
        while True:
            try:
                response = requests.get(
                    self.url,
                    params={"json": json.dumps(command, separators=(",", ":"))},
                    timeout=(connect_timeout, read_timeout),
                )
                response.raise_for_status()
                if attempt:
                    print(f"Robot Wi-Fi connection restored after {attempt} failed attempt(s).")
                try:
                    return response.json()
                except ValueError:
                    return json.loads(response.text.strip())
            except (requests.RequestException, ValueError) as exc:
                attempt += 1
                if attempt == 1:
                    print(
                        "Robot communication was interrupted. Reconnect this computer to the "
                        "RoArm network if needed; N10 will keep retrying the current command "
                        "(Ctrl+C to stop)."
                    )
                    print(f"  {type(exc).__name__}: {exc}")
                elif attempt % report_every == 0:
                    print(f"  Still waiting for {self.settings['robot_ip']} ({attempt} attempts)...")
                if not retry_forever:
                    raise
                time.sleep(retry_seconds)

    def feedback(self) -> dict | None:
        if self.dry_run:
            return None
        invalid_attempt = 0
        while True:
            data = self.send({"T": 105})
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except ValueError:
                    data = None
            if isinstance(data, list):
                data = next(
                    (
                        item
                        for item in data
                        if isinstance(item, dict) and all(axis in item for axis in ("x", "y", "z"))
                    ),
                    None,
                )
            if isinstance(data, dict) and all(axis in data for axis in ("x", "y", "z")):
                try:
                    xyz = tuple(float(data[axis]) for axis in ("x", "y", "z"))
                except (TypeError, ValueError):
                    xyz = ()
                if len(xyz) == 3 and all(math.isfinite(value) for value in xyz):
                    data["xyz"] = xyz
                    return data

            invalid_attempt += 1
            if invalid_attempt == 1 or invalid_attempt % 5 == 0:
                print(
                    f"Robot returned incomplete XYZ feedback; retrying "
                    f"({invalid_attempt} attempt(s), Ctrl+C to stop)."
                )
            time.sleep(float(self.settings.get("network", {}).get("retry_seconds", 1.0)))

    def status(self):
        if self.dry_run:
            print("DRY RUN: no network request was made.")
            return
        data = self.feedback()
        assert data is not None
        print("Connected to", self.url)
        print("XYZ:", tuple(round(value, 2) for value in data["xyz"]))
        if "t" in data:
            print("Gripper angle:", round(float(data["t"]), 4))
        if "torH" in data:
            print("Gripper load:", data["torH"])

    def open_gripper_now(self):
        """Open the jaws without moving any arm joint, for recovery after a stopped run."""
        grip = self.settings["gripper"]
        print("Opening gripper without moving the arm")
        self.set_gripper_torque(int(grip["settle_torque"]))
        self.gripper(float(grip["open_angle"]), label="recovery open")

    def set_torque_lock(self, enabled: bool):
        self.send({"T": 210, "cmd": 1 if enabled else 0})
        if self.live:
            time.sleep(0.3)

    def move_init(self, label: str = "Initial position", *, restore_grip: float | None = None):
        """Run the same firmware reset as the web interface's INIT button.

        T:100 resets every joint, including the end joint.  When carrying a
        piece, restore_grip puts the jaw back at the configured safe angle
        after the arm reaches INIT.
        """
        motion = self.settings["motion"]
        pre_init_xyz: tuple[float, float, float] | None = None
        if self.live:
            data = self.feedback()
            assert data is not None
            current_x, current_y, current_z = data["xyz"]
            open_angle = float(self.settings["gripper"]["open_angle"])
            carrying = restore_grip is not None and restore_grip > open_angle + 0.02
            clearance_key = "carrying_clearance_z_mm" if carrying else "init_clearance_z_mm"
            clearance_z = float(motion.get(clearance_key, motion.get("init_clearance_z_mm", 80.0)))
            if current_z < clearance_z:
                self.vertical(
                    current_x,
                    current_y,
                    current_z,
                    clearance_z,
                    f"{label} straight-up clearance before INIT",
                    check_feedback=False,
                    feedback_target=(current_x, current_y, clearance_z),
                )
                data = self.feedback()
                assert data is not None
            pre_init_xyz = data["xyz"]

        print(f"{label}: firmware INIT")
        self.send({"T": 100})
        self.current_grip = 3.14
        if restore_grip is not None:
            self.gripper(
                restore_grip,
                wait_seconds=0.0,
                label="restore after INIT",
            )
        if self.live and pre_init_xyz is not None:
            init_retry_count = 0
            precision_limit = max(0, int(motion.get("precision_retry_limit", 10)))
            while not self._wait_for_init_ready(pre_init_xyz):
                if self._init_reference is not None:
                    print(
                        f"{label}: firmware INIT stalled; moving directly toward the "
                        "learned INIT coordinates instead of repeating the same reset."
                    )
                    self.move_xyz(self._init_reference)
                    data, error, reached = self._wait_for_xyz(
                        self._init_reference,
                        float(motion.get("init_reference_tolerance_mm", 15.0)),
                    )
                    if reached or data["xyz"][2] >= float(motion.get("init_ready_z_mm", 150.0)):
                        print(
                            f"  Cartesian INIT recovery: {error:.1f} mm residual; "
                            "safe high pose reached, continuing."
                        )
                        break
                init_retry_count += 1
                if init_retry_count >= precision_limit:
                    print(
                        f"{label}: INIT precision retry limit ({precision_limit}) reached; "
                        "forcing continuation with the best current pose."
                    )
                    break
                print(f"{label}: INIT made no safe upward progress; reissuing firmware INIT.")
                self.send({"T": 100})
                self.current_grip = 3.14
                if restore_grip is not None:
                    self.gripper(
                        restore_grip,
                        wait_seconds=0.0,
                        label="restore after INIT retry",
                    )

    def _wait_for_init_ready(self, start_xyz: tuple[float, float, float]) -> bool:
        """Wait only for real INIT motion; add no fixed post-arrival dwell."""
        motion = self.settings["motion"]
        poll = max(0.02, float(motion.get("init_feedback_poll_seconds", 0.04)))
        timeout = max(poll, float(motion.get("init_feedback_timeout_seconds", 3.0)))
        ready_z = float(motion.get("init_ready_z_mm", 150.0))
        stable_mm = max(0.05, float(motion.get("init_stable_mm", 0.8)))
        stable_needed = max(1, int(motion.get("init_stable_samples", 2)))
        movement_threshold = max(0.1, float(motion.get("init_motion_threshold_mm", 2.0)))
        reference_tolerance = max(
            1.0,
            float(motion.get("init_reference_tolerance_mm", 15.0)),
        )
        deadline = time.monotonic() + timeout
        previous = start_xyz
        started_at_known_init = (
            self._init_reference is not None
            and math.dist(start_xyz, self._init_reference) <= reference_tolerance
        )
        movement_seen = started_at_known_init or (
            self._init_reference is None and start_xyz[2] >= ready_z
        )
        stable_count = 0

        while time.monotonic() < deadline:
            data = self.feedback()
            assert data is not None
            measured = data["xyz"]
            if math.dist(measured, start_xyz) >= movement_threshold:
                movement_seen = True
            step = math.dist(measured, previous)
            at_init = measured[2] >= ready_z
            if movement_seen and at_init and step <= stable_mm:
                stable_count += 1
                if stable_count >= stable_needed:
                    if self._init_reference is None:
                        self._init_reference = measured
                    reference_error = math.dist(measured, self._init_reference)
                    if reference_error > reference_tolerance:
                        print(
                            f"  INIT is stable and safely high but {reference_error:.1f} mm "
                            "from the unloaded reference; applying one Cartesian correction."
                        )
                        self.move_xyz(self._init_reference)
                        corrected, residual, reached = self._wait_for_xyz(
                            self._init_reference,
                            reference_tolerance,
                        )
                        if reached:
                            measured = corrected["xyz"]
                            print(f"  INIT Cartesian correction reached {residual:.1f} mm residual.")
                        else:
                            print(
                                f"  Loaded INIT retained {residual:.1f} mm residual but is "
                                "stable and safely high; continuing instead of looping."
                            )
                    print(
                        "  INIT ready: "
                        f"{tuple(round(value, 1) for value in measured)}; continuing immediately"
                    )
                    return True
            else:
                stable_count = 0
            previous = measured
            time.sleep(poll)
        return False

    def set_gripper_torque(self, torque: int):
        torque = int(torque)
        if not 0 <= torque <= 1000:
            raise ValueError("Gripper torque must be between 0 and 1000")
        if self.current_gripper_torque == torque:
            return
        print(f"Gripper torque limit: {torque}/1000")
        self.send({"T": 107, "tor": torque})
        self.current_gripper_torque = torque
        if self.live:
            time.sleep(float(self.settings["gripper"].get("torque_wait_seconds", 0.25)))

    def gripper(
        self,
        angle: float,
        *,
        speed: int | None = None,
        acceleration: int | None = None,
        wait_seconds: float | None = None,
        label: str | None = None,
    ):
        angle = float(angle)
        if not 1.08 <= angle <= 3.14:
            raise ValueError(f"Unsafe gripper angle {angle}; expected 1.08 to 3.14 radians")
        grip = self.settings["gripper"]
        speed = int(grip["command_speed"] if speed is None else speed)
        acceleration = int(grip["command_acceleration"] if acceleration is None else acceleration)
        if wait_seconds is None:
            wait_seconds = float(grip["command_wait_seconds"])
        if label:
            print(f"Gripper {label}: {angle:.3f} rad")
        self.current_grip = round(angle, 4)
        self.send({"T": 106, "cmd": self.current_grip, "spd": speed, "acc": acceleration})
        if self.live and wait_seconds > 0:
            time.sleep(wait_seconds)

    def move_xyz(self, target: tuple[float, float, float], *, speed: float | None = None):
        x, y, z = target
        speed = float(self.settings["motion"]["travel_speed"] if speed is None else speed)
        self.send(
            {
                "T": 104,
                "x": round(x, 2),
                "y": round(y, 2),
                "z": round(z, 2),
                "t": self.current_grip,
                "spd": speed,
            }
        )
        if self.live:
            time.sleep(float(self.settings["motion"]["move_wait_seconds"]))

    def coarse_move(
        self,
        target: tuple[float, float, float],
        label: str,
        *,
        tolerance_mm: float,
        speed: float | None = None,
    ) -> tuple[float, float, float]:
        """Move to a broad safe waypoint without iterative millimetre correction."""
        print(f"{label}: target {tuple(round(value, 1) for value in target)}")
        self.move_xyz(target, speed=speed)
        if self.dry_run:
            return target
        data, error, reached = self._wait_for_xyz(target, float(tolerance_mm))
        print(
            f"  coarse arrival={tuple(round(value, 1) for value in data['xyz'])} | "
            f"error={error:.1f} mm"
        )
        if not reached:
            print(
                f"  WARNING: {label} is outside its broad {float(tolerance_mm):.1f} mm "
                "window; the following vertical/precision phase will correct it."
            )
        return data["xyz"]

    def approach_high(
        self,
        desired: tuple[float, float, float],
        label: str,
        *,
        tolerance: float | None = None,
        correct: bool | None = None,
    ) -> tuple[float, float, float]:
        motion = self.settings["motion"]
        tolerance = float(motion["high_tolerance_mm"] if tolerance is None else tolerance)
        if correct is None:
            correct = bool(self.settings["debug"]["correct_high_positions"])
        if self.dry_run:
            commanded = desired
            print(f"{label}: target {tuple(round(value, 1) for value in desired)}")
            self.move_xyz(commanded)
            return commanded

        maximum = int(motion["maximum_high_corrections"])
        precision_limit = max(0, int(motion.get("precision_retry_limit", 10)))
        retry_round = 0
        while True:
            commanded = desired
            suffix = "" if retry_round == 0 else f" (retry {retry_round})"
            print(f"{label}{suffix}: target {tuple(round(value, 1) for value in desired)}")
            self.move_xyz(commanded)
            for attempt in range(maximum + 1):
                data, error, reached = self._wait_for_xyz(desired, tolerance)
                measured = data["xyz"]
                delta = tuple(target - actual for target, actual in zip(desired, measured))
                print(f"  measured={tuple(round(v, 1) for v in measured)} | error={error:.1f} mm")
                if reached:
                    return commanded
                if not correct:
                    raise RuntimeError(f"{label}: still {error:.1f} mm off with correction disabled")
                if attempt == maximum:
                    break
                length = math.sqrt(sum(value * value for value in delta))
                limit = float(motion["maximum_correction_mm"])
                gain = float(motion["correction_gain"])
                scale = gain * min(1.0, limit / max(length, 1e-9))
                commanded = tuple(current + scale * change for current, change in zip(commanded, delta))
                print(f"  correction #{attempt + 1}: {tuple(round(scale * v, 2) for v in delta)} mm")
                self.move_xyz(commanded)

            retry_round += 1
            if retry_round >= precision_limit:
                print(
                    f"  {label}: precision retry limit ({precision_limit}) reached; "
                    f"forcing continuation at {error:.1f} mm residual."
                )
                return commanded
            print(
                f"  {label} remained {error:.1f} mm off. Returning to Init and "
                f"retrying automatically (round {retry_round}; Ctrl+C to stop)."
            )
            held_angle = self.current_grip
            self.set_gripper_torque(int(self.settings["gripper"]["carry_torque"]))
            self.move_init(f"{label} recovery", restore_grip=held_angle)

    def vertical(
        self,
        x: float,
        y: float,
        z_start: float,
        z_end: float,
        label: str,
        *,
        step_mm: float | None = None,
        check_feedback: bool = True,
        tolerance_mm: float | None = None,
        feedback_target: tuple[float, float, float] | None = None,
        adaptive_corrections: bool = True,
    ):
        motion = self.settings["motion"]
        step_mm = float(motion["vertical_step_mm"] if step_mm is None else step_mm)
        command_x, command_y, command_z = x, y, z_end
        count = max(1, math.ceil(abs(z_end - z_start) / step_mm))
        print(f"{label}: Z {z_start:.1f} -> {z_end:.1f} mm ({count} steps)")
        for index in range(1, count + 1):
            z = z_start + (command_z - z_start) * index / count
            self.send(
                {
                    "T": 1041,
                    "x": round(command_x, 2),
                    "y": round(command_y, 2),
                    "z": round(z, 2),
                    "t": self.current_grip,
                }
            )
            if self.live:
                time.sleep(float(motion["vertical_step_delay_seconds"]))
        if self.live:
            time.sleep(float(motion["vertical_settle_seconds"]))
        if self.live:
            # `x/y/z_end` is the compensated command sent to the arm.  When a
            # high-position correction has shifted that command, feedback must
            # still be judged against the intended physical piece location,
            # not against the deliberately offset command.
            target = feedback_target or (x, y, z_end)
            tolerance = float(
                motion["low_tolerance_mm"] if tolerance_mm is None else tolerance_mm
            )
            retries = (
                max(0, int(motion.get("vertical_target_retries", 0)))
                if adaptive_corrections
                else 0
            )
            data = None
            error = float("inf")
            reached = False
            for retry in range(retries + 1):
                data, error, reached = self._wait_for_xyz(target, tolerance)
                if reached:
                    break
                if retry < retries:
                    measured = data["xyz"]
                    delta_x = target[0] - measured[0]
                    delta_y = target[1] - measured[1]
                    delta_z = target[2] - measured[2]
                    xy_length = math.hypot(delta_x, delta_y)
                    xy_limit = float(motion.get("vertical_xy_correction_limit_mm", 5.0))
                    xy_scale = min(1.0, xy_limit / max(xy_length, 1e-9))
                    correction_x = delta_x * xy_scale
                    correction_y = delta_y * xy_scale
                    z_limit = float(motion.get("vertical_z_correction_limit_mm", 20.0))
                    correction_z = max(-z_limit, min(z_limit, delta_z))
                    command_x += correction_x
                    command_y += correction_y
                    command_z += correction_z
                    print(
                        f"  target not settled yet ({error:.1f} mm off); "
                        f"adaptive correction {retry + 1}/{retries}: "
                        f"({correction_x:+.1f}, {correction_y:+.1f}, {correction_z:+.1f}) mm"
                    )
                    self.send(
                        {
                            "T": 1041,
                            "x": round(command_x, 2),
                            "y": round(command_y, 2),
                            "z": round(command_z, 2),
                            "t": self.current_grip,
                        }
                    )
                    time.sleep(float(motion.get("vertical_retry_pause_seconds", 0.15)))

            assert data is not None
            lateral_error = math.hypot(
                data["xyz"][0] - target[0],
                data["xyz"][1] - target[1],
            )
            z_error = data["xyz"][2] - target[2]
            print(
                f"  settled={tuple(round(value, 1) for value in data['xyz'])} | "
                f"XYZ error={error:.1f} mm | lateral={lateral_error:.1f} mm | "
                f"Z error={z_error:+.1f} mm"
            )
            if check_feedback and not reached:
                raise RuntimeError(
                    f"{label}: arm stopped {error:.1f} mm from target after "
                    f"{retries + 1} completion attempt(s)"
                )
            if not check_feedback and not reached:
                print(f"  WARNING: {label} retained {error:.1f} mm residual error; continuing safely.")
        return command_x, command_y, command_z

    def lift_current_xy(
        self,
        target_z: float,
        label: str,
        *,
        fallback_xyz: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Lift at the measured XY before any lateral/Init motion."""
        if self.dry_run:
            current_x, current_y, current_z = fallback_xyz
        else:
            data = self.feedback()
            assert data is not None
            current_x, current_y, current_z = data["xyz"]
        if current_z >= target_z:
            print(f"{label}: already at safe clearance Z {current_z:.1f} mm")
            return current_x, current_y, current_z
        return self.vertical(
            current_x,
            current_y,
            current_z,
            target_z,
            label,
            check_feedback=False,
            feedback_target=(current_x, current_y, target_z),
        )

    def _wait_for_xyz(
        self,
        target: tuple[float, float, float],
        tolerance: float,
    ) -> tuple[dict, float, bool]:
        """Wait adaptively until a commanded target is reached or motion truly settles."""
        motion = self.settings["motion"]
        poll = max(0.03, float(motion.get("feedback_poll_seconds", 0.12)))
        timeout = max(poll, float(motion.get("feedback_timeout_seconds", 4.0)))
        minimum_stable_time = max(
            0.0,
            float(motion.get("feedback_min_stable_seconds", 0.3)),
        )
        stable_mm = max(0.05, float(motion.get("feedback_stable_mm", 0.8)))
        stable_needed = max(1, int(motion.get("feedback_stable_samples", 3)))
        started_at = time.monotonic()
        deadline = started_at + timeout
        previous: tuple[float, float, float] | None = None
        stable_count = 0
        while True:
            data = self.feedback()
            assert data is not None
            measured = data["xyz"]
            error = math.dist(measured, target)
            if error <= tolerance:
                return data, error, True
            if previous is not None and math.dist(measured, previous) <= stable_mm:
                stable_count += 1
                if (
                    stable_count >= stable_needed
                    and time.monotonic() - started_at >= minimum_stable_time
                ):
                    return data, error, False
            else:
                stable_count = 0
            if time.monotonic() >= deadline:
                return data, error, False
            previous = measured
            time.sleep(poll)

    def confirm(self, message: str):
        if self.dry_run:
            print(f"[DRY] confirmation skipped: {message}")
            return
        if self.assume_yes:
            print(f"[AUTO] {message}")
            return
        if input(f"{message} [Enter / q]: ").strip().lower() in {"q", "quit"}:
            raise KeyboardInterrupt

    def _raw_square_xyz(self, square: str) -> tuple[float, float, float]:
        if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
            raise ValueError(f"Invalid square: {square!r}")
        file_index = ord(square[0]) - ord("a")
        rank_index = int(square[1]) - 1
        u, v = file_index / 7, rank_index / 7
        corners = self.calibration
        point = []
        for axis in range(3):
            point.append(
                (1 - u) * (1 - v) * corners["a1"][axis]
                + u * (1 - v) * corners["h1"][axis]
                + (1 - u) * v * corners["a8"][axis]
                + u * v * corners["h8"][axis]
            )
        calibration = self.settings["calibration"]
        x, y, z = point
        x += float(calibration["x_base_mm"]) + float(calibration["x_slope"]) * (
            x - float(calibration["x_reference_mm"])
        )
        y += float(calibration["y_offset_mm"])
        return x, y, z

    def _board_axes(self, square: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        file_index = ord(square[0]) - ord("a")
        rank_index = int(square[1]) - 1
        u, v = file_index / 7, rank_index / 7
        corners = self.calibration
        file_vector = tuple(
            (1 - v) * (corners["h1"][axis] - corners["a1"][axis])
            + v * (corners["h8"][axis] - corners["a8"][axis])
            for axis in range(3)
        )
        rank_vector = tuple(
            (1 - u) * (corners["a8"][axis] - corners["a1"][axis])
            + u * (corners["h8"][axis] - corners["h1"][axis])
            for axis in range(3)
        )
        return _unit(file_vector), _unit(rank_vector)

    def _file_square_size_mm(self, square: str) -> float:
        """Return the calibrated width of one square along this rank."""
        rank_fraction = (int(square[1]) - 1) / 7.0
        corners = self.calibration
        file_vector = tuple(
            (1 - rank_fraction) * (corners["h1"][axis] - corners["a1"][axis])
            + rank_fraction * (corners["h8"][axis] - corners["a8"][axis])
            for axis in range(3)
        )
        return math.sqrt(sum(component * component for component in file_vector)) / 7.0

    def _reach_fraction(self, square: str) -> float:
        """Normalize board-square extension from the near edge to the far edge."""
        point = self._raw_square_xyz(square)
        target_reach = math.hypot(point[0], point[1])
        near_reach = max(
            math.hypot(*self._raw_square_xyz(f"{file}1")[:2])
            for file in "abcdefgh"
        )
        far_reach = max(
            math.hypot(*self._raw_square_xyz(f"{file}{rank}")[:2])
            for file in "abcdefgh"
            for rank in range(1, 9)
        )
        if far_reach <= near_reach:
            return 0.0
        return max(0.0, min(1.0, (target_reach - near_reach) / (far_reach - near_reach)))

    def _destination_reach_file_correction_mm(self, square: str) -> float:
        """Interpolate the calibrated landing correction by physical arm reach.

        Profile samples use board coordinates rather than an observer-dependent
        left/right convention.  A positive sample moves the commanded landing
        toward the h-file; a negative sample moves it toward the a-file.
        """
        profile = self.corrections.get("destination_reach_profile", {})
        if not profile or not bool(profile.get("enabled", True)):
            return 0.0
        raw_samples = profile.get("samples", [])
        if not isinstance(raw_samples, list):
            raise ValueError("destination_reach_profile.samples must be a list")

        max_abs_squares = float(profile.get("max_abs_file_squares", 1.0))
        if not math.isfinite(max_abs_squares) or max_abs_squares <= 0.0:
            raise ValueError("destination_reach_profile.max_abs_file_squares must be positive")

        samples: list[tuple[float, float]] = []
        seen_squares: set[str] = set()
        for sample in raw_samples:
            if not isinstance(sample, dict):
                raise ValueError("Each destination reach sample must be an object")
            sample_square = str(sample.get("square", "")).lower()
            if len(sample_square) != 2 or sample_square[0] not in "abcdefgh" or sample_square[1] not in "12345678":
                raise ValueError(f"Invalid destination reach sample square: {sample_square!r}")
            if sample_square in seen_squares:
                raise ValueError(f"Duplicate destination reach sample for {sample_square}")
            seen_squares.add(sample_square)
            correction_squares = float(sample.get("correction_file_squares", 0.0))
            if not math.isfinite(correction_squares) or abs(correction_squares) > max_abs_squares:
                raise ValueError(
                    f"Reach correction at {sample_square} must be within "
                    f"+/-{max_abs_squares:g} square(s)"
                )
            sample_point = self._raw_square_xyz(sample_square)
            sample_reach = math.hypot(sample_point[0], sample_point[1])
            correction_mm = correction_squares * self._file_square_size_mm(sample_square)
            samples.append((sample_reach, correction_mm))

        if not samples:
            return 0.0
        samples.sort(key=lambda item: item[0])
        target_point = self._raw_square_xyz(square)
        target_reach = math.hypot(target_point[0], target_point[1])
        if target_reach <= samples[0][0]:
            return samples[0][1]
        if target_reach >= samples[-1][0]:
            return samples[-1][1]
        for (near_reach, near_mm), (far_reach, far_mm) in zip(samples, samples[1:]):
            if target_reach <= far_reach:
                reach_span = far_reach - near_reach
                if reach_span <= 1e-9:
                    return (near_mm + far_mm) / 2.0
                fraction = (target_reach - near_reach) / reach_span
                return near_mm + (far_mm - near_mm) * fraction
        return samples[-1][1]

    def _placement_adjustment(self, square: str, piece: str) -> dict[str, float]:
        total = {"file_mm": 0.0, "rank_mm": 0.0, "z_mm": 0.0}
        sources = [
            self.corrections.get("global", {}),
            self.corrections.get("squares", {}).get(square, {}),
            self.corrections.get("piece_square", {}).get(piece, {}).get(square, {}),
        ]
        for source in sources:
            for key in total:
                total[key] += float(source.get(key, 0.0))
        total["file_mm"] += self._destination_reach_file_correction_mm(square)
        return total

    def square_xyz(
        self,
        square: str,
        *,
        destination: bool = False,
        piece: str = "pawn",
    ) -> tuple[float, float, float]:
        point = self._raw_square_xyz(square)
        if not destination:
            return point
        correction = self._placement_adjustment(square, piece)
        file_axis, rank_axis = self._board_axes(square)
        return tuple(
            point[axis]
            + correction["file_mm"] * file_axis[axis]
            + correction["rank_mm"] * rank_axis[axis]
            + (correction["z_mm"] if axis == 2 else 0.0)
            for axis in range(3)
        )

    def _apply_gripper_alignment(
        self,
        point: tuple[float, float, float],
        *,
        square: str,
        piece: str,
        pickup: bool = False,
        destination: bool = False,
    ) -> tuple[float, float, float]:
        """Apply a padding/tool-centre offset in the gripper's rotating local frame."""
        x, y, z = point
        radius = math.hypot(x, y)
        if radius <= 1e-9:
            return point
        alignment = self.settings.get("gripper_alignment", {})
        radial = float(alignment.get("radial_mm", 0.0))
        sideways = float(alignment.get("sideways_mm", 0.0))
        if pickup:
            # The asymmetric foam protrudes farther with the jaws open than it
            # does after it compresses around a carried piece.  Keep this
            # source-only clearance separate so fixing pickup does not move an
            # already calibrated landing point.
            radial += float(alignment.get("pickup_radial_extra_mm", 0.0))
            sideways += float(alignment.get("pickup_sideways_extra_mm", 0.0))
            piece_rank = alignment.get("pickup_radial_extra_by_piece_rank_mm", {})
            radial += float(piece_rank.get(piece, {}).get(square[1], 0.0))
        if destination:
            # A small angular/tool-centre error becomes a larger sideways miss
            # at the far ranks.  Keep this destination-only so correcting the
            # landing point cannot undo the separately calibrated pickup.
            reach_fraction = self._reach_fraction(square)
            near_extra = float(alignment.get("placement_sideways_near_extra_mm", 0.0))
            far_extra = float(alignment.get("placement_sideways_far_extra_mm", near_extra))
            sideways += near_extra + (far_extra - near_extra) * reach_fraction
        radial_x, radial_y = x / radius, y / radius
        sideways_x, sideways_y = -radial_y, radial_x
        return (
            x + radial * radial_x + sideways * sideways_x,
            y + radial * radial_y + sideways * sideways_y,
            z,
        )

    def _release_hover_height_mm(self, source: str, destination: str) -> tuple[float, float]:
        """Return a hover high enough for a tilted piece to pivot before descent.

        The normal hover assumes an upright piece.  The required clearance is
        based on the greatest extension used by the transfer, not destination
        rank: a8->a1 needs the far-arm clearance, while a1->b1 does not.
        """
        placement = self.settings["placement"]
        base_height = float(placement["hover_height_mm"])
        far_clearance = float(placement.get("far_rank_tilt_clearance_mm", 0.0))
        exponent = float(placement.get("tilt_clearance_curve_exponent", 2.0))
        if base_height < 0.0 or far_clearance < 0.0:
            raise ValueError("Placement hover heights must not be negative")
        if not math.isfinite(exponent) or exponent <= 0.0:
            raise ValueError("placement.tilt_clearance_curve_exponent must be positive")
        reach_fraction = max(self._reach_fraction(source), self._reach_fraction(destination))
        tilt_clearance = far_clearance * (reach_fraction**exponent)
        hover_height = base_height + tilt_clearance
        if hover_height > 60.0:
            raise ValueError("Total release hover must not exceed 60 mm")
        return hover_height, tilt_clearance

    def _release_drop_height_mm(self, source: str, destination: str) -> float:
        """Use a small near-arm drop and a slightly larger far-arm drop."""
        placement = self.settings["placement"]
        base_height = float(placement["drop_release_height_mm"])
        far_extra = float(placement.get("far_reach_drop_extra_mm", 0.0))
        exponent = float(placement.get("tilt_clearance_curve_exponent", 2.0))
        reach_fraction = max(self._reach_fraction(source), self._reach_fraction(destination))
        return base_height + far_extra * (reach_fraction**exponent)

    def preview(self, source: str, destination: str, piece: str) -> dict:
        self._validate_piece(piece)
        source_unaligned = self.square_xyz(source, piece=piece)
        destination_unaligned = self.square_xyz(destination, destination=True, piece=piece)
        source_xyz = self._apply_gripper_alignment(
            source_unaligned,
            square=source,
            piece=piece,
            pickup=True,
        )
        destination_xyz = self._apply_gripper_alignment(
            destination_unaligned,
            square=destination,
            piece=piece,
            destination=True,
        )
        height = float(self.settings["pieces"]["heights_mm"][piece])
        below_top = float(self.settings["pieces"]["grip_below_top_mm"][piece])
        grip_height = height - below_top
        placement = self.settings["placement"]
        hover_height, tilt_clearance = self._release_hover_height_mm(source, destination)
        destination_board_z = (
            destination_xyz[2]
            + grip_height
            + float(placement["drop_z_adjust_mm"])
        )
        drop_release_height = self._release_drop_height_mm(source, destination)
        if not 8.0 <= drop_release_height <= 20.0:
            raise ValueError(
                "placement.drop_release_height_mm must stay between 8 and 20 mm "
                "so release remains airborne without becoming a large drop"
            )
        return {
            "source_surface": source_xyz,
            "destination_surface": destination_xyz,
            "source_alignment_delta": tuple(
                aligned - raw for aligned, raw in zip(source_xyz, source_unaligned)
            ),
            "destination_alignment_delta": tuple(
                aligned - raw for aligned, raw in zip(destination_xyz, destination_unaligned)
            ),
            "grip_height_above_surface": grip_height,
            "hover_height_mm": hover_height,
            "tilt_clearance_mm": tilt_clearance,
            "source_grip": (source_xyz[0], source_xyz[1], source_xyz[2] + grip_height),
            "destination_board_grip": (
                destination_xyz[0],
                destination_xyz[1],
                destination_board_z,
            ),
            "destination_release_grip": (
                destination_xyz[0],
                destination_xyz[1],
                destination_board_z + drop_release_height,
            ),
            "destination_grip": (
                destination_xyz[0],
                destination_xyz[1],
                destination_board_z + hover_height,
            ),
            "placement_correction": self._placement_adjustment(destination, piece),
        }

    def _validate_piece(self, piece: str):
        if piece not in PIECES:
            raise ValueError(f"Unknown piece {piece!r}; use one of {', '.join(PIECES)}")

    def _settle_wiggle(self, x: float, y: float, z: float):
        placement = self.settings["placement"]
        amount = float(placement["settle_wiggle_mm"])
        cycles = int(placement["settle_wiggle_cycles"])
        if amount <= 0 or cycles <= 0:
            return
        print(f"Settling wiggle: {amount:.2f} mm x {cycles} cycle(s)")
        pattern = ((amount, 0), (0, amount), (-amount, 0), (0, -amount), (0, 0))
        for _ in range(cycles):
            for dx, dy in pattern:
                self.send(
                    {
                        "T": 1041,
                        "x": round(x + dx, 2),
                        "y": round(y + dy, 2),
                        "z": round(z, 2),
                        "t": self.current_grip,
                    }
                )
                if self.live:
                    time.sleep(0.12)

    def _release_open_angle(self, piece: str) -> float:
        grip = self.settings["gripper"]
        angle = float(grip.get("release_open_angles", {}).get(piece, grip["open_angle"]))
        if not 1.08 <= angle <= 3.14:
            raise ValueError(f"Unsafe {piece} release angle {angle}; expected 1.08 to 3.14 radians")
        return angle

    def _pickup_open_angle(self, piece: str) -> float:
        """Open wider for broad, square pieces before approaching their centre."""
        grip = self.settings["gripper"]
        angle = float(grip.get("pickup_open_angles", {}).get(piece, grip["open_angle"]))
        if not 1.08 <= angle <= 3.14:
            raise ValueError(f"Unsafe {piece} pickup opening angle {angle}; expected 1.08 to 3.14 radians")
        return angle

    def hover_soften(self, piece: str):
        """Gradually loosen at the high hover, then reduce torque for gravity pivoting."""
        grip = self.settings["gripper"]
        placement = self.settings["placement"]
        open_angle = float(grip["open_angle"])
        start_angle = self.current_grip
        configured_target = placement.get("hover_target_angle")
        if configured_target is None:
            fraction = float(placement["hover_release_fraction"])
            if not 0.0 < fraction < 1.0:
                raise ValueError("placement.hover_release_fraction must be between 0 and 1")
            target_angle = start_angle + (open_angle - start_angle) * fraction
        else:
            target_angle = float(configured_target)
        if not open_angle < target_angle < start_angle:
            raise ValueError(
                f"Hover target {target_angle:.3f} must be between open {open_angle:.3f} "
                f"and carried {start_angle:.3f}"
            )
        steps = max(1, int(placement["hover_release_steps"]))
        actuation_torque = int(grip.get("hover_actuation_torque", grip["carry_torque"]))
        self.set_gripper_torque(actuation_torque)
        print(
            f"Hover loosen: {start_angle:.3f} -> {target_angle:.3f} rad "
            f"in {steps} steps at {actuation_torque}/1000; piece remains airborne"
        )
        for step in range(1, steps + 1):
            angle = start_angle + (target_angle - start_angle) * step / steps
            self.gripper(
                angle,
                speed=int(grip["release_speed"]),
                acceleration=int(grip["release_acceleration"]),
                wait_seconds=float(grip["release_step_delay_seconds"]),
                label=f"hover loosen {step}/{steps}",
            )

        if self.live:
            tolerance = float(placement.get("hover_angle_tolerance_rad", 0.04))
            attempts = max(0, int(placement.get("hover_angle_reissue_attempts", 2)))
            for attempt in range(attempts + 1):
                data = self.feedback()
                assert data is not None
                actual_raw = data.get("t")
                if actual_raw is None:
                    print("  Gripper feedback has no angle field; continuing with commanded hover angle.")
                    break
                actual = float(actual_raw)
                error = actual - target_angle
                print(f"  hover grip feedback={actual:.3f} rad | too-closed error={error:+.3f} rad")
                if error <= tolerance:
                    break
                if attempt < attempts:
                    self.gripper(
                        target_angle,
                        wait_seconds=float(grip["release_step_delay_seconds"]),
                        label=f"hover loosen reissue {attempt + 1}/{attempts}",
                    )

        swivel_torque = int(grip.get("swivel_torque", grip["settle_torque"]))
        self.set_gripper_torque(swivel_torque)
        if self.live:
            time.sleep(float(placement["hover_release_dwell_seconds"]))

    def correct_release_hover_upward(
        self,
        command_x: float,
        command_y: float,
        command_z: float,
        *,
        physical_target: tuple[float, float, float] | None = None,
        label: str = "Release hover",
        tolerance_mm: float | None = None,
        maximum_attempts: int | None = None,
        maximum_total_mm: float | None = None,
    ) -> float:
        """Correct only upward using physical feedback, never a compensated command target.

        Returns the Z command that achieved (or most closely approached) the
        requested physical height.  It never issues a downward correction.
        """
        if self.dry_run:
            return command_z
        placement = self.settings["placement"]
        target_x, target_y, target_z = physical_target or (command_x, command_y, command_z)
        tolerance = float(
            placement["hover_z_tolerance_mm"] if tolerance_mm is None else tolerance_mm
        )
        attempts = max(
            0,
            int(
                placement["hover_upward_corrections"]
                if maximum_attempts is None
                else maximum_attempts
            ),
        )
        maximum_total = float(
            placement["hover_max_total_correction_mm"]
            if maximum_total_mm is None
            else maximum_total_mm
        )
        total_correction = 0.0

        for attempt in range(attempts + 1):
            try:
                data = self.feedback()
            except Exception as exc:
                print(
                    f"  WARNING: {label} feedback failed ({exc}); keeping the last safe "
                    "upward command instead of moving lower."
                )
                return command_z
            assert data is not None
            measured_z = data["xyz"][2]
            error = math.dist(data["xyz"], (target_x, target_y, target_z))
            z_shortfall = target_z - measured_z
            print(
                f"  {label.lower()} feedback: "
                f"measured={tuple(round(value, 1) for value in data['xyz'])} | "
                f"XYZ error={error:.1f} mm | Z shortfall={z_shortfall:.1f} mm"
            )
            if z_shortfall <= tolerance:
                if error > float(self.settings["motion"]["low_tolerance_mm"]):
                    print(f"  WARNING: lateral {label.lower()} error is high; Z remains safely airborne.")
                return command_z
            if attempt == attempts or total_correction >= maximum_total:
                print(
                    f"  WARNING: {label} remains lower than requested; no downward command "
                    "will be sent."
                )
                remaining = max(0.0, maximum_total - total_correction)
                safety_lift = min(remaining, max(5.0, z_shortfall))
                if safety_lift > 0.0:
                    next_command_z = command_z + safety_lift
                    print(
                        f"  applying final conservative upward safety lift: "
                        f"+{safety_lift:.1f} mm"
                    )
                    self.vertical(
                        command_x,
                        command_y,
                        command_z,
                        next_command_z,
                        f"{label} final upward safety lift",
                        step_mm=5.0,
                        check_feedback=False,
                        feedback_target=(target_x, target_y, target_z),
                        adaptive_corrections=False,
                    )
                    command_z = next_command_z
                return command_z

            upward = min(z_shortfall, maximum_total - total_correction)
            next_command_z = command_z + upward
            print(f"  safe upward {label.lower()} correction #{attempt + 1}: +{upward:.1f} mm")
            try:
                self.vertical(
                    command_x,
                    command_y,
                    command_z,
                    next_command_z,
                    f"{label} upward correction",
                    step_mm=5.0,
                    check_feedback=False,
                    tolerance_mm=max(tolerance, float(self.settings["motion"]["low_tolerance_mm"])),
                    feedback_target=(target_x, target_y, target_z),
                    adaptive_corrections=False,
                )
            except Exception as exc:
                print(
                    f"  WARNING: upward {label.lower()} correction failed ({exc}); "
                    "no downward command will be sent."
                )
                return command_z
            command_z = next_command_z
            total_correction += upward
        return command_z

    def airborne_full_release(self, piece: str):
        """Open decisively at the verified airborne drop point."""
        grip = self.settings["gripper"]
        open_angle = self._release_open_angle(piece)
        self.set_gripper_torque(int(grip["carry_torque"]))
        print(
            f"Airborne full release: {self.current_grip:.3f} -> {open_angle:.3f} rad "
            "in one high-torque command"
        )
        self.gripper(
            open_angle,
            speed=int(grip["command_speed"]),
            acceleration=int(grip["command_acceleration"]),
            wait_seconds=float(grip.get("transport_grip_wait_seconds", 0.12)),
            label="airborne full open",
        )
        if self.live:
            data = self.feedback()
            assert data is not None
            actual_raw = data.get("t")
            tolerance = float(self.settings["placement"].get("hover_angle_tolerance_rad", 0.04))
            if actual_raw is not None and float(actual_raw) > open_angle + tolerance:
                print(
                    f"  Gripper remained at {float(actual_raw):.3f} rad; reissuing full open "
                    "before retreat."
                )
                self.gripper(
                    open_angle,
                    speed=int(grip["command_speed"]),
                    acceleration=int(grip["command_acceleration"]),
                    wait_seconds=float(grip.get("transport_grip_wait_seconds", 0.12)),
                    label="airborne full-open reissue",
                )

    def release_from_hover(
        self,
        *,
        command_x: float,
        command_y: float,
        hover_command_z: float,
        physical_hover_target: tuple[float, float, float],
        physical_release_target: tuple[float, float, float],
        piece: str,
    ) -> float:
        """Swivel while high, descend without downward correction, drop, and rise."""
        placement = self.settings["placement"]
        motion = self.settings["motion"]
        self.hover_soften(piece)

        separation_lift = float(placement.get("post_soften_lift_mm", 0.0))
        lifted_hover_z = hover_command_z + separation_lift
        if separation_lift > 0.0:
            self.vertical(
                command_x,
                command_y,
                hover_command_z,
                lifted_hover_z,
                "Post-soften airborne separation lift",
                step_mm=5.0,
                check_feedback=False,
                feedback_target=(
                    physical_hover_target[0],
                    physical_hover_target[1],
                    physical_hover_target[2] + separation_lift,
                ),
                adaptive_corrections=False,
            )
            if self.live:
                time.sleep(float(placement.get("post_soften_lift_dwell_seconds", 0.0)))

        release_command_z = hover_command_z + (
            physical_release_target[2] - physical_hover_target[2]
        )
        self.vertical(
            command_x,
            command_y,
            lifted_hover_z,
            release_command_z,
            "Airborne release descent (downward correction disabled)",
            step_mm=float(placement["final_lower_step_mm"]),
            check_feedback=False,
            tolerance_mm=float(motion["low_tolerance_mm"]),
            feedback_target=physical_release_target,
            adaptive_corrections=False,
        )
        release_command_z = self.correct_release_hover_upward(
            command_x,
            command_y,
            release_command_z,
            physical_target=physical_release_target,
            label="Airborne release",
            tolerance_mm=float(placement.get("release_z_tolerance_mm", 2.0)),
            maximum_attempts=int(placement.get("release_upward_corrections", 2)),
            maximum_total_mm=float(placement.get("release_max_total_correction_mm", 20.0)),
        )
        self.confirm("Open fully and drop the piece from the verified airborne height?")
        self.airborne_full_release(piece)

        clearance_z = release_command_z + float(placement["slow_clearance_lift_mm"])
        self.vertical(
            command_x,
            command_y,
            release_command_z,
            clearance_z,
            "Immediate straight-up post-release retreat",
            step_mm=10.0,
            check_feedback=False,
            adaptive_corrections=False,
        )
        return clearance_z

    def centred_pickup_close(self, closed_angle: float):
        """Close progressively at reduced torque so symmetric pads can self-centre."""
        grip = self.settings["gripper"]
        start_angle = self.current_grip
        steps = max(1, int(grip["pickup_close_steps"]))
        self.set_gripper_torque(int(grip["pickup_torque"]))
        print(f"Centred pickup close: {start_angle:.3f} -> {closed_angle:.3f} rad in {steps} steps")
        for step in range(1, steps + 1):
            angle = start_angle + (closed_angle - start_angle) * step / steps
            self.gripper(
                angle,
                wait_seconds=float(grip["pickup_step_delay_seconds"]),
                label=f"pickup close {step}/{steps}",
            )
        if self.live:
            time.sleep(float(grip["pickup_settle_seconds"]))
        self.set_gripper_torque(int(grip["carry_torque"]))

    def secure_transport_grip(self, piece: str, closed_angle: float) -> float:
        """Compress the foam firmly before any lift, pitch change, or lateral travel."""
        grip = self.settings["gripper"]
        extra = float(grip.get("transport_close_extra_rad", 0.0))
        maximum = float(grip.get("transport_max_angle_rad", 3.1))
        transport_angle = min(3.14, maximum, float(closed_angle) + extra)
        self.set_gripper_torque(int(grip["carry_torque"]))
        self.gripper(
            transport_angle,
            wait_seconds=float(grip.get("transport_grip_wait_seconds", 0.12)),
            label=f"secure {piece} transport grip",
        )
        return transport_angle

    def prepare_floating_carry(self):
        """Let the suspended piece pivot in the foam while remaining clamped."""
        grip = self.settings["gripper"]
        if not bool(grip.get("floating_carry_enabled", False)):
            print("Rigid carry grip selected")
            return
        start_angle = self.current_grip
        open_angle = self._pickup_open_angle(piece)
        fraction = float(grip["floating_carry_release_fraction"])
        if not 0.0 < fraction < 0.5:
            raise ValueError("gripper.floating_carry_release_fraction must be between 0 and 0.5")
        target_angle = start_angle + (open_angle - start_angle) * fraction
        steps = max(1, int(grip["floating_carry_steps"]))
        self.set_gripper_torque(int(grip["floating_carry_torque"]))
        print(
            f"Floating carry grip: {start_angle:.3f} -> {target_angle:.3f} rad "
            f"at torque {int(grip['floating_carry_torque'])}/1000"
        )
        for step in range(1, steps + 1):
            angle = start_angle + (target_angle - start_angle) * step / steps
            self.gripper(
                angle,
                speed=int(grip["release_speed"]),
                acceleration=int(grip["release_acceleration"]),
                wait_seconds=float(grip["floating_carry_step_delay_seconds"]),
                label=f"floating regrip {step}/{steps}",
            )
        if self.live:
            time.sleep(float(grip["floating_carry_dwell_seconds"]))

    def pickup_test(self, square: str, piece: str = "pawn"):
        """Pick up briefly and replace on the same square to tune jaw symmetry."""
        self.reload_tuning()
        square = square.lower()
        piece = piece.lower()
        self._validate_piece(piece)
        preview = self.preview(square, square, piece)
        source_x, source_y, source_z = preview["source_grip"]
        motion = self.settings["motion"]
        grip = self.settings["gripper"]
        placement = self.settings["placement"]
        source_high = (source_x, source_y, source_z + float(motion["lift_mm"]))
        open_angle = self._pickup_open_angle(piece)
        closed_angle = float(grip["closed_angles"][piece])

        print(f"\nPickup test: {square.upper()} | {piece}")
        print("Grip point:", tuple(round(value, 1) for value in preview["source_grip"]))
        if self.dry_run:
            print("DRY RUN: commands will only be printed.")
        self.confirm("Clear the nearby squares and path. Start pickup test?")

        self.set_gripper_torque(int(grip["carry_torque"]))
        self.move_init("Start", restore_grip=open_angle)
        source_command = self.approach_high(source_high, "Source above")
        corrected_x, corrected_y, corrected_high_z = source_command
        corrected_source_z = source_z + (corrected_high_z - source_high[2])
        if placement["confirm_low_moves"]:
            self.confirm("Descend at source?")
        self.vertical(
            corrected_x,
            corrected_y,
            corrected_high_z,
            corrected_source_z,
            "Pickup descent",
            tolerance_mm=float(motion.get("pickup_low_tolerance_mm", motion["low_tolerance_mm"])),
            feedback_target=(source_x, source_y, source_z),
        )
        self.confirm("Close the gripper progressively?")
        self.centred_pickup_close(closed_angle)
        self.secure_transport_grip(piece, closed_angle)

        test_lift_z = min(
            corrected_high_z,
            corrected_source_z + float(motion["pickup_test_lift_mm"]),
        )
        self.vertical(
            corrected_x,
            corrected_y,
            corrected_source_z,
            test_lift_z,
            "Short inspection lift",
        )
        self.confirm("Inspect whether the piece is centred. Begin hover release?")
        hover_height = float(preview["hover_height_mm"])
        return_hover_z = corrected_source_z + hover_height
        physical_hover_target = (source_x, source_y, source_z + hover_height)
        self.vertical(
            corrected_x,
            corrected_y,
            test_lift_z,
            return_hover_z,
            "Return to release hover",
            check_feedback=False,
            feedback_target=physical_hover_target,
        )
        release_command_z = self.correct_release_hover_upward(
            corrected_x,
            corrected_y,
            return_hover_z,
            physical_target=physical_hover_target,
        )
        physical_release_target = (
            source_x,
            source_y,
            source_z + float(placement["drop_release_height_mm"]),
        )
        clearance_z = self.release_from_hover(
            command_x=corrected_x,
            command_y=corrected_y,
            hover_command_z=release_command_z,
            physical_hover_target=physical_hover_target,
            physical_release_target=physical_release_target,
            piece=piece,
        )
        self.vertical(
            corrected_x,
            corrected_y,
            clearance_z,
            corrected_high_z,
            "Retreat",
        )
        self.move_init("Finish")
        print("Pickup test done. Adjust the foam or pickup settings before repeating if it was off-centre.")

    def remove_to_capture_drop(
        self,
        source: str,
        piece: str,
        drop_xyz: tuple[float, float, float],
    ):
        """Pick a captured piece from the board and release it over a tray."""
        self.reload_tuning()
        source = source.lower()
        piece = piece.lower()
        self._validate_piece(piece)
        preview = self.preview(source, source, piece)
        source_x, source_y, source_z = preview["source_grip"]
        drop_x, drop_y, drop_z = (float(value) for value in drop_xyz)
        motion = self.settings["motion"]
        grip = self.settings["gripper"]
        lift = float(motion["lift_mm"])
        source_high = (source_x, source_y, source_z + lift)
        carrying_clearance_z = float(
            motion.get("carrying_clearance_z_mm", motion.get("init_clearance_z_mm", 80.0))
        )
        drop_high = (drop_x, drop_y, max(drop_z + 60.0, carrying_clearance_z))
        open_angle = self._pickup_open_angle(piece)
        closed_angle = float(grip["closed_angles"][piece])

        print(f"\nCapture removal: {source.upper()} | {piece}")
        if self.dry_run:
            print("DRY RUN: commands will only be printed.")
        self.confirm("Clear the capture path and tray. Start?")
        self.set_gripper_torque(int(grip["carry_torque"]))
        self.move_init("Start capture removal", restore_grip=open_angle)

        pickup_attempt = 0
        precision_limit = max(0, int(motion.get("precision_retry_limit", 10)))
        while True:
            source_command = self.approach_high(source_high, "Captured piece above")
            corrected_source_x, corrected_source_y, corrected_source_high_z = source_command
            corrected_source_z = source_z + (corrected_source_high_z - source_high[2])
            try:
                self.vertical(
                    corrected_source_x,
                    corrected_source_y,
                    corrected_source_high_z,
                    corrected_source_z,
                    "Captured-piece descent",
                    tolerance_mm=float(motion.get("pickup_low_tolerance_mm", motion["low_tolerance_mm"])),
                    feedback_target=(source_x, source_y, source_z),
                )
                break
            except RuntimeError as exc:
                pickup_attempt += 1
                if pickup_attempt >= precision_limit:
                    print(
                        f"  Captured-piece pickup precision retry limit ({precision_limit}) "
                        f"reached; forcing continuation at the current pose ({exc})."
                    )
                    break
                print(
                    f"  Captured-piece pickup did not settle ({exc}). Returning to Init and "
                    f"retrying automatically (round {pickup_attempt}; Ctrl+C to stop)."
                )
                self.set_gripper_torque(int(grip["carry_torque"]))
                self.move_init("Captured-piece pickup recovery", restore_grip=open_angle)
        self.centred_pickup_close(closed_angle)
        carry_angle = self.secure_transport_grip(piece, closed_angle)
        self.lift_current_xy(
            max(
                corrected_source_high_z,
                carrying_clearance_z,
            ),
            "Captured-piece straight-up clearance",
            fallback_xyz=(corrected_source_x, corrected_source_y, corrected_source_z),
        )

        self.move_init("Initial position carrying captured piece", restore_grip=carry_angle)
        self.coarse_move(
            drop_high,
            "Capture tray high transit",
            tolerance_mm=float(grip.get("capture_tray_tolerance_mm", 35.0)),
            speed=float(grip.get("capture_tray_travel_speed", motion["travel_speed"])),
        )
        self.vertical(
            drop_x,
            drop_y,
            drop_high[2],
            drop_z,
            "Capture-tray descent",
            check_feedback=False,
            tolerance_mm=float(grip.get("capture_tray_tolerance_mm", 35.0)),
        )
        tray_open_angle = float(grip.get("capture_tray_open_angle", grip["open_angle"]))
        self.set_gripper_torque(int(grip["carry_torque"]))
        self.gripper(
            tray_open_angle,
            wait_seconds=float(grip.get("transport_grip_wait_seconds", 0.12)),
            label="capture tray immediate drop",
        )
        self.vertical(
            drop_x,
            drop_y,
            drop_z,
            drop_high[2],
            "Capture-tray retreat",
            check_feedback=False,
            tolerance_mm=float(grip.get("capture_tray_tolerance_mm", 35.0)),
        )
        self.move_init("Finish capture removal")

    def execute(self, source: str, destination: str, piece: str = "pawn"):
        self.reload_tuning()
        source = source.lower()
        destination = destination.lower()
        piece = piece.lower()
        if source == destination:
            raise ValueError("Source and destination must differ")
        self._validate_piece(piece)
        preview = self.preview(source, destination, piece)
        source_x, source_y, source_z = preview["source_grip"]
        dest_x, dest_y, dest_z = preview["destination_grip"]
        motion = self.settings["motion"]
        grip = self.settings["gripper"]
        placement = self.settings["placement"]
        lift = float(motion["lift_mm"])
        source_high = (source_x, source_y, source_z + lift)
        destination_high = (dest_x, dest_y, dest_z + lift)
        open_angle = self._pickup_open_angle(piece)
        closed_angle = float(grip["closed_angles"][piece])

        print(f"\n{source.upper()} -> {destination.upper()} | {piece}")
        print("Pickup:", tuple(round(value, 1) for value in preview["source_grip"]))
        print(
            "Pickup cushion compensation:",
            tuple(round(value, 1) for value in preview["source_alignment_delta"]),
        )
        print(
            "Release hover:",
            tuple(round(value, 1) for value in preview["destination_grip"]),
            f"| {preview['hover_height_mm']:.1f} mm above upright board level "
            f"({preview['tilt_clearance_mm']:.1f} mm far-rank tilt clearance)",
        )
        print("Free-drop release:", tuple(round(value, 1) for value in preview["destination_release_grip"]))
        print("Board level reference:", tuple(round(value, 1) for value in preview["destination_board_grip"]))
        print("Destination correction:", preview["placement_correction"])
        if self.dry_run:
            print("DRY RUN: commands will only be printed.")
        self.confirm("Check the board and clear the entire path. Start?")

        self.set_gripper_torque(int(grip["carry_torque"]))
        self.move_init("Start", restore_grip=open_angle)

        pickup_attempt = 0
        precision_limit = max(0, int(motion.get("precision_retry_limit", 10)))
        while True:
            source_command = self.approach_high(
                source_high,
                "Source above",
                tolerance=float(motion.get("pickup_high_tolerance_mm", motion["high_tolerance_mm"])),
            )
            corrected_source_x, corrected_source_y, corrected_source_high_z = source_command
            corrected_source_z = source_z + (corrected_source_high_z - source_high[2])
            if placement["confirm_low_moves"]:
                self.confirm("Descend at source?")
            try:
                self.vertical(
                    corrected_source_x,
                    corrected_source_y,
                    corrected_source_high_z,
                    corrected_source_z,
                    "Pickup descent",
                    tolerance_mm=float(motion.get("pickup_low_tolerance_mm", motion["low_tolerance_mm"])),
                    feedback_target=(source_x, source_y, source_z),
                )
                break
            except RuntimeError as exc:
                pickup_attempt += 1
                if pickup_attempt >= precision_limit:
                    print(
                        f"  Pickup precision retry limit ({precision_limit}) reached; "
                        f"forcing continuation at the current pose ({exc})."
                    )
                    break
                print(
                    f"  Pickup positioning did not settle ({exc}). Returning to Init and "
                    f"retrying automatically (round {pickup_attempt}; Ctrl+C to stop)."
                )
                self.set_gripper_torque(int(grip["carry_torque"]))
                self.move_init("Pickup recovery", restore_grip=open_angle)
        self.confirm("Close gripper?")
        self.centred_pickup_close(closed_angle)
        carry_angle = self.secure_transport_grip(piece, closed_angle)
        if placement["confirm_low_moves"]:
            self.confirm("Lift the piece?")
        self.lift_current_xy(
            max(
                corrected_source_high_z,
                float(
                    motion.get(
                        "carrying_clearance_z_mm",
                        motion.get("init_clearance_z_mm", 80.0),
                    )
                ),
            ),
            "Pickup straight-up clearance",
            fallback_xyz=(corrected_source_x, corrected_source_y, corrected_source_z),
        )

        self.move_init("Initial position carrying piece", restore_grip=carry_angle)
        transit_z = max(
            destination_high[2],
            float(
                motion.get(
                    "carrying_clearance_z_mm",
                    motion.get("init_clearance_z_mm", 80.0),
                )
            ),
        )
        self.coarse_move(
            (destination_high[0], destination_high[1], transit_z),
            "Destination high transit",
            tolerance_mm=float(motion.get("high_transit_tolerance_mm", 20.0)),
            speed=float(motion.get("high_transit_speed", motion["travel_speed"])),
        )
        destination_command = self.approach_high(destination_high, "Destination above")
        corrected_dest_x, corrected_dest_y, corrected_dest_high_z = destination_command
        # Keep the full carry grip during the lateral trip.  Optional floating
        # carry may only begin once a dropped piece is already over its square.
        self.prepare_floating_carry()
        corrected_dest_z = dest_z + (corrected_dest_high_z - destination_high[2])
        if placement["confirm_low_moves"]:
            self.confirm("Descend to the release hover?")
        self.vertical(
            corrected_dest_x,
            corrected_dest_y,
            corrected_dest_high_z,
            corrected_dest_z,
            "Release-hover descent",
            check_feedback=False,
            feedback_target=(dest_x, dest_y, dest_z),
        )
        release_command_z = self.correct_release_hover_upward(
            corrected_dest_x,
            corrected_dest_y,
            corrected_dest_z,
            physical_target=(dest_x, dest_y, dest_z),
        )
        self.confirm("Loosen the padded grip and let the piece hang vertically?")
        clearance_z = self.release_from_hover(
            command_x=corrected_dest_x,
            command_y=corrected_dest_y,
            hover_command_z=release_command_z,
            physical_hover_target=(dest_x, dest_y, dest_z),
            physical_release_target=tuple(preview["destination_release_grip"]),
            piece=piece,
        )
        self.vertical(
            corrected_dest_x,
            corrected_dest_y,
            clearance_z,
            corrected_dest_high_z,
            "Drop lift",
            # The piece has already been released. A lagging upward-retreat
            # reading must not roll back the logical chess move.
            check_feedback=False,
        )
        self.move_init("Finish")
        print("Done. Record whether the piece is centred and upright before the next trial.")

