# N10 final physical chess bot

N10 combines the `timing_hybrid_aggressive` engine and exact best-checkpoint weights from N6 with the latest calibrated RoArm movement controller. It is standalone: it does not import anything from N6, N9, or the robotics test folder.

Normal play uses only the original N6 neural/search engine. The optional Stockfish verifier is disabled in `config.json`, is not launched by `play.bat`, and does not inspect or replace moves.

## Hardware and Wi-Fi

This project is built for the **Waveshare RoArm-M2-S** robotic arm in its normal clamp/gripper configuration. The arm is controlled over its built-in ESP32 web/HTTP interface.

Factory/default access point:

| Setting | Value |
|---|---|
| Wi-Fi network | `RoArm-M2` |
| Wi-Fi password | `12345678` |
| Arm address in AP mode | `192.168.4.1` |
| Control page | `http://192.168.4.1` |

The computer running N10 must be connected to the same network as the arm. If the arm has been changed to router/STA mode, use the IP shown on the arm's OLED screen and update `robot_ip` in `settings.json`.

Useful manufacturer documentation:

- [RoArm-M2-S product/wiki page](https://www.waveshare.com/wiki/RoArm-M2-S)
- [RoArm-M2-S Wi-Fi configuration](https://www.waveshare.com/wiki/RoArm-M2-S_WIFI_Configuration)
- [RoArm-M2-S HTTP/Python control](https://www.waveshare.com/wiki/RoArm-M2-S_Python_HTTP_Request_Communication)

## Setup on a new Windows computer

1. Copy the complete `n10` folder; keep the `assets` folder, the `.pth` model weights, calibration JSON files, and `games` folder together.
2. Install 64-bit Python 3.10 or newer and enable **Add Python to PATH** during installation. A CUDA-capable NVIDIA GPU is optional; PyTorch falls back to CPU if CUDA is unavailable.
3. Open Command Prompt or PowerShell in this folder and run:

   ```powershell
   .\setup.bat
   ```

   The setup script reuses packages already installed and installs only the required `python-chess`, NumPy, PyTorch, and `requests` packages when needed. It does not create a virtual environment.
4. Connect to the arm's Wi-Fi using the credentials above and confirm that `http://192.168.4.1` opens.
5. Keep the board, arm base, padding, and table in the same physical arrangement as the supplied calibration. If any of those move, follow the recalibration section below.

The supplied `board_corners.json`, `placement_corrections.json`, `capture_drop.json`, and `settings.json` are the current tuned values for this arm and board. Do not delete them before the first run.

## Start playing

1. Power on the RoArm and connect the computer to the RoArm Wi-Fi network.
2. Put all pieces in the normal starting position, empty the capture tray, and keep the arm/board in their calibrated locations.
3. Double-click `play.bat`, or run it from a terminal so you can see the saved-game path if the game is interrupted.

N10 now uses the packages in your normal Python installation; it does not create a virtual environment. On a different computer, run `setup.bat` once. It first reuses any packages already installed and only downloads missing ones.

The supplied `config.json` makes the robot **White** and you **Black**, so the robot makes the first move. After it physically makes a move, make your reply on the real board and type that same move at the prompt. SAN such as `e4`, `Nf6`, `Nxd5`, and `O-O` is accepted; UCI such as `e7e5` and `g8f6` is also accepted. The program checks the move for legality before continuing. Captured pieces are removed to the calibrated tray automatically.

To play White instead:

```powershell
.\play.bat --human white
```

The robot will then wait for your White move before choosing and physically playing Black's reply. To test the chess engine without moving the arm, use `python main.py --no-arm`.

Each completed move is written to `games/latest.pgn` and to a timestamped PGN in `games`. If a move is interrupted, restore the real board to the displayed position and resume with the command printed by the program, for example:

```powershell
python main.py --resume "games\game_20261028_035123.pgn"
```

## First capture-tray setup

The capture tray is the only coordinate that cannot be safely copied or guessed. On the first real launch, press Enter when asked to calibrate it, support the torque-free arm, and place the open jaws over a tray outside the board at the desired release height. This is saved to `capture_drop.json`; later games do not ask again.

You can calibrate it directly:

```powershell
python main.py --calibrate-capture
```

During an engine capture, N10 first picks up the captured piece, releases it over the tray, returns to Init, and then moves its own piece onto the destination square. It also handles en passant and both castling directions. Promotions require you to replace the pawn manually.

## Current arm behaviour

- real firmware Init at game start, between carrying phases, and after every move;
- current board-corner calibration and padded-gripper local offset;
- progressive low-torque centring, followed by a `3.06 rad` / `420`-torque transport clamp before the first lift and throughout Init/lateral travel;
- a destination-only measured-reach profile: no added residual at D2, about `2.0 mm` toward the a-file at D4, about `15.5 mm` at A7, and about `14.5 mm` at A8, on top of the existing tool-centre alignment; pickup coordinates are untouched;
- a rank-1 rook pickup correction `3.0 mm` inward so the fixed padded jaw does not pass in front of the rook;
- piece-specific wider pickup openings, especially for the square king and queen, so the fixed jaw clears their 20 mm bases before closing;
- straight-up clearance to `100 mm` before any lateral carrying move, plus a high destination waypoint so the piece cannot sweep through occupied squares;
- faster `0.80` high-transit profile with 20 mm clearance steps while retaining fine 3 mm final lowering;
- zero fixed Cartesian/Init dwell: feedback advances each phase immediately when real motion completes;
- adaptive completion polling, so fast non-blocking arm commands cannot be mistaken for failed motion;
- automatic Init-and-retry loops for elevated approaches and pre-grip descents instead of ending the game;
- precision retries are capped at 4 attempts; after that the controller logs a warning and continues from the best current pose rather than repeating the same correction forever;
- automatic indefinite retry if Windows temporarily switches away from the RoArm Wi-Fi;
- reach-aware gravity-alignment hover is based on the greatest radial extension of the whole transfer, not rank alone: `18 mm` for near-arm moves and up to `53.2 mm` for far-arm moves; while airborne, the jaws now loosen only from about `3.06` to `2.99 rad`, then do most of the opening at the lower airborne release point so pawns and other pieces stay held during the high-stage swivel;
- final drop height also follows transfer reach: about `8 mm` near the arm and up to `15 mm` on far transfers; adaptive downward correction is disabled, only upward safety corrections are permitted, and full opening is followed by an immediate `25 mm` straight-up retreat;
- compensated robot commands and intended physical hover/release coordinates are tracked separately, preventing live feedback from driving the arm below a safe release target;
- capture-tray disposal uses one broad high waypoint and an immediate full-power opening; it deliberately skips square-level correction, soft alignment, and progressive release;
- recovery command that opens only the jaws.

## Commands

```powershell
python main.py --self-test
python main.py --guard-test
python main.py --human white
python main.py --human black
python main.py --no-arm
python main.py --dry-arm
python main.py --calibrate-capture
python main.py --open-gripper
python main.py --resume games/latest.pgn
python selfplay.py
```

## Full physical movement replay

`movement_test.bat` replays the latest completed human-versus-N10 game on the real board, with the arm moving both White and Black. The bundled game is a legal 39-ply checkmate containing 13 captures and kingside castling, so it exercises ordinary transfers, far-rank placement, capture-tray disposal, and the two-piece castling sequence. It does not load the neural engine, CUDA weights, or Stockfish.

Before starting, put every piece in the normal initial chess position and empty the capture tray. Then double-click:

```powershell
.\movement_test.bat
```

To run only the first six plies on the real arm:

```powershell
python replay_game.py --live --yes --end-ply 6
```

Running `python replay_game.py` without `--live` prints the complete command sequence without contacting the arm. A successful physical ply is checkpointed before the next one; if replay stops, the program prints the exact FEN to restore and a `--start-ply` resume command. Any other PGN can be supplied as the positional argument.

`python main.py --guard-test` exists only as an optional diagnostic for the previously tested verifier. Normal games and setup do not run it. `--no-tactical-guard` can also explicitly enforce the already-disabled setting.

`python selfplay.py` runs the exact original N6 timing-hybrid-aggressive player against itself at 30 seconds per non-book move. Stockfish is forcibly disabled, the original shared-cache self-play behavior is preserved, and every move is annotated with depth, evaluation, nodes, and think time in `games/latest_selfplay.pgn`.

The opening is intentionally deterministic, exactly like the strongest N6 configuration: the same position/history selects the same most-common book move. A repeated opening therefore confirms reproducibility, not that the wrong engine or weights were loaded. Random opening selection is not enabled because it would no longer be the exact N6 player and can select a weaker line.

If a run is interrupted while holding a piece, support the piece and run `python main.py --open-gripper`.

For arm-only testing and fine tuning:

```powershell
python placement_lab.py move e2 e4 pawn --live --yes
python placement_lab.py alignment-nudge left 0.5
python placement_lab.py alignment-nudge left 0.5 --pickup-only
python placement_lab.py pickup-test d4 pawn --live --yes --left-offset-delta-mm 0.3
python placement_lab.py landing-profile
python placement_lab.py landing-error d8 toward-h 0.10
```

`alignment-nudge left` moves the hand left as viewed from the robot base looking outward, compensating for the thicker left jaw pad. The correction rotates with the arm, so do not replace it with a fixed board-axis offset.
Use `--pickup-only` when the open foam bumps a piece during descent but the landing position is already good.
Use `--left-offset-delta-mm` to try a temporary cushion correction without saving it. Once a trial works, save the same amount with `python placement_lab.py alignment-nudge left AMOUNT`.

`landing-profile` is a display-only command: it never contacts or moves the arm. The reach profile is destination-only, so it cannot change pickup. Its samples are expressed in fractions of the board's calibrated square width and are interpolated by the arm's physical radial reach, which allows a small near-board correction and a much larger far-board correction without assuming the error is linear by rank. Positive command values mean toward the h-file and negative values mean toward the a-file.

The current residual profile adds 0 at D2, 0.05 square toward the a-file at D4, 0.40 square at A7, and 0.375 square at A8. The profile is interpolated by measured arm extension rather than rank, and is applied on top of the existing gripper/padding alignment. At the far end, the arm's right side is the h-file side, but use the printed a/h board labels rather than visual left/right when measuring.

After a physical test, `landing-error SQUARE toward-h FRACTION` means the piece actually missed toward the h-file; the tool saves an equal correction toward the a-file. Use `toward-a` for the opposite miss. The update is cumulative, so if a corrected D8 move still misses 0.10 square toward h, `python placement_lab.py landing-error d8 toward-h 0.10` adds another 0.10-square correction toward a. This command only edits `placement_corrections.json`; it does not move the arm.

For a measured millimetre error instead, add `--unit mm`, for example `python placement_lab.py landing-error d8 toward-h 4 --unit mm`.

## Recalibration

If the arm, board, padding, or table position changes:

```powershell
python calibrate_board.py --live
python calibrate_gripper_alignment.py d4 e4 d5 --piece pawn --live
python main.py --calibrate-capture
```

Every calibration returns the arm to Init. A different person or robot must perform all three calibrations before physical play.

## Safety and recovery

- Keep a hand near the power switch and do not run the arm unattended.
- Clear the capture path and tray area.
- If a physical move fails, the logical board is not advanced; restore the real pieces to the displayed position before resuming.
- Games are saved after every completed move in `games/latest.pgn`.

## Troubleshooting

- **The web page or program cannot connect:** reconnect to `RoArm-M2`, confirm `192.168.4.1` opens in a browser, and make sure no VPN or other network adapter is routing the request elsewhere. If the arm is in STA mode, replace `robot_ip` in `settings.json` with the address shown on its OLED screen.
- **`ModuleNotFoundError` or missing packages:** run `.\setup.bat` from the N10 folder. If it needs to install packages, the computer must have internet access for that installation step.
- **The arm or board was physically moved:** stop using the old calibration and run the three commands in [Recalibration](#recalibration) before playing another game.
- **A run stopped while the jaws still hold a piece:** support the piece, run `python main.py --open-gripper`, restore the board to the displayed position, and then use the printed `--resume` command.

## Included runtime files

- `engine.py`, `assets/timing_hybrid_aggressive.pth`, `assets/openings.npz` — original timing-hybrid-aggressive player and exact N6 best weights.
- `assets/stockfish/` — optional disabled diagnostic verifier, Windows AVX2 executable, GPLv3 license, and corresponding source code; normal play does not launch it.
- `main.py` — complete legal game loop and PGN recovery.
- `replay_game.py`, `movement_test.bat`, `movement_test_game.pgn` — engine-free full-game physical movement test using the latest completed game.
- `physical_arm.py`, `roarm_controller.py` — chess adapter and current arm movement.
- `settings.json`, `board_corners.json`, `placement_corrections.json` — current physical tuning.
- calibration scripts, setup scripts, and requirements.
