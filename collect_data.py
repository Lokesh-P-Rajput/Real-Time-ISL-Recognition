"""STEP 1 - Record landmark samples for every gesture with your webcam.

    python collect_data.py                        record all gestures + Neutral
    python collect_data.py --gestures Hello Yes   record only some gestures
    python collect_data.py --gestures "Thank You" quote names that contain a space
    python collect_data.py --list                 show how many samples exist
    python collect_data.py --delete Hello         remove the recorded samples of a sign
    python collect_data.py --redo Hello           delete the old samples of a sign and record it again
    python collect_data.py --rename Hello Namaste rename a sign (dataset, database, photo and sheet)

The list of signs is assets/sign_guide.csv (edit it in Excel). A name that is not in the sheet yet is added to it automatically.

A photo of each gesture is saved automatically to assets/signs/ for the app's "Sign guide" tab
(replace it with your own picture any time; it is only overwritten with --refresh-photos).

Keys in the camera window:  SPACE = start | S = skip gesture | Q = quit (keeps what was recorded)
Run the script again on another day / with another background and light: every run is
stored as a new "session", which makes the model (and the evaluation) more honest.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

from isl import config
from isl import database as db
from isl import guide
from isl.camera import open_camera
from isl.features import FEATURE_NAMES
from isl.hand_tracker import HandTracker

WINDOW = "ISL data collection"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def sample_counts() -> pd.Series:
    if not config.DATASET_CSV.exists():
        return pd.Series(dtype=int)
    return pd.read_csv(config.DATASET_CSV, usecols=["label"])["label"].value_counts()


def all_labels() -> list:
    return list(config.GESTURES) + [config.NEUTRAL_LABEL]


def print_counts() -> None:
    counts = sample_counts()
    if counts.empty:
        print("No samples recorded yet.")
        return
    print("\nSamples in dataset:")
    for label in all_labels() + [x for x in counts.index if x not in all_labels()]:
        print(f"  {label:<12} {int(counts.get(label, 0))}")
    print(f"  {'TOTAL':<12} {int(counts.sum())}")


def save_samples(label: str, rows: list, session: str) -> None:
    frame = pd.DataFrame(np.vstack(rows), columns=FEATURE_NAMES)
    frame.insert(0, "session", session)
    frame.insert(0, "label", label)
    frame.to_csv(config.DATASET_CSV, mode="a", header=not config.DATASET_CSV.exists(), index=False)
    db.add_dataset_samples(label, rows)
    print(f"  saved {len(rows)} samples of '{label}'")


def delete_label(label: str) -> None:
    if config.DATASET_CSV.exists():
        data = pd.read_csv(config.DATASET_CSV)
        removed = int((data["label"] == label).sum())
        data[data["label"] != label].to_csv(config.DATASET_CSV, index=False)
        print(f"Removed {removed} samples of '{label}' from {config.DATASET_CSV.name}")
    db.delete_dataset_samples(label)


def rename_label(old: str, new: str) -> None:
    """Rename a sign in the dataset, the database, the photo and the sign sheet."""
    new = new.strip()
    if config.DATASET_CSV.exists():
        data = pd.read_csv(config.DATASET_CSV)
        if (data["label"] == new).any():
            sys.exit(f"'{new}' already has recorded samples. Choose another name or delete those first.")
        changed = int((data["label"] == old).sum())
        data.loc[data["label"] == old, "label"] = new
        data.to_csv(config.DATASET_CSV, index=False)
        print(f"Renamed {changed} samples in {config.DATASET_CSV.name}")
    try:
        guide.rename_sign(old, new)
    except ValueError as exc:
        print(f"Sheet: {exc}")
    try:
        db.rename_gesture(old, new)
    except Exception as exc:
        print(f"Database: could not rename ({exc})")
    print(f"'{old}' is now '{new}'. Run train_model.py again so the model learns the new name.")


def banner(frame, line1: str, line2: str = "") -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 62), (30, 30, 30), -1)
    cv2.putText(frame, line1, (12, 28), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if line2:
        cv2.putText(frame, line2, (12, 52), FONT, 0.55, (190, 190, 190), 1, cv2.LINE_AA)


def collect_label(cap, tracker, label: str, target: int, existing: int, overwrite_photo: bool = False):
    """Record `target` samples of one gesture. Returns (status, rows)."""
    state, t_start, frame_no, rows = "wait", 0.0, 0, []
    while True:
        ok, raw = cap.read()
        if not ok:
            continue
        frame, features = tracker.process(raw, draw=True)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return "quit", rows
        if key == ord("s"):
            return "skip", []

        if state == "wait":
            banner(frame, f"Gesture: {label}   (already in dataset: {existing})",
                   "Hold the sign, then press SPACE | S = skip | Q = quit")
            if key == 32:
                state, t_start = "countdown", time.time()
        elif state == "countdown":
            left = 3 - int(time.time() - t_start)
            banner(frame, f"Gesture: {label}", "Get ready ...")
            if left <= 0:
                state = "record"
            else:
                cv2.putText(frame, str(left), (frame.shape[1] // 2 - 30, frame.shape[0] // 2 + 30),
                            FONT, 4, (0, 255, 255), 8, cv2.LINE_AA)
        else:  # record
            frame_no += 1
            if features is not None and frame_no % config.CAPTURE_EVERY_N_FRAMES == 0:
                rows.append(features)
                if len(rows) == max(1, target // 2) and label != config.NEUTRAL_LABEL:
                    # halfway through: keep a clean (mirrored, no skeleton) photo for the Sign guide
                    guide.save_reference_photo(label, cv2.flip(raw, 1), overwrite=overwrite_photo)
            banner(frame, f"Recording {label}: {len(rows)}/{target}",
                   "Slowly tilt, move and change distance a little - variety makes a better model")
            width = int(frame.shape[1] * min(len(rows) / target, 1.0))
            cv2.rectangle(frame, (0, 62), (width, 72), (0, 200, 0), -1)
            if features is None:
                cv2.putText(frame, "No hand detected!", (12, frame.shape[0] - 16),
                            FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            if len(rows) >= target:
                return "done", rows
        cv2.imshow(WINDOW, frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record ISL gesture samples")
    parser.add_argument("--gestures", nargs="+", help="gesture names (default: all + Neutral)")
    parser.add_argument("--samples", type=int, default=config.SAMPLES_PER_GESTURE)
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--refresh-photos", action="store_true",
                        help="replace the Sign guide photos with new ones from this run")
    parser.add_argument("--list", action="store_true", help="show sample counts and exit")
    parser.add_argument("--delete", nargs="+", metavar="GESTURE", help="delete samples and exit")
    parser.add_argument("--redo", nargs="+", metavar="GESTURE",
                        help="delete the old samples of these signs, then record them again")
    parser.add_argument("--rename", nargs=2, metavar=("OLD", "NEW"), help="rename a sign and exit")
    args = parser.parse_args()

    db.init_db()
    db.seed_gestures(config.GESTURES)
    guide.ensure_guide_csv()

    if args.list:
        print_counts()
        return
    if args.delete:
        for name in args.delete:
            delete_label(name)
        return
    if args.rename:
        rename_label(*args.rename)
        return

    if args.redo:
        for name in args.redo:
            delete_label(name)
    labels = args.redo or args.gestures or all_labels()
    for name in labels:
        if name != config.NEUTRAL_LABEL and name not in config.GESTURES:
            try:
                added = guide.add_sign(name)
            except ValueError as exc:
                sys.exit(f"Error: {exc}")
            if added:
                print(f"Added '{name}' to assets/sign_guide.csv (change its spoken text or description there).")
    db.seed_gestures(config.GESTURES)

    cap = open_camera(args.camera)
    if not cap.isOpened():
        sys.exit("Camera could not be opened. Close other apps that use it or try --camera 1.")
    tracker = HandTracker()
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    counts = sample_counts()      # after any --redo delete
    try:
        for label in labels:
            status, rows = collect_label(cap, tracker, label, args.samples, int(counts.get(label, 0)),
                                         args.refresh_photos or bool(args.redo))
            if rows:
                save_samples(label, rows, session)
            elif status == "skip":
                print(f"  skipped '{label}'")
            if status == "quit":
                break
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
    print_counts()


if __name__ == "__main__":
    main()
