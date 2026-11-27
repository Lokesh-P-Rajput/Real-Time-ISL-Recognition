"""The sign list (assets/sign_guide.csv) plus reference photos for the Sign guide tab.

sign_guide.csv has three columns:  gesture, spoken_text, description
One row per sign. Add a row = add a sign, delete a row = remove a sign.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import pandas as pd

from . import config

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
COLUMNS = ["gesture", "spoken_text", "description"]


def _file_stem(label: str) -> str:
    return re.sub(r"[^\w\- ]", "_", label).strip()


# ------------------------------------------------------------------ photos
def find_sign_image(label: str) -> Path | None:
    """Return assets/signs/<label>.jpg|png|webp if it exists."""
    for ext in IMAGE_EXTENSIONS:
        path = config.SIGN_IMAGE_DIR / f"{_file_stem(label)}{ext}"
        if path.exists():
            return path
    return None


def save_reference_photo(label: str, frame_bgr, overwrite: bool = False) -> Path | None:
    """Save a camera frame as the reference photo of a sign.

    An existing photo is kept (so a picture you added yourself is never replaced)
    unless overwrite=True. Returns the saved path, or None if nothing was written.
    """
    if find_sign_image(label) and not overwrite:
        return None
    config.SIGN_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in IMAGE_EXTENSIONS:                     # drop an older photo with another extension
        old = config.SIGN_IMAGE_DIR / f"{_file_stem(label)}{ext}"
        if old.exists():
            old.unlink()
    path = config.SIGN_IMAGE_DIR / f"{_file_stem(label)}.jpg"
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    buffer.tofile(str(path))          # also works for non-ASCII folder names on Windows
    return path


# --------------------------------------------------------------- the sheet
def _read_table() -> pd.DataFrame | None:
    path = config.SIGN_GUIDE_CSV
    if not path.exists():
        return None
    try:
        table = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    except Exception:
        return None
    table.columns = [str(c).strip().lower() for c in table.columns]
    return table if "gesture" in table.columns else None


def _write_table(table: pd.DataFrame) -> None:
    for column in COLUMNS:
        if column not in table.columns:
            table[column] = ""
    try:
        table[COLUMNS].to_csv(config.SIGN_GUIDE_CSV, index=False, encoding="utf-8-sig")
    except PermissionError as exc:
        raise ValueError(f"Close {config.SIGN_GUIDE_CSV.name} in Excel first, then run this again.") from exc


def load_descriptions() -> dict[str, str]:
    """sign -> description, read from assets/sign_guide.csv."""
    table = _read_table()
    if table is None or "description" not in table.columns:
        return {}
    return {g.strip(): d.strip() for g, d in zip(table["gesture"], table["description"])}


def ensure_guide_csv(gestures: dict | None = None) -> None:
    """Create assets/sign_guide.csv the first time, or upgrade an old two-column file."""
    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.SIGN_GUIDE_CSV
    if path.exists() and path.stat().st_size > 0:
        table = _read_table()
        if table is not None and "spoken_text" not in table.columns:      # older file: add the column
            table.insert(1, "spoken_text", [config.GESTURES.get(g.strip(), g.strip()) for g in table["gesture"]])
            _write_table(table)
        return
    gestures = config.DEFAULT_GESTURES if gestures is None else gestures
    _write_table(pd.DataFrame({"gesture": list(gestures), "spoken_text": list(gestures.values()),
                               "description": [""] * len(gestures)}))


def add_sign(name: str, spoken_text: str | None = None) -> bool:
    """Add a row for a new sign. Returns False when the sign already exists."""
    name = name.strip()
    if not name:
        raise ValueError("The sign name is empty.")
    if name.lower() == config.NEUTRAL_LABEL.lower():
        raise ValueError(f"'{config.NEUTRAL_LABEL}' is reserved for the no-sign class.")
    ensure_guide_csv()
    table = _read_table()
    if table is None:
        raise ValueError(f"{config.SIGN_GUIDE_CSV.name} could not be read. Close it in Excel and try again.")
    if name in set(table["gesture"].str.strip()):
        return False
    spoken = (spoken_text or name).strip()
    row = pd.DataFrame([{"gesture": name, "spoken_text": spoken, "description": ""}])
    _write_table(pd.concat([table, row], ignore_index=True))
    config.GESTURES[name] = spoken
    return True


def rename_sign(old: str, new: str) -> None:
    """Rename a sign in the sheet and rename its photo. Dataset/database are handled by collect_data.py."""
    new = new.strip()
    if not new or new.lower() == config.NEUTRAL_LABEL.lower():
        raise ValueError("Choose another new name.")
    table = _read_table()
    if table is None or old not in set(table["gesture"].str.strip()):
        raise ValueError(f"'{old}' is not in {config.SIGN_GUIDE_CSV.name}.")
    if new in set(table["gesture"].str.strip()):
        raise ValueError(f"'{new}' already exists in {config.SIGN_GUIDE_CSV.name}.")
    table["gesture"] = [new if g.strip() == old else g for g in table["gesture"]]
    _write_table(table)
    renamed = {(new if key == old else key): value for key, value in config.GESTURES.items()}
    config.GESTURES.clear()
    config.GESTURES.update(renamed)
    photo = find_sign_image(old)
    if photo and not find_sign_image(new):
        photo.rename(photo.with_name(_file_stem(new) + photo.suffix))
