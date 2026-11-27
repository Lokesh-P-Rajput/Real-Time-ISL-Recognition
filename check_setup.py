"""STEP 0 - Check that everything needed by the project works on this computer.

    python check_setup.py
"""
from __future__ import annotations

import importlib
import sys
import time

PASS, FAIL = "[ OK ]", "[FAIL]"
problems = 0


def report(ok: bool, name: str, detail: str = "") -> None:
    global problems
    problems += 0 if ok else 1
    print(f"{PASS if ok else FAIL} {name}" + (f" - {detail}" if detail else ""))


print(f"Python {sys.version.split()[0]}")
report((3, 9) <= sys.version_info[:2] <= (3, 12), "Python version supported by MediaPipe (3.9 to 3.12)",
       "" if sys.version_info[:2] == (3, 11) else "this project is built and tested for 3.11")

for module, pip_name in [("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"),
                         ("joblib", "joblib"), ("matplotlib", "matplotlib"), ("cv2", "opencv-contrib-python"),
                         ("mediapipe", "mediapipe"), ("xgboost", "xgboost"), ("streamlit", "streamlit"),
                         ("pyttsx3", "pyttsx3")]:
    try:
        mod = importlib.import_module(module)
        report(True, module, getattr(mod, "__version__", ""))
    except Exception as exc:
        hint = f"{type(exc).__name__}: run  pip install {pip_name}"
        if module == "xgboost":
            hint += "  (Windows also needs the Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe)"
        report(False, module, hint)

try:
    from mediapipe.tasks.python import vision
    report(hasattr(vision, "HandLandmarker"), "MediaPipe HandLandmarker (Tasks API)")
except Exception as exc:
    report(False, "MediaPipe HandLandmarker (Tasks API)", str(exc))

try:
    from isl.hand_tracker import HandTracker, ensure_hand_model
    ensure_hand_model()
    report(True, "MediaPipe hand model file")
    tracker = HandTracker()
    tracker.close()
    report(True, "Hand detector starts")
except Exception as exc:
    report(False, "Hand detector", str(exc))

try:
    from isl.camera import open_camera
    cap = open_camera()
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    cap.release()
    report(ok, "Webcam", f"frame {frame.shape[1]}x{frame.shape[0]}" if ok else
           "cannot read a frame - close other apps using the camera or change CAMERA_INDEX in isl/config.py")
except Exception as exc:
    report(False, "Webcam", str(exc))

try:
    from isl.speaker import Speaker
    speaker = Speaker()
    speaker.speak("Setup test successful")
    time.sleep(4)
    report(speaker.available, "Text-to-speech", "you should have heard 'Setup test successful'"
           if speaker.available else speaker.diagnose())
    speaker.close()
    import subprocess
    voices = subprocess.run([sys.executable, "-m", "isl.tts_worker", "--list"], capture_output=True, text=True,
                            timeout=30).stdout.strip()
    print("       installed voices:" + ("\n       " + voices.replace("\n", "\n       ") if voices else " none found"))
    print("       (to use another voice, set TTS_VOICE_CONTAINS in isl/config.py, e.g. \"Zira\")")
except Exception as exc:
    report(False, "Text-to-speech", str(exc))

from isl import config  # noqa: E402

print(f"\nDataset: {'found' if config.DATASET_CSV.exists() else 'not created yet - run collect_data.py'}")
print(f"Model  : {'found' if config.MODEL_PATH.exists() else 'not trained yet - run train_model.py'}")
print("\nAll checks passed." if not problems else f"\n{problems} check(s) failed - fix them before continuing.")
