"""Real-Time Indian Sign Language -> Text and Voice.

Run with:   streamlit run app.py
"""
from __future__ import annotations

import html
import time
from collections import deque
from time import perf_counter

import cv2
import pandas as pd
import streamlit as st

from isl import config
from isl import database as db
from isl import guide
from isl.camera import open_camera
from isl.hand_tracker import HandTracker
from isl.pipeline import LivePipeline
from isl.recognizer import GestureRecognizer
from isl.speaker import Speaker

st.set_page_config(page_title="ISL to Text and Voice", page_icon="🤟", layout="wide")

st.markdown(
    """
    <style>
    .gesture-card {border: 1px solid rgba(128,128,128,.35); border-radius: 10px;
                   padding: 18px 22px; background: rgba(128,128,128,.10);}
    .gesture-label {font-size: 2.6rem; font-weight: 700; line-height: 1.15;}
    .gesture-sub {opacity: .7; font-size: .95rem; margin-top: 4px;}
    .sentence-box {border: 1px solid rgba(128,128,128,.35); border-radius: 10px;
                   padding: 14px 20px; font-size: 1.5rem; min-height: 64px;}
    </style>
    """,
    unsafe_allow_html=True,
)

guide.ensure_guide_csv()
config.reload_gestures()          # picks up signs added/removed in assets/sign_guide.csv
db.init_db()
db.seed_gestures(config.GESTURES)


# --------------------------------------------------------------- cached objects
@st.cache_resource
def get_speaker() -> Speaker:
    return Speaker()


@st.cache_resource
def load_recognizer(path: str, modified: float) -> GestureRecognizer:
    """`modified` is part of the cache key, so a re-trained model is picked up."""
    return GestureRecognizer(path)


def model_ready() -> bool:
    return config.MODEL_PATH.exists()


# ----------------------------------------------------------------- session state
for key, default in {"running": False, "sentence": [], "session_id": None, "flash": None}.items():
    st.session_state.setdefault(key, default)


def start_cb() -> None:
    name = (st.session_state.get("user_name") or "Guest").strip() or "Guest"
    st.session_state.session_id = db.start_session(db.get_or_create_user(name))
    st.session_state.running = True


def stop_cb() -> None:
    if st.session_state.session_id:
        db.end_session(st.session_state.session_id)
    st.session_state.session_id = None
    st.session_state.running = False


def clear_text_cb() -> None:
    st.session_state.sentence = []


def fail(message: str) -> None:
    """Stop the live run, remember the message and redraw the page."""
    stop_cb()
    st.session_state.flash = message
    st.rerun()


# ------------------------------------------------------------------ small renderers
def render_sentence(placeholder) -> None:
    text = " ".join(st.session_state.sentence)
    body = html.escape(text) if text else '<span style="opacity:.5">Recognised words appear here</span>'
    placeholder.markdown(f'<div class="sentence-box">{body}</div>', unsafe_allow_html=True)


def render_gesture(placeholder, label, confidence, threshold) -> None:
    if label is None:
        title, sub = "No hand", "Show your hand to the camera"
    elif label == config.NEUTRAL_LABEL:
        title, sub = "No sign", "Hand is relaxed"
    else:
        title = config.GESTURES.get(label, label)
        state = "recognised" if confidence >= threshold else "not sure yet"
        sub = f"{confidence * 100:.0f}% confident, {state}"
    placeholder.markdown(
        f'<div class="gesture-card"><div class="gesture-label">{html.escape(title)}</div>'
        f'<div class="gesture-sub">{html.escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------- sidebar
locked = st.session_state.running
speaker = get_speaker()
with st.sidebar:
    st.header("Settings")
    st.text_input("User name", value="Guest", key="user_name", disabled=locked)
    threshold = st.slider("Confidence threshold", 0.50, 0.99, config.CONFIDENCE_THRESHOLD, 0.01, disabled=locked)
    stable_frames = st.slider("Frames to hold a sign", 3, 20, config.STABLE_FRAMES, disabled=locked)
    voice_on = st.toggle("Voice output", value=True, disabled=locked)
    speech_rate = st.slider("Speech rate", 100, 250, config.TTS_RATE, 10, disabled=locked)
    show_landmarks = st.toggle("Show hand landmarks", value=True, disabled=locked)
    camera_index = st.number_input("Camera index", 0, 5, config.CAMERA_INDEX, disabled=locked)
    speaker.rate = speech_rate
    st.divider()
    st.caption(speaker.diagnose())
    st.button("Test voice", on_click=speaker.speak, args=("Voice test. Hello, this is Indian Sign Language.",),
              disabled=locked)
    st.caption("Settings are locked while the camera is running. Press Stop to change them.")

# ---------------------------------------------------------------------------- layout
st.title("Indian Sign Language to text and voice")
tab_live, tab_guide, tab_history, tab_model = st.tabs(
    ["Live translation", "Sign guide", "History", "Model and gestures"])

with tab_live:
    if st.session_state.flash:
        st.error(st.session_state.flash)
        st.session_state.flash = None
    if not model_ready():
        st.warning("No trained model yet. Run  python collect_data.py  and then  python train_model.py,  "
                   "then reload this page.")
    else:
        model_signs = set(load_recognizer(str(config.MODEL_PATH), config.MODEL_PATH.stat().st_mtime).classes)
        model_signs -= {config.NEUTRAL_LABEL}
        not_trained = sorted(set(config.GESTURES) - model_signs)
        old_signs = sorted(model_signs - set(config.GESTURES))
        if not_trained or old_signs:
            st.warning("The sign list changed after the model was trained, so train again "
                       "(START_HERE.bat, option 4; record new signs with option 7 first). "
                       + (f"Not in the model yet: {', '.join(not_trained)}. " if not_trained else "")
                       + (f"Still in the model but removed from the list (ignored): {', '.join(old_signs)}."
                          if old_signs else ""))

    b_start, b_stop, b_status = st.columns([1, 1, 4])
    b_start.button("Start", on_click=start_cb, type="primary",
                   disabled=st.session_state.running or not model_ready())
    b_stop.button("Stop", on_click=stop_cb, disabled=not st.session_state.running)
    status_ph = b_status.empty()

    col_video, col_text = st.columns([3, 2])
    with col_video:
        video_ph = st.empty()
    with col_text:
        st.subheader("Detected sign")
        gesture_ph = st.empty()
        stats_ph = st.empty()
        st.subheader("Translated text")
        sentence_ph = st.empty()
        st.button("Clear text", on_click=clear_text_cb)

with tab_guide:
    st.subheader("Sign guide")
    st.caption("The key pose the app expects for each sign. The list of signs is assets/sign_guide.csv "
               "(one row per sign: gesture, spoken_text, description). Edit it in Excel and reload this page. "
               "Photos are saved by collect_data.py, or add your own as assets/signs/<gesture>.jpg.")
    descriptions = guide.load_descriptions()
    recorded = (pd.read_csv(config.DATASET_CSV, usecols=["label"])["label"].value_counts()
                if config.DATASET_CSV.exists() else pd.Series(dtype=int))
    columns = st.columns(3)
    for index, (label, spoken) in enumerate(config.GESTURES.items()):
        with columns[index % 3]:
            with st.container(border=True):
                photo = guide.find_sign_image(label)
                if photo:
                    st.image(str(photo))
                else:
                    st.caption("No photo yet. Record this sign with collect_data.py.")
                st.markdown(f"**{label}**")
                st.caption(f"Shown and spoken as: {spoken}")
                st.write(descriptions.get(label) or "No description yet. Add one in assets/sign_guide.csv.")
                st.caption(f"{int(recorded.get(label, 0))} samples recorded")

with tab_history:
    history = db.get_history(500)
    if history.empty:
        st.info("Nothing translated yet. Start the camera and show a sign.")
    else:
        st.dataframe(history)
        st.download_button("Download history as CSV", history.to_csv(index=False),
                           file_name="isl_translation_history.csv", mime="text/csv")
    st.button("Clear history", on_click=db.clear_history)

with tab_model:
    if model_ready():
        info = load_recognizer(str(config.MODEL_PATH), config.MODEL_PATH.stat().st_mtime).meta
        st.info(f"Model in use: {info.get('model_name', 'unknown')}, "
                f"test accuracy {info.get('test_accuracy', 0) * 100:.1f}% "
                f"({info.get('split', 'unknown')} split, {info.get('n_samples', '?')} samples)")
    st.subheader("Supported gestures")
    st.dataframe(pd.DataFrame({"Gesture": list(config.GESTURES),
                               "Shown and spoken as": list(config.GESTURES.values())}))
    comparison = config.REPORT_DIR / "model_comparison.csv"
    if comparison.exists():
        st.subheader("Model comparison (test set)")
        st.dataframe(pd.read_csv(comparison))
    strict_table = config.REPORT_DIR / "session_split" / "model_comparison.csv"
    if strict_table.exists():
        st.subheader("Stricter test (recording sessions the model has never seen)")
        st.dataframe(pd.read_csv(strict_table))
    matrix = config.REPORT_DIR / "confusion_matrix.png"
    if matrix.exists():
        st.image(str(matrix), caption="Confusion matrix of the selected model")
    if config.DATASET_CSV.exists():
        st.subheader("Samples per gesture")
        st.bar_chart(pd.read_csv(config.DATASET_CSV, usecols=["label"])["label"].value_counts())


# --------------------------------------------------------------------- live loop
def run_live() -> None:
    recognizer = load_recognizer(str(config.MODEL_PATH), config.MODEL_PATH.stat().st_mtime)
    speaker = get_speaker()
    speaker.rate = speech_rate

    try:
        tracker = HandTracker()
    except Exception as exc:
        fail(f"The hand detector could not start: {exc}")
        return
    cap = open_camera(camera_index)
    if not cap.isOpened():
        tracker.close()
        fail("The camera could not be opened. Close other apps that use it, "
             "or try another camera index in the sidebar.")
        return

    pipeline = LivePipeline(recognizer, tracker, threshold, stable_frames, show_landmarks)
    latencies: deque = deque(maxlen=30)
    status_ph.success("Camera is on. Show a sign to the camera.")
    render_sentence(sentence_ph)
    fps, last_tick, frame_no, read_failures = 0.0, perf_counter(), 0, 0

    try:
        while True:        # Streamlit ends this loop by interrupting the script when you press Stop
            ok, raw = cap.read()
            if not ok:
                read_failures += 1
                if read_failures > 30:
                    fail("The camera stopped sending frames.")
                    return
                time.sleep(0.03)
                continue
            read_failures = 0

            result = pipeline.process(raw)
            frame_no += 1
            latencies.append(result.latency_ms)
            now = perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last_tick, 1e-6) if frame_no > 1 else 0.0
            last_tick = now

            frame = result.frame
            cv2.putText(frame, f"{fps:.0f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if result.label and result.label != config.NEUTRAL_LABEL:
                good = result.confidence >= threshold
                cv2.putText(frame, f"{result.label} {result.confidence * 100:.0f}%",
                            (10, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 255) if good else (0, 165, 255), 2)
            video_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            if result.committed:
                text = config.GESTURES.get(result.committed, result.committed)
                st.session_state.sentence.append(text)
                db.add_translation(st.session_state.session_id, result.committed, text, result.confidence)
                if voice_on:
                    speaker.speak(text)
                render_sentence(sentence_ph)
                render_gesture(gesture_ph, result.committed, result.confidence, threshold)
            elif frame_no % 3 == 0:
                render_gesture(gesture_ph, result.label, result.confidence, threshold)
            if frame_no % 15 == 0:
                stats_ph.caption(f"{fps:.0f} frames/s, average recognition time "
                                 f"{sum(latencies) / len(latencies):.1f} ms")
    finally:
        cap.release()
        pipeline.close()


if st.session_state.running and model_ready():
    run_live()
else:
    video_ph.info("The camera is off. Press Start to begin.")
    render_gesture(gesture_ph, None, 0.0, threshold)
    render_sentence(sentence_ph)
