"""Text-to-speech worker. It runs as its own small process:  python -m isl.tts_worker

Why a separate process? Windows speech (SAPI/COM) must run on a normal main thread. Inside
Streamlit's background threads it often fails silently. Here pyttsx3 always runs on the
main thread of its own process, and the app just sends it text.

Protocol: one request per line on stdin:   rate|volume|text
Options:  --voice NAME   use the installed voice whose name contains NAME
          --list         print the installed voices and exit
"""
from __future__ import annotations

import sys


def pick_voice(engine, wanted: str) -> None:
    if not wanted:
        return
    for voice in engine.getProperty("voices"):
        if wanted.lower() in (voice.name or "").lower() or wanted.lower() in (voice.id or "").lower():
            engine.setProperty("voice", voice.id)
            return
    print(f"[tts] no installed voice contains '{wanted}', using the default voice", file=sys.stderr, flush=True)


def main(argv: list) -> int:
    import pyttsx3

    engine = pyttsx3.init()
    if "--list" in argv:
        for index, voice in enumerate(engine.getProperty("voices")):
            print(f"{index}: {voice.name}")
        return 0

    voice = argv[argv.index("--voice") + 1] if "--voice" in argv and argv.index("--voice") + 1 < len(argv) else ""
    pick_voice(engine, voice)
    print("[tts] ready", file=sys.stderr, flush=True)

    while True:
        line = sys.stdin.readline()
        if not line:                       # the app closed the pipe -> quit
            return 0
        line = line.rstrip("\r\n")
        if not line:
            continue
        try:
            rate, volume, text = line.split("|", 2)
            engine.setProperty("rate", int(float(rate)))
            engine.setProperty("volume", float(volume))
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            print(f"[tts] could not speak {line!r}: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    try:
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main(sys.argv[1:]))
