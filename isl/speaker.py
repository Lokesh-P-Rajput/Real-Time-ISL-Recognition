"""Offline text-to-speech.

The speech engine (pyttsx3) runs in its own process (isl/tts_worker.py), so a problem
with Windows speech can never freeze the video or the app. `speak()` returns immediately.
If the voice does not work, `diagnose()` tells why and data/tts_log.txt has the details.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

from . import config


class Speaker:
    def __init__(self, rate: int | None = None, volume: float | None = None, voice: str | None = None,
                 enabled: bool = True, extra_env: dict | None = None):
        self.rate = config.TTS_RATE if rate is None else rate
        self.volume = config.TTS_VOLUME if volume is None else volume
        self.voice = config.TTS_VOICE_CONTAINS if voice is None else voice
        self.enabled = enabled
        self._extra_env = extra_env or {}
        self._proc: subprocess.Popen | None = None
        self._log = None
        self._started_at = 0.0
        self._quick_failures = 0
        self._lock = threading.Lock()
        self.log_path = config.DATA_DIR / "tts_log.txt"
        if self.enabled:
            with self._lock:
                self._start()

    # ------------------------------------------------------------ process handling
    @property
    def available(self) -> bool:
        return bool(self.enabled and self._proc is not None and self._proc.poll() is None)

    def _start(self) -> None:
        if self._quick_failures >= 2:                     # keeps failing right away -> stop trying
            return
        self._close_log()
        try:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_path, "w", encoding="utf-8")
            command = [sys.executable, "-m", "isl.tts_worker"]
            if self.voice:
                command += ["--voice", self.voice]
            self._proc = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log,
                text=True, encoding="utf-8", cwd=str(config.BASE_DIR),
                env={**os.environ, "PYTHONIOENCODING": "utf-8", **self._extra_env},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._started_at = time.monotonic()
        except Exception as exc:
            self._proc = None
            self._quick_failures += 1
            if self._log:
                self._log.write(f"Could not start the voice process: {exc}\n")
                self._log.flush()

    def _drop_proc(self) -> None:
        """Forget a dead voice process and release its pipe."""
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        self._proc = None

    def _close_log(self) -> None:
        if self._log:
            try:
                self._log.close()
            except Exception:
                pass
            self._log = None

    # ------------------------------------------------------------------- public
    def speak(self, text: str) -> None:
        """Queue a sentence. Never blocks and never raises."""
        if not self.enabled or not text:
            return
        clean = " ".join(str(text).replace("|", " ").split())
        line = f"{int(self.rate)}|{float(self.volume)}|{clean}\n"
        with self._lock:
            for _ in range(2):
                if self._proc is None or self._proc.poll() is not None:
                    if self._proc is not None and time.monotonic() - self._started_at < 3:
                        self._quick_failures += 1         # it died almost immediately
                    self._drop_proc()
                    self._start()
                if self._proc is None:
                    return
                try:
                    self._proc.stdin.write(line)
                    self._proc.stdin.flush()
                    return
                except (OSError, ValueError):
                    self._drop_proc()

    def diagnose(self) -> str:
        """One short sentence for the user: is the voice engine running, and if not, why."""
        if not self.enabled:
            return "Voice is switched off."
        if self.available:
            return "Voice engine is running. If you hear nothing, check the volume and Windows Speech settings."
        reason = ""
        try:
            lines = [ln.strip() for ln in self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                     if ln.strip()]
            reason = lines[-1] if lines else ""
        except OSError:
            pass
        return "Voice engine is NOT running. " + (reason or "Is pyttsx3 installed? Run check_setup.py.")

    def close(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.stdin.close()
                    self._proc.wait(timeout=3)
                except Exception:
                    self._proc.kill()
            self._drop_proc()
            self._close_log()

    stop = close
