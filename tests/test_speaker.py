import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from isl import config
from isl.speaker import Speaker

# A fake "pyttsx3" that writes what it would say into a file instead of making sound.
FAKE_PYTTSX3 = '''
import os
OUT = os.environ["STUB_TTS_OUT"]

class Engine:
    def __init__(self):
        self.props, self.text = {}, ""
    def setProperty(self, key, value):
        self.props[key] = value
    def getProperty(self, key):
        return []
    def say(self, text):
        self.text = text
    def runAndWait(self):
        with open(OUT, "a", encoding="utf-8") as handle:
            handle.write("%s|%s|%s\\n" % (self.props.get("rate"), self.props.get("volume"), self.text))

def init():
    if os.environ.get("STUB_TTS_FAIL"):
        raise RuntimeError("no speech driver found")
    return Engine()
'''


class SpeakerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "pyttsx3.py").write_text(FAKE_PYTTSX3, encoding="utf-8")
        self.out = self.root / "spoken.txt"
        patcher = mock.patch.object(config, "DATA_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def env(self, **extra):
        return {"PYTHONPATH": str(self.root), "STUB_TTS_OUT": str(self.out), **extra}

    def wait_for(self, condition, seconds=15):
        end = time.time() + seconds
        while time.time() < end:
            if condition():
                return True
            time.sleep(0.1)
        return False

    def spoken(self):
        return self.out.read_text(encoding="utf-8").splitlines() if self.out.exists() else []

    def test_disabled_speaker_is_silent_and_safe(self):
        speaker = Speaker(enabled=False)
        self.assertFalse(speaker.available)
        speaker.speak("Hello")
        speaker.speak("")
        self.assertIn("switched off", speaker.diagnose())

    def test_sentences_reach_the_voice_process_in_order_with_current_rate(self):
        speaker = Speaker(rate=170, extra_env=self.env())
        self.addCleanup(speaker.close)
        speaker.speak("Hello")
        speaker.rate = 120
        speaker.speak("Thank  you | very much\n")          # spaces, pipe and newline are cleaned
        self.assertTrue(self.wait_for(lambda: len(self.spoken()) == 2), self.spoken())
        self.assertEqual(self.spoken(), ["170|1.0|Hello", "120|1.0|Thank you very much"])
        self.assertTrue(speaker.available)
        self.assertIn("running", speaker.diagnose())

    def test_broken_speech_engine_is_reported_and_never_crashes_the_app(self):
        speaker = Speaker(extra_env=self.env(STUB_TTS_FAIL="1"))
        self.addCleanup(speaker.close)
        for _ in range(4):
            speaker.speak("Hello")
            time.sleep(0.5)
        self.assertTrue(self.wait_for(lambda: not speaker.available))
        message = speaker.diagnose()
        self.assertIn("NOT running", message)
        self.assertIn("no speech driver", message)

    def test_voice_process_restarts_if_it_dies(self):
        speaker = Speaker(extra_env=self.env())
        self.addCleanup(speaker.close)
        speaker.speak("one")
        self.assertTrue(self.wait_for(lambda: len(self.spoken()) == 1))
        speaker._proc.kill()
        speaker._proc.wait()
        speaker._started_at -= 60                          # it did not die "immediately"
        speaker.speak("two")
        self.assertTrue(self.wait_for(lambda: len(self.spoken()) == 2), self.spoken())


if __name__ == "__main__":
    unittest.main()
