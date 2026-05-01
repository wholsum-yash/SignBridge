import pyttsx3
import threading
import time


class TTS:
    """
    Thread-safe Text-to-Speech module for SignBridge

    Features:
    - Non-blocking speech (no FPS drop)
    - Duplicate prevention
    - Cooldown protection
    """

    def __init__(self, rate=170, volume=1.0, cooldown=1.2):
        self.engine = pyttsx3.init()

        # voice settings
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        # state control
        self.last_spoken = None
        self.last_time = 0
        self.cooldown = cooldown  # seconds

        self.lock = threading.Lock()

    def _speak_blocking(self, text):
        """Internal blocking call (runs in thread)"""
        self.engine.say(text)
        self.engine.runAndWait()

    def speak(self, text):
        """
        Safe public call

        Conditions:
        - avoids repeating same word
        - enforces cooldown
        - runs async (non-blocking)
        """

        current_time = time.time()

        # duplicate filter 
        if text == self.last_spoken:
            return

        # COOLDOWN 
        if current_time - self.last_time < self.cooldown:
            return

        self.last_spoken = text
        self.last_time = current_time

        # thread safe execution 
        thread = threading.Thread(
            target=self._thread_wrapper,
            args=(text,),
            daemon=True
        )
        thread.start()

    def _thread_wrapper(self, text):
        """Ensures only one TTS runs at a time"""
        with self.lock:
            self._speak_blocking(text)

    def reset(self):
        """Manual reset if needed"""
        self.last_spoken = None
        self.last_time = 0
