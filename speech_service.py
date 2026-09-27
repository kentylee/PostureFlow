"""
Reusable speech-to-text service for PostureFlow.

This module handles:
- microphone validation
- short audio recording
- WAV creation
- faster-whisper transcription
- basic recording diagnostics

The service itself is blocking.
PostureFlow will run it inside a worker thread so that
the webcam loop remains responsive.
"""

import os
import time
import wave
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


@dataclass
class SpeechResult:
    audio_path: str
    transcript: str
    transcription_seconds: float
    peak_level: float
    rms_level: float
    device_name: str


class SpeechService:
    def __init__(
        self,
        model_size="base.en",
        input_device=None,
        sample_rate=16000,
        channels=1,
        use_vad=True,
        audio_dir="audio_tests",
    ):
        self.model_size = model_size
        self.input_device = input_device
        self.sample_rate = sample_rate
        self.channels = channels
        self.use_vad = use_vad
        self.audio_dir = audio_dir

        self.model = None
        self.model_load_seconds = None

    def get_device_info(self):
        """
        Return information about the currently selected microphone.
        """
        return sd.query_devices(self.input_device, "input")

    def validate_input(self):
        """
        Confirm that the selected microphone supports the requested
        recording configuration.
        """
        sd.check_input_settings(
            device=self.input_device,
            channels=self.channels,
            dtype="int16",
            samplerate=self.sample_rate,
        )

        return True

    def load_model(self):
        """
        Load Whisper once.

        Calling this again does not reload the model.
        """
        if self.model is not None:
            return self.model_load_seconds

        start = time.perf_counter()

        self.model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8",
        )

        self.model_load_seconds = time.perf_counter() - start

        return self.model_load_seconds

    def record(self, seconds=5):
        """
        Record one microphone utterance and save it as a WAV file.
        """

        self.validate_input()

        os.makedirs(self.audio_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        filename = f"postureflow_speech_{timestamp}.wav"
        filepath = os.path.join(self.audio_dir, filename)

        audio = sd.rec(
            int(seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            device=self.input_device,
        )

        sd.wait()

        audio_float = audio.astype(np.float32)

        peak = np.max(np.abs(audio_float)) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2)) / 32768.0

        with wave.open(filepath, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio.tobytes())

        return filepath, peak, rms

    def transcribe(self, filepath):
        """
        Transcribe one existing WAV file.
        """

        if self.model is None:
            raise RuntimeError(
                "Whisper model has not been loaded. "
                "Call load_model() first."
            )

        start = time.perf_counter()

        segments, _ = self.model.transcribe(
            filepath,
            language="en",
            vad_filter=self.use_vad,
        )

        transcript_parts = []

        for segment in segments:
            text = segment.text.strip()

            if text:
                transcript_parts.append(text)

        transcript = " ".join(transcript_parts).strip()

        transcription_seconds = time.perf_counter() - start

        return transcript, transcription_seconds

    def record_and_transcribe(self, seconds=5):
        """
        Perform one complete speech-input operation.
        """

        device_info = self.get_device_info()

        filepath, peak, rms = self.record(seconds)

        transcript, transcription_seconds = self.transcribe(filepath)

        return SpeechResult(
            audio_path=filepath,
            transcript=transcript,
            transcription_seconds=transcription_seconds,
            peak_level=peak,
            rms_level=rms,
            device_name=device_info["name"],
        )