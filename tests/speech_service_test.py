from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from speech_service import SpeechService


def main():
    service = SpeechService(
        model_size="base.en",
        input_device=None,
    )

    print("Selected microphone:")
    print(service.get_device_info()["name"])

    print("\nLoading Whisper...")

    load_time = service.load_model()

    print(f"Model loaded in {load_time:.2f} seconds.")

    input("\nPress Enter when ready to speak...")

    print("Recording...")

    result = service.record_and_transcribe(seconds=5)

    print("\n--- Result ---")
    print(f"Device: {result.device_name}")
    print(f"Transcript: {result.transcript}")
    print(f"Transcription time: {result.transcription_seconds:.2f}s")
    print(f"Peak: {result.peak_level:.4f}")
    print(f"RMS: {result.rms_level:.4f}")
    print(f"Audio: {result.audio_path}")


if __name__ == "__main__":
    main()