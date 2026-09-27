from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_service import LLMService

def main():
    record = {
        "request_id": "424bf0d8",
        "session_id": "19c07bb0",

        "pose": {
            "mode": "Shoulder Alignment Check",
            "tracking_status": "valid",
            "metrics": {
                "shoulder_diff": 0.029,
            },
            "rule_feedback":
                "Shoulders appear level in this observation.",
        },

        "speech": {
            "raw_transcript":
                "Let's see, my shoulders feel fine.",
            "confirmed_transcript":
                "Let's see, my shoulders feel fine.",
        },

        "safety_status": "clear",
    }

    service = LLMService(
        model="qwen3:4b-instruct",
    )

    print("Sending confirmed multimodal record to Ollama...")

    result = service.generate_feedback(record)

    print("\n--- LLM Result ---")
    print(f"Model: {result.model}")
    print(f"Status: {result.status}")
    print(
        f"Used pose metrics: "
        f"{result.used_pose_metrics}"
    )
    print(f"Feedback: {result.feedback}")
    print(
        f"Generation time: "
        f"{result.generation_seconds:.2f}s"
    )
    print("------------------")


if __name__ == "__main__":
    main()