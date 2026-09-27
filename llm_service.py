"""
Local LLM service for PostureFlow.

Purpose:
Convert a confirmed multimodal PostureFlow record into a short,
structured, non-diagnostic explanation using a local Ollama model.

The LLM is not responsible for emergency stopping.
Safety-critical stop handling occurs before this service is called.
"""

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass


@dataclass
class LLMResult:
    status: str
    feedback: str
    used_pose_metrics: bool
    generation_seconds: float
    model: str


class LLMService:
    def __init__(
        self,
        model="qwen3:4b-instruct",
        base_url="http://localhost:11434",
        timeout=45,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate_feedback(self, multimodal_record):
        """
        Generate one bounded posture-awareness response.

        Safety stop conditions should already have been handled
        outside the LLM.
        """

        safety_status = multimodal_record.get(
            "safety_status",
            "unknown",
        )

        # Do not wait for an LLM when the deterministic safety layer has already indicated a stop condition.
        if safety_status == "stop":
            return LLMResult(
                status="stop",
                feedback=(
                    "Coaching is paused. Stop the movement if you "
                    "feel pain or discomfort."
                ),
                used_pose_metrics=False,
                generation_seconds=0.0,
                model="deterministic_safety_layer",
            )

        output_schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": [
                        "ok",
                        "tracking_limited",
                    ],
                },
                "feedback": {
                    "type": "string",
                },
                "used_pose_metrics": {
                    "type": "boolean",
                },
            },
            "required": [
                "status",
                "feedback",
                "used_pose_metrics",
            ],
            "additionalProperties": False,
        }

        system_prompt = """
You are the explanation component of PostureFlow, an educational
non-diagnostic posture-awareness prototype.

You receive:
1. measured pose information from MediaPipe,
2. deterministic rule-based feedback,
3. tracking quality,
4. a user-confirmed speech transcript.

Rules:
- Only discuss information explicitly supplied in the record.
- Do not invent body measurements or observations.
- Do not diagnose any condition or injury.
- Do not provide medical treatment.
- Do not claim that posture is medically correct or incorrect.
- Do not claim to replace a physiotherapist or clinician.
- Keep the response brief and understandable.
- Prefer one or two short sentences.
- The user's transcript is a user-reported statement, not a
  measured physical fact.
- The rule-based feedback is an application observation, not a
  clinical diagnosis.
- Never describe the user's overall posture as "good", "bad",
  "correct", "incorrect", "healthy", or "unhealthy".
- Describe only the specific supplied observation or metric.
- Prefer wording such as:
  "Your shoulders appeared level in this observation."
- Do not generalise one measured feature into an assessment of
  overall posture.
  

Tracking rule:
- If tracking_status is not "valid", set status to
  "tracking_limited".
- When tracking is limited, do not interpret pose metric values and
  set used_pose_metrics to false.
- Tell the user that the pose cannot currently be assessed reliably
  and that they should reposition so the required landmarks are
  visible.

If tracking_status is "valid":
- status may be "ok".
- You may explain the supplied pose metrics and rule feedback.
- Do not infer information that is absent.

Return only the requested structured JSON.
""".strip()

        record_json = json.dumps(
            multimodal_record,
            ensure_ascii=False,
            indent=2,
        )

        user_prompt = (
            "Generate a brief PostureFlow response for this "
            "confirmed multimodal observation:\n\n"
            f"{record_json}"
        )

        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": "5m",
            "format": output_schema,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "options": {
                "temperature": 0.2,
            },
        }

        request_data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=request_data,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        start = time.perf_counter()

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                response_body = response.read().decode("utf-8")

        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")

            raise RuntimeError(
                f"Ollama API returned HTTP {error.code}: {error_body}"
            ) from error

        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Could not contact Ollama: {error}"
            ) from error

        generation_seconds = time.perf_counter() - start

        api_response = json.loads(response_body)

        if "message" not in api_response:
            raise RuntimeError(
                "Ollama response did not contain a message."
            )

        content = api_response["message"].get(
            "content",
            "",
        )

        if not content:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        try:
            parsed = json.loads(content)

        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Ollama returned invalid JSON."
            ) from error

        self._validate_output(
            parsed,
            multimodal_record,
        )

        return LLMResult(
            status=parsed["status"],
            feedback=parsed["feedback"].strip(),
            used_pose_metrics=parsed["used_pose_metrics"],
            generation_seconds=generation_seconds,
            model=self.model,
        )

    def _validate_output(
        self,
        output,
        multimodal_record,
    ):
        """
        Validate important application-level constraints in addition
        to the JSON schema returned by Ollama.
        """

        required_fields = {
            "status",
            "feedback",
            "used_pose_metrics",
        }

        if set(output.keys()) != required_fields:
            raise ValueError(
                "Unexpected fields in LLM response."
            )

        if output["status"] not in {
            "ok",
            "tracking_limited",
        }:
            raise ValueError(
                "Invalid LLM status."
            )

        if not isinstance(output["feedback"], str):
            raise ValueError(
                "LLM feedback must be a string."
            )

        if not output["feedback"].strip():
            raise ValueError(
                "LLM feedback cannot be empty."
            )

        if len(output["feedback"]) > 500:
            raise ValueError(
                "LLM feedback exceeded the allowed length."
            )

        if not isinstance(
            output["used_pose_metrics"],
            bool,
        ):
            raise ValueError(
                "used_pose_metrics must be boolean."
            )

        tracking_status = (
            multimodal_record
            .get("pose", {})
            .get("tracking_status")
        )

        if (
            tracking_status != "valid"
            and output["used_pose_metrics"]
        ):
            raise ValueError(
                "LLM attempted to use pose metrics while "
                "tracking was not valid."
            )