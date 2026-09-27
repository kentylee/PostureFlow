"""
PostureFlow - Multi-Modal Posture Awareness Prototype

Educational prototype for general posture awareness only.
It is not a medical device and does not diagnose, treat, or replace a clinician.
"""

import copy
import csv
import json
import math
import os
import queue
import threading
import time
import uuid
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

from llm_service import LLMService
from speech_service import SpeechService


# =============================================================================
# Configuration
# =============================================================================

CAMERA_INDEX = 0
MIRROR_DISPLAY = True

SPEECH_MODEL_SIZE = "base.en"
MICROPHONE_INDEX = None  # Use the current Windows default input device.
SPEECH_RECORD_SECONDS = 5

SHOULDER_LEVEL_THRESHOLD = 0.035
HEAD_OFFSET_THRESHOLD = 0.050
WRIST_LEVEL_THRESHOLD = 0.060
VISIBILITY_THRESHOLD = 0.50

SCREENSHOT_DIR = "screenshots"
LOG_FILE = "postureflow_multimodal_log.csv"

MODE_NAMES = {
    1: "Shoulder Alignment Check",
    2: "Head Position Check",
    3: "Bilateral Arm Raise Check",
}

STOP_TERMS = (
    "stop",
    "stop exercise",
    "stop the exercise",
    "please stop",
)

PAIN_TERMS = (
    "pain",
    "hurts",
    "hurt",
)

NEGATED_PAIN_TERMS = (
    "no pain",
    "not in pain",
    "do not have pain",
    "don't have pain",
)

CSV_FIELDS = [
    "timestamp",
    "session_id",
    "request_id",
    "mode",
    "tracking_status",
    "feedback_source",
    "rule_feedback",
    "metrics",
    "screenshot_filename",
    "raw_transcript",
    "confirmed_transcript",
    "safety_status",
    "ai_model",
    "ai_status",
    "ai_used_pose_metrics",
    "ai_feedback",
    "speech_seconds",
    "ai_generation_seconds",
]

# =============================================================================
# MediaPipe Setup
# =============================================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

FONT = cv2.FONT_HERSHEY_SIMPLEX

UI_WIDTH = 1440
UI_HEIGHT = 840
UI_BG = (12, 14, 15)
UI_SURFACE = (24, 27, 28)
UI_SURFACE_ALT = (31, 35, 36)
UI_SURFACE_SOFT = (19, 22, 23)
UI_BORDER = (54, 60, 62)
UI_TEXT = (245, 247, 248)
UI_MUTED = (157, 166, 169)
UI_DIM = (112, 120, 123)
UI_ACCENT = (118, 236, 170)
UI_CYAN = (232, 196, 92)
UI_SUCCESS = (112, 220, 118)
UI_WARNING = (86, 184, 244)
UI_DANGER = (96, 92, 240)

POSE_MODEL_LABEL = "MediaPipe Pose"
SPEECH_MODEL_LABEL = f"Whisper {SPEECH_MODEL_SIZE}"
AI_MODEL_LABEL = "Qwen3 4B"


# =============================================================================
# Camera and Hardware Utilities
# =============================================================================

def open_working_camera():
    """Open the configured camera, preferring DirectShow on Windows."""
    backends = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("DEFAULT", 0),
    ]

    for backend_name, backend in backends:
        print(f"Trying camera index {CAMERA_INDEX} with backend {backend_name}...")
        cap = (
            cv2.VideoCapture(CAMERA_INDEX)
            if backend == 0
            else cv2.VideoCapture(CAMERA_INDEX, backend)
        )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print(f"Failed to open: index {CAMERA_INDEX}, backend {backend_name}")
            cap.release()
            continue

        success, frame = cap.read()
        if success and frame is not None:
            print(
                f"Camera opened successfully: index {CAMERA_INDEX}, "
                f"backend {backend_name}"
            )
            return cap

        print(
            f"Opened but could not read frame: index {CAMERA_INDEX}, "
            f"backend {backend_name}"
        )
        cap.release()

    print("Error: No working webcam found.")
    return None


def required_landmarks_visible(landmarks, required_ids, threshold=VISIBILITY_THRESHOLD):
    """Reject missing, non-finite, out-of-frame, or low-visibility landmarks."""
    if landmarks is None:
        return False

    for landmark_id in required_ids:
        if landmark_id >= len(landmarks):
            return False

        landmark = landmarks[landmark_id]
        if not math.isfinite(landmark.x) or not math.isfinite(landmark.y):
            return False
        if not 0.0 <= landmark.x <= 1.0 or not 0.0 <= landmark.y <= 1.0:
            return False
        if landmark.visibility < threshold:
            return False

    return True


def safe_filename_part(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def save_screenshot(frame, identifier, mode_name, kind="request"):
    """Save an evidence or manual screenshot and return only its filename."""
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_mode = safe_filename_part(mode_name)
    filename = (
        f"postureflow_{kind}_{identifier}_{safe_mode}_{timestamp}.png"
    )
    filepath = os.path.join(SCREENSHOT_DIR, filename)

    if not cv2.imwrite(filepath, frame):
        print(f"Warning: screenshot could not be saved: {filepath}")
        return ""

    print(f"Screenshot saved: {filepath}")
    return filename


# =============================================================================
# Logging and Evidence Management
# =============================================================================

def create_log_file(path):
    with open(path, mode="w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(CSV_FIELDS)


def initialise_log():
    """Create a compatible log, using a new file if an old schema is detected."""
    if not os.path.exists(LOG_FILE):
        create_log_file(LOG_FILE)
        return LOG_FILE

    try:
        with open(LOG_FILE, mode="r", newline="", encoding="utf-8") as file:
            existing_header = next(csv.reader(file), [])
    except OSError as error:
        raise RuntimeError(f"Could not inspect log file: {error}") from error

    if existing_header == CSV_FIELDS:
        return LOG_FILE

    base, extension = os.path.splitext(LOG_FILE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = f"{base}_schema2_{timestamp}{extension or '.csv'}"
    create_log_file(versioned_path)
    print(
        "Warning: existing CSV header does not match the final multimodal schema. "
        f"Logging to: {versioned_path}"
    )
    return versioned_path


def log_multimodal_result(log_file, multimodal_record, ai_event):
    """Write one request-consistent multimodal observation."""
    if multimodal_record["request_id"] != ai_event["request_id"]:
        raise ValueError("Cannot log mismatched multimodal and AI request IDs.")

    pose = multimodal_record["pose"]
    speech = multimodal_record["speech"]

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": multimodal_record["session_id"],
        "request_id": multimodal_record["request_id"],
        "mode": pose["mode"],
        "tracking_status": pose["tracking_status"],
        "feedback_source": "multimodal_ai",
        "rule_feedback": pose["rule_feedback"],
        "metrics": json.dumps(pose["metrics"], ensure_ascii=False, sort_keys=True),
        "screenshot_filename": multimodal_record.get("screenshot_filename", ""),
        "raw_transcript": speech["raw_transcript"],
        "confirmed_transcript": speech["confirmed_transcript"],
        "safety_status": multimodal_record["safety_status"],
        "ai_model": ai_event["model"],
        "ai_status": ai_event["llm_status"],
        "ai_used_pose_metrics": ai_event["used_pose_metrics"],
        "ai_feedback": ai_event["feedback"],
        "speech_seconds": speech.get("transcription_seconds", ""),
        "ai_generation_seconds": ai_event["generation_seconds"],
    }

    with open(log_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writerow(row)

    print(f"Multimodal result logged: {multimodal_record['request_id']}")


# =============================================================================
# Background AI Workers
# =============================================================================

def speech_worker(speech_service, result_queue, request_context):
    """Record and transcribe without blocking the webcam loop."""
    request_id = request_context["request_id"]

    try:
        result_queue.put({
            "type": "status",
            "request_id": request_id,
            "status": "RECORDING",
        })
        audio_path, peak_level, rms_level = speech_service.record(
            seconds=SPEECH_RECORD_SECONDS
        )

        result_queue.put({
            "type": "status",
            "request_id": request_id,
            "status": "TRANSCRIBING",
        })
        transcript, transcription_seconds = speech_service.transcribe(audio_path)

        result_queue.put({
            "type": "result",
            "request_id": request_id,
            "context": request_context,
            "transcript": transcript,
            "audio_path": audio_path,
            "peak_level": peak_level,
            "rms_level": rms_level,
            "transcription_seconds": transcription_seconds,
        })
    except Exception as error:
        result_queue.put({
            "type": "error",
            "request_id": request_id,
            "context": request_context,
            "error": str(error),
        })


def llm_worker(llm_service, result_queue, multimodal_record):
    """Generate AI feedback without blocking the webcam loop."""
    request_id = multimodal_record["request_id"]

    try:
        result_queue.put({
            "type": "status",
            "request_id": request_id,
            "status": "GENERATING",
        })
        result = llm_service.generate_feedback(multimodal_record)
        result_queue.put({
            "type": "result",
            "request_id": request_id,
            "record": multimodal_record,
            "llm_status": result.status,
            "feedback": result.feedback,
            "used_pose_metrics": result.used_pose_metrics,
            "generation_seconds": result.generation_seconds,
            "model": result.model,
        })
    except Exception as error:
        result_queue.put({
            "type": "error",
            "request_id": request_id,
            "error": str(error),
        })

# =============================================================================
# Voice Command and Safety Logic
# =============================================================================

def detect_voice_command(transcript):
    """
    Detect simple deterministic voice commands.

    Returns:
        ("mode", 1/2/3)
        ("stop", None)
        ("resume", None)
        or None when the transcript should be treated as user feedback.
    """

    text = transcript.lower().strip()

    shoulder_phrases = (
        "shoulder mode",
        "switch to shoulder",
        "shoulder alignment",
        "check my shoulder",
        "check my shoulders",
        "select shoulder",
    )

    head_phrases = (
        "head mode",
        "switch to head",
        "head position",
        "check my head",
        "check my head position",
        "select head",
    )

    arm_phrases = (
        "arm mode",
        "switch to arm",
        "arm raise",
        "arm raise mode",
        "check my arms",
        "check my arm",
        "select arm",
    )

    stop_phrases = (
        "stop",
        "stop coaching",
        "stop the exercise",
        "pause coaching",
    )

    resume_phrases = (
        "resume",
        "resume coaching",
        "continue coaching",
    )

    if any(phrase in text for phrase in shoulder_phrases):
        return ("mode", 1)

    if any(phrase in text for phrase in head_phrases):
        return ("mode", 2)

    if any(phrase in text for phrase in arm_phrases):
        return ("mode", 3)

    if any(phrase in text for phrase in stop_phrases):
        return ("stop", None)

    if any(phrase in text for phrase in resume_phrases):
        return ("resume", None)

    return None

def assess_transcript_safety(transcript):
    """Apply deterministic stop/pain handling before any generative step."""
    text = transcript.lower().strip()
    if not text:
        return "empty"

    if any(phrase in text for phrase in NEGATED_PAIN_TERMS):
        return "clear"
    if any(phrase in text for phrase in STOP_TERMS):
        return "stop"
    if any(phrase in text for phrase in PAIN_TERMS):
        return "stop"
    return "clear"


# =============================================================================
# Pose Analysis
# =============================================================================

def analyse_shoulder_alignment(landmarks):
    required = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    ]
    if not required_landmarks_visible(landmarks, required):
        return (
            "Tracking issue: keep both shoulders clearly visible.",
            {"tracking": "low confidence"},
        )

    left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    difference = abs(left.y - right.y)

    feedback = (
        "Shoulders appear level in this observation."
        if difference < SHOULDER_LEVEL_THRESHOLD
        else "One shoulder appears higher. Try to keep both shoulders level."
    )
    return feedback, {"shoulder_diff": round(difference, 3)}


def analyse_head_position(landmarks):
    required = [
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    ]
    if not required_landmarks_visible(landmarks, required):
        return (
            "Tracking issue: keep your head and shoulders clearly visible.",
            {"tracking": "low confidence"},
        )

    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    shoulder_mid_x = (left.x + right.x) / 2
    shoulder_mid_y = (left.y + right.y) / 2
    head_offset = nose.x - shoulder_mid_x
    nose_to_shoulder_y = abs(nose.y - shoulder_mid_y)

    if nose_to_shoulder_y < 0.12:
        feedback = "Head appears lowered. Keep the movement gentle and the head upright."
    elif abs(head_offset) < HEAD_OFFSET_THRESHOLD:
        feedback = "Head appears centred over the shoulders."
    else:
        feedback = "Head appears off-centre. Try to keep it centred over the shoulders."

    return feedback, {
        "head_offset": round(head_offset, 3),
        "nose_to_shoulder_y": round(nose_to_shoulder_y, 3),
    }


def analyse_arm_raise(landmarks):
    required = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value,
    ]
    if not required_landmarks_visible(landmarks, required):
        return (
            "Tracking issue: keep shoulders, elbows, and wrists clearly visible.",
            {"tracking": "low confidence"},
        )

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    wrist_diff = abs(left_wrist.y - right_wrist.y)
    elbow_diff = abs(left_elbow.y - right_elbow.y)
    shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
    arms_raised = (
        left_wrist.y < left_shoulder.y
        and right_wrist.y < right_shoulder.y
    )

    if not arms_raised:
        feedback = "Raise both arms slowly until the wrists are above shoulder level."
    elif (
        wrist_diff < WRIST_LEVEL_THRESHOLD
        and shoulder_diff < SHOULDER_LEVEL_THRESHOLD
    ):
        feedback = "Both arms appear raised to a similar height with level shoulders."
    elif wrist_diff >= WRIST_LEVEL_THRESHOLD:
        feedback = "Try to raise both arms to a similar height."
    else:
        feedback = "Keep the shoulders level while raising the arms."

    return feedback, {
        "wrist_diff": round(wrist_diff, 3),
        "elbow_diff": round(elbow_diff, 3),
        "shoulder_diff": round(shoulder_diff, 3),
        "arms_raised": arms_raised,
    }


def analyse_current_mode(current_mode, landmarks):
    analysers = {
        1: analyse_shoulder_alignment,
        2: analyse_head_position,
        3: analyse_arm_raise,
    }
    analyser = analysers.get(current_mode)
    if analyser is None:
        return "Invalid mode selected.", {}
    return analyser(landmarks)

# =============================================================================
# UI Theme and Drawing Utilities
# =============================================================================

def rounded_rectangle(image, x1, y1, x2, y2, colour, radius=18, thickness=-1):
    """Draw a filled or outlined rounded rectangle without internal border lines."""
    width = x2 - x1
    height = y2 - y1
    radius = max(1, min(radius, width // 2, height // 2))

    if thickness < 0:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), colour, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), colour, -1)
        for centre in (
            (x1 + radius, y1 + radius),
            (x2 - radius, y1 + radius),
            (x2 - radius, y2 - radius),
            (x1 + radius, y2 - radius),
        ):
            cv2.circle(image, centre, radius, colour, -1, cv2.LINE_AA)
        return

    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), colour, thickness, cv2.LINE_AA)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), colour, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), colour, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), colour, thickness, cv2.LINE_AA)
    cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 0, 180, 270, colour, thickness, cv2.LINE_AA)
    cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 0, 270, 360, colour, thickness, cv2.LINE_AA)
    cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, colour, thickness, cv2.LINE_AA)
    cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 0, 90, 180, colour, thickness, cv2.LINE_AA)


def draw_card(image, rect, fill=UI_SURFACE, border=None, radius=18):
    """Draw a restrained dashboard surface; borders are optional to reduce visual noise."""
    x1, y1, x2, y2 = rect
    rounded_rectangle(image, x1, y1, x2, y2, fill, radius=radius, thickness=-1)
    if border is not None:
        rounded_rectangle(image, x1, y1, x2, y2, border, radius=radius, thickness=1)


def wrap_text(text, max_width, font_scale=0.52, thickness=1):
    """Wrap text using measured pixel width rather than character count."""
    if not text:
        return []

    lines = []
    current = ""
    for word in str(text).split():
        candidate = word if not current else f"{current} {word}"
        width = cv2.getTextSize(candidate, FONT, font_scale, thickness)[0][0]
        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped_text(
    image,
    text,
    x,
    y,
    max_width,
    max_lines=3,
    font_scale=0.52,
    colour=UI_TEXT,
    thickness=1,
    line_gap=24,
):
    lines = wrap_text(text, max_width, font_scale, thickness)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and cv2.getTextSize(last + "...", FONT, font_scale, thickness)[0][0] > max_width:
            last = last[:-1]
        lines[-1] = last.rstrip() + "..."

    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x, y + index * line_gap),
            FONT,
            font_scale,
            colour,
            thickness,
            cv2.LINE_AA,
        )
    return y + max(0, len(lines) - 1) * line_gap


def status_colour(status):
    normalised = str(status).strip().lower().replace("_", " ")
    if normalised in {"valid", "confirmed", "complete", "ok", "saved"}:
        return UI_SUCCESS
    if normalised == "ready":
        return UI_ACCENT
    if normalised in {"recording", "transcribing", "generating", "starting"}:
        return UI_CYAN
    if normalised in {"review", "low confidence", "paused"}:
        return UI_WARNING
    if normalised in {"invalid", "error", "stopped", "stop", "unavailable", "no speech"}:
        return UI_DANGER
    return UI_MUTED


def tracking_label(status):
    labels = {
        "valid": "VALID",
        "low_confidence": "LOW CONFIDENCE",
        "invalid": "NO POSE",
        "paused": "PAUSED",
    }
    return labels.get(status, str(status).replace("_", " ").upper())


def draw_status_pill(image, text, status, x, y, align_right=False):
    colour = status_colour(status)
    font_scale = 0.42
    text_width = cv2.getTextSize(text, FONT, font_scale, 1)[0][0]
    width = text_width + 42
    height = 28
    if align_right:
        x -= width

    rounded_rectangle(image, x, y, x + width, y + height, UI_SURFACE_ALT, radius=14, thickness=-1)
    cv2.circle(image, (x + 14, y + 14), 5, colour, -1, cv2.LINE_AA)
    cv2.putText(image, text, (x + 27, y + 19), FONT, font_scale, UI_TEXT, 1, cv2.LINE_AA)
    return width


def fit_image_to_rect(image, width, height, background=UI_BG):
    """Resize a camera frame with letterboxing while preserving aspect ratio."""
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    new_w = max(1, int(source_w * scale))
    new_h = max(1, int(source_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def friendly_metric_name(key):
    names = {
        "shoulder_diff": "Shoulder difference",
        "head_offset": "Head offset",
        "nose_to_shoulder_y": "Head-to-shoulder distance",
        "wrist_diff": "Wrist difference",
        "elbow_diff": "Elbow difference",
        "arms_raised": "Arms raised",
    }
    return names.get(key, key.replace("_", " ").title())


def required_landmark_summary(mode_name):
    required = {
        "Shoulder Alignment Check": "Shoulders",
        "Head Position Check": "Head | Shoulders",
        "Bilateral Arm Raise Check": "Shoulders | Elbows | Wrists",
    }
    return required.get(mode_name, "Required body landmarks")


def draw_metric_rows(image, metrics, x, y, width, max_rows=4):
    if not metrics or metrics.get("tracking") == "low confidence":
        return

    row_y = y
    for index, (key, value) in enumerate(metrics.items()):
        if index >= max_rows:
            break
        label = friendly_metric_name(key).upper()
        value_text = "YES" if value is True else "NO" if value is False else str(value)
        cv2.putText(image, label, (x, row_y), FONT, 0.35, UI_MUTED, 1, cv2.LINE_AA)
        value_width = cv2.getTextSize(value_text, FONT, 0.52, 1)[0][0]
        cv2.putText(
            image,
            value_text,
            (x + width - value_width, row_y),
            FONT,
            0.52,
            UI_TEXT,
            1,
            cv2.LINE_AA,
        )
        row_y += 27


def split_measured_feedback(tracking_status, feedback, mode_name):
    if tracking_status == "low_confidence":
        return (
            "Tracking limited",
            f"Required landmarks are not reliable enough for measurement. Required: {required_landmark_summary(mode_name)}",
        )
    if tracking_status == "invalid":
        return "No pose detected", "Move back so the required body landmarks are visible."
    if tracking_status == "paused":
        return "Coaching paused", "Press R to resume when you are ready."

    suffix = " in this observation."
    if feedback.lower().endswith(suffix):
        primary = feedback[:-len(suffix)].rstrip(". ")
        return primary, "In this observation"
    return feedback, ""


def draw_key_control(image, x, y, key, label, active=False, accent=None):
    """Draw a compact keycap plus plain-text label; only the key itself is boxed."""
    accent = accent or UI_ACCENT
    key_fill = UI_SURFACE_ALT if not active else (40, 60, 49)
    key_border = UI_BORDER if not active else accent
    label_colour = UI_TEXT if active else UI_MUTED
    key_colour = accent if active else UI_TEXT
    key_width = 30 if len(key) == 1 else 42

    rounded_rectangle(image, x, y, x + key_width, y + 28, key_fill, radius=7, thickness=-1)
    rounded_rectangle(image, x, y, x + key_width, y + 28, key_border, radius=7, thickness=1)
    key_text_width = cv2.getTextSize(key, FONT, 0.41, 1)[0][0]
    cv2.putText(
        image,
        key,
        (x + (key_width - key_text_width) // 2, y + 19),
        FONT,
        0.41,
        key_colour,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(image, label, (x + key_width + 7, y + 19), FONT, 0.41, label_colour, 1, cv2.LINE_AA)
    label_width = cv2.getTextSize(label, FONT, 0.41, 1)[0][0]
    return key_width + 7 + label_width + 18


def draw_system_item(image, x, y, label, value, width):
    cv2.putText(image, label.upper(), (x, y), FONT, 0.30, UI_DIM, 1, cv2.LINE_AA)
    draw_wrapped_text(image, value, x, y + 24, width, max_lines=1, font_scale=0.45, colour=UI_TEXT, line_gap=20)


def set_toast(toast, message, kind="info", title="", duration=2.2):
    """Set a non-blocking notification rendered from the main UI loop."""
    toast["message"] = message
    toast["kind"] = kind
    toast["title"] = title
    toast["expires_at"] = time.monotonic() + duration


def draw_toast(image, toast):
    if not toast.get("message") or time.monotonic() >= toast.get("expires_at", 0.0):
        return

    colours = {
        "success": UI_SUCCESS,
        "warning": UI_WARNING,
        "error": UI_DANGER,
        "info": UI_CYAN,
    }
    accent = colours.get(toast.get("kind"), UI_CYAN)
    width = 470
    height = 58
    x1 = (UI_WIDTH - width) // 2
    y1 = 72
    x2 = x1 + width
    y2 = y1 + height

    rounded_rectangle(image, x1, y1, x2, y2, UI_SURFACE_ALT, radius=14, thickness=-1)
    cv2.rectangle(image, (x1, y1 + 9), (x1 + 4, y2 - 9), accent, -1)
    if toast.get("title"):
        cv2.putText(image, toast["title"].upper(), (x1 + 18, y1 + 20), FONT, 0.31, accent, 1, cv2.LINE_AA)
        cv2.putText(image, toast["message"], (x1 + 18, y1 + 43), FONT, 0.47, UI_TEXT, 1, cv2.LINE_AA)
    else:
        cv2.putText(image, toast["message"], (x1 + 18, y1 + 35), FONT, 0.48, UI_TEXT, 1, cv2.LINE_AA)


def determine_workflow_step(
    speech_status,
    llm_status,
    confirmed_record,
    completed_ai_event,
    logged_request_ids,
):
    if confirmed_record is not None:
        confirmed_id = confirmed_record.get("request_id")
        if confirmed_id in logged_request_ids:
            return "voice"

    if completed_ai_event is not None:
        request_id = completed_ai_event.get("request_id")
        if request_id and request_id not in logged_request_ids:
            return "log"

    if str(llm_status).upper() in {"STARTING", "GENERATING"}:
        return "ai"
    if confirmed_record is not None:
        return "ai"
    if str(speech_status).upper() == "REVIEW":
        return "confirm"
    return "voice"


# =============================================================================
# Dashboard Components
# =============================================================================

def build_dashboard(
    camera_frame,
    current_mode,
    tracking_status,
    measured_feedback,
    metrics,
    speech_status,
    transcript,
    llm_status,
    ai_feedback,
    coaching_paused,
    session_id,
    current_request_id,
    workflow_step,
    toast,
):
    """Build the examiner-facing split-screen dashboard around the live camera feed."""
    canvas = np.full((UI_HEIGHT, UI_WIDTH, 3), UI_BG, dtype=np.uint8)
    mode_name = MODE_NAMES[current_mode]

    margin = 22
    header_h = 58
    footer_h = 132
    content_top = margin + header_h
    content_bottom = UI_HEIGHT - footer_h - 14
    content_h = content_bottom - content_top

    camera_x1 = margin
    camera_y1 = content_top
    camera_w = 846
    camera_h = content_h
    camera_x2 = camera_x1 + camera_w
    camera_y2 = camera_y1 + camera_h

    sidebar_x1 = camera_x2 + 16
    sidebar_x2 = UI_WIDTH - margin
    sidebar_w = sidebar_x2 - sidebar_x1
    card_gap = 10

    # Header: product identity, active mode and tracking state.
    cv2.circle(canvas, (margin + 8, 32), 6, UI_ACCENT, -1, cv2.LINE_AA)
    cv2.putText(canvas, "POSTUREFLOW", (margin + 24, 41), FONT, 0.80, UI_TEXT, 2, cv2.LINE_AA)
    cv2.putText(canvas, "MULTIMODAL POSTURE AWARENESS", (margin + 220, 39), FONT, 0.34, UI_MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, "H  HELP", (674, 39), FONT, 0.34, UI_MUTED, 1, cv2.LINE_AA)

    mode_text = mode_name.upper()
    mode_width = cv2.getTextSize(mode_text, FONT, 0.40, 1)[0][0]
    cv2.putText(canvas, mode_text, (UI_WIDTH - margin - mode_width - 182, 39), FONT, 0.40, UI_TEXT, 1, cv2.LINE_AA)
    draw_status_pill(canvas, tracking_label(tracking_status), tracking_status, UI_WIDTH - margin, 18, align_right=True)

    # Camera remains the primary visual surface.
    draw_card(canvas, (camera_x1, camera_y1, camera_x2, camera_y2), fill=(7, 8, 9), border=UI_BORDER)
    inner = 7
    camera_view = fit_image_to_rect(camera_frame, camera_w - 2 * inner, camera_h - 2 * inner, background=(5, 6, 7))
    canvas[camera_y1 + inner:camera_y2 - inner, camera_x1 + inner:camera_x2 - inner] = camera_view

    rounded_rectangle(canvas, camera_x1 + 18, camera_y1 + 18, camera_x1 + 126, camera_y1 + 47, UI_SURFACE, radius=14, thickness=-1)
    cv2.circle(canvas, (camera_x1 + 34, camera_y1 + 32), 5, UI_ACCENT, -1, cv2.LINE_AA)
    cv2.putText(canvas, "LIVE CAMERA", (camera_x1 + 47, camera_y1 + 37), FONT, 0.37, UI_TEXT, 1, cv2.LINE_AA)

    if coaching_paused:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (camera_x1 + 8, camera_y1 + 8), (camera_x2 - 8, camera_y2 - 8), (20, 20, 90), -1)
        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)
        pause_text = "COACHING PAUSED"
        size = cv2.getTextSize(pause_text, FONT, 0.86, 2)[0]
        cv2.putText(
            canvas,
            pause_text,
            (camera_x1 + (camera_w - size[0]) // 2, camera_y1 + camera_h // 2),
            FONT,
            0.86,
            UI_DANGER,
            2,
            cv2.LINE_AA,
        )

    # Sidebar layout: system state, measured output, speech and generated explanation.
    system_h = 124
    measured_h = 188
    speech_h = 130
    system_rect = (sidebar_x1, content_top, sidebar_x2, content_top + system_h)
    measured_rect = (sidebar_x1, system_rect[3] + card_gap, sidebar_x2, system_rect[3] + card_gap + measured_h)
    speech_rect = (sidebar_x1, measured_rect[3] + card_gap, sidebar_x2, measured_rect[3] + card_gap + speech_h)
    ai_rect = (sidebar_x1, speech_rect[3] + card_gap, sidebar_x2, content_bottom)

    draw_card(canvas, system_rect, fill=UI_SURFACE_SOFT, border=UI_BORDER)
    x1, y1, x2, y2 = system_rect
    cv2.putText(canvas, "SYSTEM / PIPELINE", (x1 + 18, y1 + 25), FONT, 0.36, UI_MUTED, 1, cv2.LINE_AA)
    cell_w = (sidebar_w - 54) // 2
    draw_system_item(canvas, x1 + 18, y1 + 46, "Pose", POSE_MODEL_LABEL, cell_w)
    draw_system_item(canvas, x1 + 30 + cell_w, y1 + 46, "Speech", SPEECH_MODEL_LABEL, cell_w)
    draw_system_item(canvas, x1 + 18, y1 + 88, "AI", AI_MODEL_LABEL, cell_w)
    draw_system_item(canvas, x1 + 30 + cell_w, y1 + 88, "Request", current_request_id or "-", cell_w)

    draw_card(canvas, measured_rect, border=UI_BORDER)
    mx1, my1, mx2, my2 = measured_rect
    cv2.putText(canvas, "MEASURED FEEDBACK", (mx1 + 18, my1 + 27), FONT, 0.36, UI_MUTED, 1, cv2.LINE_AA)
    draw_status_pill(canvas, tracking_label(tracking_status), tracking_status, mx2 - 18, my1 + 10, align_right=True)
    primary, secondary = split_measured_feedback(tracking_status, measured_feedback, mode_name)
    primary_y = draw_wrapped_text(canvas, primary, mx1 + 18, my1 + 67, sidebar_w - 36, max_lines=2, font_scale=0.61, colour=UI_TEXT, line_gap=28)
    if secondary:
        secondary_y = primary_y + 26
        draw_wrapped_text(canvas, secondary, mx1 + 18, secondary_y, sidebar_w - 36, max_lines=2, font_scale=0.42, colour=UI_MUTED, line_gap=21)
    metrics_y = my2 - 47 if tracking_status == "valid" else my2 - 27
    if tracking_status == "valid":
        draw_metric_rows(canvas, metrics, mx1 + 18, metrics_y, sidebar_w - 36, max_rows=2)
    elif tracking_status == "low_confidence":
        cv2.putText(canvas, f"REQUIRED  {required_landmark_summary(mode_name).upper()}", (mx1 + 18, my2 - 18), FONT, 0.31, UI_WARNING, 1, cv2.LINE_AA)

    draw_card(canvas, speech_rect, border=UI_BORDER)
    sx1, sy1, sx2, sy2 = speech_rect
    cv2.putText(canvas, "SPEECH", (sx1 + 18, sy1 + 27), FONT, 0.36, UI_MUTED, 1, cv2.LINE_AA)
    speech_label = str(speech_status).replace("_", " ").upper()
    draw_status_pill(canvas, speech_label, speech_status, sx2 - 18, sy1 + 10, align_right=True)
    if transcript:
        speech_text = f'"{transcript}"'
        speech_colour = UI_TEXT
    elif str(speech_status).upper() == "READY":
        speech_text = "Press V to speak. Voice commands can also change modes."
        speech_colour = UI_MUTED
    elif str(speech_status).upper() == "REVIEW":
        speech_text = "Review the transcript, then press C to confirm."
        speech_colour = UI_WARNING
    else:
        speech_text = "Waiting for speech input..."
        speech_colour = UI_MUTED
    draw_wrapped_text(canvas, speech_text, sx1 + 18, sy1 + 68, sidebar_w - 36, max_lines=2, font_scale=0.50, colour=speech_colour, line_gap=24)

    draw_card(canvas, ai_rect, border=UI_BORDER)
    ax1, ay1, ax2, ay2 = ai_rect
    cv2.putText(canvas, "AI EXPLANATION", (ax1 + 18, ay1 + 27), FONT, 0.36, UI_MUTED, 1, cv2.LINE_AA)
    draw_status_pill(canvas, str(llm_status).replace("_", " ").upper(), llm_status, ax2 - 18, ay1 + 10, align_right=True)
    if ai_feedback:
        ai_text = ai_feedback
        ai_colour = UI_TEXT
    elif str(llm_status).upper() in {"GENERATING", "STARTING"}:
        ai_text = "Generating a short explanation from the confirmed pose + speech record..."
        ai_colour = UI_MUTED
    else:
        ai_text = "Confirm a transcript, then press A for an explanation."
        ai_colour = UI_MUTED
    draw_wrapped_text(canvas, ai_text, ax1 + 18, ay1 + 68, sidebar_w - 36, max_lines=3, font_scale=0.52, colour=ai_colour, line_gap=24)

    # Bottom workflow deck makes the interaction sequence explicit to first-time users.
    footer_y1 = UI_HEIGHT - footer_h
    draw_card(canvas, (margin, footer_y1, UI_WIDTH - margin, UI_HEIGHT - margin), fill=UI_SURFACE, border=UI_BORDER)

    cv2.putText(canvas, "MODES", (margin + 18, footer_y1 + 24), FONT, 0.32, UI_MUTED, 1, cv2.LINE_AA)
    x = margin + 18
    y = footer_y1 + 34
    for mode, label in [(1, "Shoulder"), (2, "Head"), (3, "Arm Raise")]:
        x += draw_key_control(canvas, x, y, str(mode), label, active=current_mode == mode)

    workflow_x = 445
    cv2.putText(canvas, "WORKFLOW", (workflow_x, footer_y1 + 24), FONT, 0.32, UI_MUTED, 1, cv2.LINE_AA)
    x = workflow_x
    workflow_controls = [("V", "Speak", "voice"), ("C", "Confirm", "confirm"), ("A", "AI", "ai"), ("L", "Log", "log")]
    for index, (key, label, stage) in enumerate(workflow_controls):
        x += draw_key_control(canvas, x, y, key, label, active=workflow_step == stage, accent=UI_CYAN)
        if index < len(workflow_controls) - 1:
            cv2.putText(canvas, ">", (x - 6, y + 19), FONT, 0.45, UI_DIM, 1, cv2.LINE_AA)
            x += 18

    second_y = footer_y1 + 77
    cv2.putText(canvas, "SAFETY / UTILITIES", (margin + 18, second_y - 9), FONT, 0.31, UI_MUTED, 1, cv2.LINE_AA)
    x = margin + 18
    for key, label, colour in [
        ("X", "Stop", UI_DANGER),
        ("R", "Resume", UI_ACCENT),
        ("S", "Screenshot", UI_MUTED),
        ("H", "Help", UI_CYAN),
        ("Q", "Quit", UI_MUTED),
    ]:
        x += draw_key_control(canvas, x, second_y, key, label, active=(key == "X" and coaching_paused), accent=colour)

    voice_hint = 'VOICE: "switch to head mode"  |  "switch to arm mode"  |  "stop"  |  "resume"'
    cv2.putText(canvas, voice_hint, (744, second_y + 18), FONT, 0.31, UI_MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"SESSION {session_id}", (UI_WIDTH - margin - 132, footer_y1 + 22), FONT, 0.28, UI_DIM, 1, cv2.LINE_AA)
    disclaimer = "General posture guidance only. Stop if you feel pain or discomfort."
    disclaimer_width = cv2.getTextSize(disclaimer, FONT, 0.36, 1)[0][0]
    cv2.putText(canvas, disclaimer, (UI_WIDTH - margin - disclaimer_width, UI_HEIGHT - 29), FONT, 0.36, UI_WARNING, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Press H for Help", (margin + 18, UI_HEIGHT - 29), FONT, 0.34, UI_MUTED, 1, cv2.LINE_AA)

    draw_toast(canvas, toast)
    return canvas


# =============================================================================
# Help Overlay
# =============================================================================

def draw_help_key(image, x, y, key, label):
    width = draw_key_control(image, x, y, key, label, active=True, accent=UI_CYAN)
    return x + width


def draw_help_overlay(image):
    """Draw an in-app quick-start guide without opening a second window."""
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (UI_WIDTH, UI_HEIGHT), (3, 4, 5), -1)
    cv2.addWeighted(overlay, 0.80, image, 0.20, 0, image)

    x1, y1, x2, y2 = 62, 42, UI_WIDTH - 62, UI_HEIGHT - 42
    draw_card(image, (x1, y1, x2, y2), fill=(18, 21, 22), border=UI_BORDER, radius=20)

    cv2.circle(image, (x1 + 26, y1 + 28), 6, UI_ACCENT, -1, cv2.LINE_AA)
    cv2.putText(image, "POSTUREFLOW", (x1 + 43, y1 + 37), FONT, 0.77, UI_TEXT, 2, cv2.LINE_AA)
    cv2.putText(image, "QUICK START", (x1 + 234, y1 + 35), FONT, 0.38, UI_MUTED, 1, cv2.LINE_AA)
    cv2.putText(image, "H / ESC  CLOSE HELP", (x2 - 195, y1 + 34), FONT, 0.34, UI_CYAN, 1, cv2.LINE_AA)

    left_x = x1 + 32
    left_w = 610
    right_x = x1 + 690
    right_w = x2 - right_x - 32

    def section_heading(text, x, y):
        cv2.putText(image, text, (x, y), FONT, 0.38, UI_ACCENT, 1, cv2.LINE_AA)

    def step(number, title, body, x, y, width):
        cv2.putText(image, str(number), (x, y), FONT, 0.52, UI_ACCENT, 2, cv2.LINE_AA)
        cv2.putText(image, title.upper(), (x + 30, y), FONT, 0.42, UI_TEXT, 1, cv2.LINE_AA)
        bottom = draw_wrapped_text(image, body, x + 30, y + 27, width - 30, max_lines=3, font_scale=0.43, colour=UI_MUTED, line_gap=21)
        return bottom + 32

    section_heading("GET STARTED", left_x, y1 + 82)
    y = y1 + 118
    y = step(1, "Select a check", "Use 1 Shoulder, 2 Head, or 3 Arm Raise. You can also select any mode by voice.", left_x, y, left_w)
    y = step(2, "Position yourself", "Keep the required body landmarks clearly visible inside the live camera frame.", left_x, y, left_w)
    y = step(3, "Speak", "Press V and describe how the movement feels, for example: My shoulders feel fine.", left_x, y, left_w)
    y = step(4, "Review", "Check the recognised transcript. Press C to confirm it before AI generation.", left_x, y, left_w)
    y = step(5, "Generate", "Press A for a short explanation based on the confirmed pose + speech observation.", left_x, y, left_w)
    step(6, "Save", "Press L to log the completed multimodal observation for evaluation.", left_x, y, left_w)

    section_heading("VOICE COMMANDS", right_x, y1 + 82)
    draw_wrapped_text(
        image,
        'Say: "switch to shoulder mode"  |  "switch to head mode"  |  "switch to arm raise mode"',
        right_x,
        y1 + 119,
        right_w,
        max_lines=3,
        font_scale=0.43,
        colour=UI_TEXT,
        line_gap=22,
    )
    draw_wrapped_text(image, 'You can also say "stop" or "resume". Navigation commands are handled deterministically and are not sent to Qwen.', right_x, y1 + 190, right_w, max_lines=3, font_scale=0.42, colour=UI_MUTED, line_gap=21)

    section_heading("WORKFLOW", right_x, y1 + 278)
    x = right_x
    for index, (key, label) in enumerate([("V", "Speak"), ("C", "Confirm"), ("A", "AI"), ("L", "Log")]):
        x = draw_help_key(image, x, y1 + 296, key, label)
        if index < 3:
            cv2.putText(image, ">", (x - 7, y1 + 315), FONT, 0.44, UI_DIM, 1, cv2.LINE_AA)
            x += 18

    section_heading("SAFETY", right_x, y1 + 382)
    draw_wrapped_text(image, "X immediately pauses coaching. R resumes coaching. Speech recognition is not the only stop mechanism.", right_x, y1 + 418, right_w, max_lines=3, font_scale=0.43, colour=UI_TEXT, line_gap=22)
    draw_wrapped_text(image, "PostureFlow provides general posture-awareness guidance only. It is not a medical device and does not diagnose or treat conditions.", right_x, y1 + 493, right_w, max_lines=4, font_scale=0.42, colour=UI_MUTED, line_gap=21)
    cv2.putText(image, "Stop the movement if you feel pain or discomfort.", (right_x, y1 + 585), FONT, 0.43, UI_WARNING, 1, cv2.LINE_AA)

    section_heading("OTHER CONTROLS", right_x, y1 + 642)
    x = right_x
    x = draw_help_key(image, x, y1 + 660, "S", "Screenshot")
    x = draw_help_key(image, x, y1 + 660, "H", "Close Help")
    draw_help_key(image, x, y1 + 660, "Q", "Quit")


# =============================================================================
# Main Application Loop
# =============================================================================

def print_controls():
    print("PostureFlow prototype running.")
    print("Controls:")
    print("1 = Shoulder Alignment Check")
    print("2 = Head Position Check")
    print("3 = Bilateral Arm Raise Check")
    print("V = Record speech input")
    print("C = Confirm displayed transcript")
    print("A = Generate AI feedback")
    print("L = Log completed multimodal result")
    print("X = Stop/pause coaching")
    print("R = Resume coaching")
    print("S = Save manual screenshot")
    print("H = Help / instructions")
    print("Q = Quit")


def main():
    log_file = initialise_log()
    logged_request_ids = set()

    session_id = str(uuid.uuid4())[:8]
    print(f"Session ID: {session_id}")
    print(f"Evaluation log: {log_file}")

    current_mode = 1
    latest_feedback = "No feedback yet."
    latest_metrics = {}
    tracking_status = "invalid"
    coaching_paused = False

    help_visible = False
    toast = {"message": "", "kind": "info", "title": "", "expires_at": 0.0}

    llm_result_queue = queue.Queue()
    llm_thread = None
    llm_status = "READY"
    latest_ai_feedback = ""
    latest_completed_ai_event = None
    llm_service = LLMService(model="qwen3:4b-instruct")

    speech_result_queue = queue.Queue()
    speech_thread = None
    speech_status = "INITIALISING"
    latest_transcript = ""
    latest_speech_context = None
    pending_speech_result = None
    confirmed_multimodal_record = None
    speech_available = False

    speech_service = SpeechService(
        model_size=SPEECH_MODEL_SIZE,
        input_device=MICROPHONE_INDEX,
        sample_rate=16000,
        channels=1,
        use_vad=True,
        audio_dir="audio_tests",
    )

    try:
        microphone_info = speech_service.get_device_info()
        print(f"Speech microphone: {microphone_info['name']}")
        speech_service.validate_input()
        print("Loading speech recognition model...")
        model_load_time = speech_service.load_model()
        print(f"Speech model loaded in {model_load_time:.2f} seconds.")
        speech_available = True
        speech_status = "READY"
    except Exception as error:
        print(f"Speech input unavailable: {error}")
        speech_status = "UNAVAILABLE"
        set_toast(toast, "Speech input unavailable", "error")

    cap = open_working_camera()
    if cap is None:
        print(
            "Could not open the webcam. Check Windows camera permissions, "
            "close other camera apps, or reconnect/reset the camera hardware."
        )
        return

    print_controls()
    set_toast(toast, "Press H for instructions", "info", duration=3.0)

    window_name = "PostureFlow"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, UI_WIDTH, UI_HEIGHT)
    cv2.moveWindow(window_name, 80, 60)

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            failed_frame_count = 0

            while True:
                success, frame = cap.read()
                if not success or frame is None:
                    failed_frame_count += 1
                    if failed_frame_count % 10 == 1:
                        print(
                            "Warning: Could not read frame from webcam. "
                            f"Retrying... ({failed_frame_count})"
                        )
                    if failed_frame_count >= 30:
                        print("Error: Webcam stopped returning frames.")
                        break
                    cv2.waitKey(100)
                    continue

                failed_frame_count = 0
                if MIRROR_DISPLAY:
                    frame = cv2.flip(frame, 1)

                # Process speech worker results.
                while True:
                    try:
                        speech_event = speech_result_queue.get_nowait()
                    except queue.Empty:
                        break

                    event_type = speech_event["type"]
                    if event_type == "status":
                        speech_status = speech_event["status"]
                        continue

                    if event_type == "error":
                        speech_status = "ERROR"
                        speech_thread = None
                        print(f"\nSpeech request failed: {speech_event['error']}")
                        set_toast(toast, "Speech recognition failed", "error")
                        continue

                    latest_transcript = speech_event["transcript"].strip()
                    latest_speech_context = speech_event["context"]
                    speech_thread = None

                    voice_command = detect_voice_command(latest_transcript)
                    if voice_command is not None:
                        command_type, command_value = voice_command

                        if command_type == "mode":
                            current_mode = command_value
                            command_message = f"Switched to {MODE_NAMES[current_mode]}"
                            print(f"Voice command: {command_message.lower()}")
                            set_toast(toast, command_message, "success", title="VOICE COMMAND")
                        elif command_type == "stop":
                            coaching_paused = True
                            command_message = "Coaching paused"
                            print("Voice command: coaching paused.")
                            set_toast(toast, command_message, "warning", title="VOICE COMMAND")
                        else:
                            coaching_paused = False
                            command_message = "Coaching resumed"
                            print("Voice command: coaching resumed.")
                            set_toast(toast, command_message, "success", title="VOICE COMMAND")

                        latest_transcript = ""
                        latest_speech_context = None
                        pending_speech_result = None
                        confirmed_multimodal_record = None
                        latest_ai_feedback = ""
                        latest_completed_ai_event = None
                        speech_status = "STOPPED" if coaching_paused else "READY"
                        llm_status = "READY"
                        continue

                    if not latest_transcript:
                        speech_status = "NO SPEECH"
                        pending_speech_result = None
                        set_toast(toast, "No speech detected", "warning")
                    else:
                        safety_status = assess_transcript_safety(latest_transcript)
                        pending_speech_result = {
                            "request_id": speech_event["request_id"],
                            "session_id": latest_speech_context["session_id"],
                            "started_at": latest_speech_context["started_at"],
                            "screenshot_filename": latest_speech_context["screenshot_filename"],
                            "pose": {
                                "mode": latest_speech_context["mode_name"],
                                "tracking_status": latest_speech_context["tracking_status"],
                                "metrics": copy.deepcopy(latest_speech_context["metrics"]),
                                "rule_feedback": latest_speech_context["rule_feedback"],
                            },
                            "speech": {
                                "raw_transcript": latest_transcript,
                                "confirmed_transcript": "",
                                "audio_path": speech_event["audio_path"],
                                "transcription_seconds": speech_event["transcription_seconds"],
                            },
                            "safety_status": safety_status,
                        }
                        speech_status = "REVIEW"

                        if safety_status == "stop":
                            coaching_paused = True
                            speech_status = "STOPPED"
                            print(
                                "Safety action: coaching paused because the transcript "
                                "contains a stop/pain statement."
                            )
                            set_toast(toast, "Coaching paused: stop/pain statement detected", "warning", title="SAFETY")

                    print("\n--- Speech Result ---")
                    print(f"Request ID: {speech_event['request_id']}")
                    print(f"Captured mode: {latest_speech_context['mode_name']}")
                    print(f"Captured tracking: {latest_speech_context['tracking_status']}")
                    print(f"Captured metrics: {latest_speech_context['metrics']}")
                    print(f"Transcript: {latest_transcript or '[NO SPEECH DETECTED]'}")
                    print(f"Transcription time: {speech_event['transcription_seconds']:.2f}s")
                    print(f"Audio: {speech_event['audio_path']}")
                    print("---------------------\n")

                # Process LLM worker results.
                while True:
                    try:
                        llm_event = llm_result_queue.get_nowait()
                    except queue.Empty:
                        break

                    event_type = llm_event["type"]
                    if event_type == "status":
                        llm_status = llm_event["status"]
                    elif event_type == "result":
                        llm_status = "COMPLETE"
                        llm_thread = None
                        latest_ai_feedback = llm_event["feedback"]
                        latest_completed_ai_event = copy.deepcopy(llm_event)
                        set_toast(toast, "AI explanation ready", "success", title="AI COMPLETE")

                        print("\n--- AI Feedback Result ---")
                        print(f"Request ID: {llm_event['request_id']}")
                        print(f"Model: {llm_event['model']}")
                        print(f"Status: {llm_event['llm_status']}")
                        print(f"Used pose metrics: {llm_event['used_pose_metrics']}")
                        print(f"Feedback: {latest_ai_feedback}")
                        print(f"Generation time: {llm_event['generation_seconds']:.2f}s")
                        print("AI result stored for logging.")
                        print("--------------------------\n")
                    elif event_type == "error":
                        llm_status = "ERROR"
                        llm_thread = None
                        latest_ai_feedback = "AI feedback unavailable. See terminal."
                        print(f"\nAI feedback failed: {llm_event['error']}")
                        set_toast(toast, "AI feedback unavailable", "error")

                # Update the live deterministic pose measurement.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = pose.process(rgb_frame)
                rgb_frame.flags.writeable = True

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

                    if coaching_paused:
                        latest_feedback = "Coaching paused. Press R to resume."
                        latest_metrics = {}
                        tracking_status = "paused"
                    else:
                        latest_feedback, latest_metrics = analyse_current_mode(
                            current_mode,
                            results.pose_landmarks.landmark,
                        )
                        tracking_status = (
                            "low_confidence"
                            if latest_metrics.get("tracking") == "low confidence"
                            else "valid"
                        )
                else:
                    latest_feedback = "No pose detected. Move back so the required body landmarks are visible."
                    latest_metrics = {}
                    tracking_status = "invalid"

                if confirmed_multimodal_record is not None:
                    current_request_id = confirmed_multimodal_record["request_id"]
                elif pending_speech_result is not None:
                    current_request_id = pending_speech_result["request_id"]
                elif latest_speech_context is not None:
                    current_request_id = latest_speech_context.get("request_id", "")
                else:
                    current_request_id = ""

                workflow_step = determine_workflow_step(
                    speech_status,
                    llm_status,
                    confirmed_multimodal_record,
                    latest_completed_ai_event,
                    logged_request_ids,
                )

                dashboard = build_dashboard(
                    camera_frame=frame,
                    current_mode=current_mode,
                    tracking_status=tracking_status,
                    measured_feedback=latest_feedback,
                    metrics=latest_metrics,
                    speech_status=speech_status,
                    transcript=latest_transcript,
                    llm_status=llm_status,
                    ai_feedback=latest_ai_feedback,
                    coaching_paused=coaching_paused,
                    session_id=session_id,
                    current_request_id=current_request_id,
                    workflow_step=workflow_step,
                    toast=toast,
                )

                if help_visible:
                    draw_help_overlay(dashboard)

                cv2.imshow(window_name, dashboard)
                key = cv2.waitKey(1) & 0xFF

                # Help captures keyboard focus so workflow actions cannot fire behind it.
                if help_visible:
                    if key in (ord("h"), 27):
                        help_visible = False
                    elif key == ord("q"):
                        break
                    continue

                if key == ord("h"):
                    help_visible = True
                    continue

                # Handle normal keyboard input.
                if key in (ord("1"), ord("2"), ord("3")):
                    speech_busy = speech_thread is not None and speech_thread.is_alive()
                    llm_busy = llm_thread is not None and llm_thread.is_alive()
                    if speech_busy or llm_busy:
                        message = "Finish the current speech/AI request before changing mode"
                        print(message + ".")
                        set_toast(toast, message, "warning")
                    else:
                        current_mode = int(chr(key))
                        message = f"Switched to {MODE_NAMES[current_mode]}"
                        print(message + ".")
                        set_toast(toast, message, "info")

                elif key == ord("v"):
                    if not speech_available:
                        print("Speech input is currently unavailable.")
                        set_toast(toast, "Speech input is unavailable", "error")
                    elif speech_thread is not None and speech_thread.is_alive():
                        print("A speech request is already running.")
                        set_toast(toast, "A speech request is already running", "warning")
                    elif llm_thread is not None and llm_thread.is_alive():
                        print("Wait for the current AI feedback before starting speech.")
                        set_toast(toast, "Wait for the current AI feedback first", "warning")
                    else:
                        request_id = str(uuid.uuid4())[:8]
                        request_screenshot = save_screenshot(
                            frame.copy(),
                            identifier=request_id,
                            mode_name=MODE_NAMES[current_mode],
                            kind="request",
                        )
                        request_context = {
                            "request_id": request_id,
                            "started_at": datetime.now().isoformat(timespec="seconds"),
                            "session_id": session_id,
                            "mode": current_mode,
                            "mode_name": MODE_NAMES[current_mode],
                            "tracking_status": tracking_status,
                            "rule_feedback": latest_feedback,
                            "metrics": copy.deepcopy(latest_metrics),
                            "screenshot_filename": request_screenshot,
                        }

                        latest_transcript = ""
                        latest_speech_context = None
                        pending_speech_result = None
                        confirmed_multimodal_record = None
                        latest_ai_feedback = ""
                        latest_completed_ai_event = None
                        llm_status = "READY"
                        speech_status = "STARTING"
                        set_toast(toast, "Listening for a short voice input", "info", title="SPEECH")

                        print("\nStarting speech request...")
                        print(f"Request ID: {request_id}")
                        print(f"Pose mode captured: {request_context['mode_name']}")
                        print(f"Tracking captured: {request_context['tracking_status']}")
                        print(f"Metrics captured: {request_context['metrics']}")
                        print(f"Screenshot: {request_screenshot or '[SAVE FAILED]'}")

                        speech_thread = threading.Thread(
                            target=speech_worker,
                            args=(speech_service, speech_result_queue, request_context),
                            daemon=True,
                        )
                        speech_thread.start()

                elif key == ord("c"):
                    if pending_speech_result is None:
                        print("There is no speech result waiting for review.")
                        set_toast(toast, "No transcript is waiting for confirmation", "warning")
                    elif not latest_transcript:
                        print("Cannot confirm an empty transcript.")
                        set_toast(toast, "Cannot confirm an empty transcript", "warning")
                    else:
                        pending_speech_result["speech"]["confirmed_transcript"] = latest_transcript
                        confirmed_multimodal_record = copy.deepcopy(pending_speech_result)
                        pending_speech_result = None
                        speech_status = "STOPPED" if coaching_paused else "CONFIRMED"
                        set_toast(toast, "Transcript confirmed", "success")

                        print("\n--- Multimodal Record Confirmed ---")
                        print(f"Request ID: {confirmed_multimodal_record['request_id']}")
                        print(f"Mode: {confirmed_multimodal_record['pose']['mode']}")
                        print(f"Tracking: {confirmed_multimodal_record['pose']['tracking_status']}")
                        print(f"Metrics: {confirmed_multimodal_record['pose']['metrics']}")
                        print(f"Rule feedback: {confirmed_multimodal_record['pose']['rule_feedback']}")
                        print(f"Confirmed transcript: {confirmed_multimodal_record['speech']['confirmed_transcript']}")
                        print(f"Safety status: {confirmed_multimodal_record['safety_status']}")
                        print("-----------------------------------\n")

                elif key == ord("a"):
                    if confirmed_multimodal_record is None:
                        print("No confirmed multimodal record is available.")
                        set_toast(toast, "Confirm a transcript before generating AI feedback", "warning")
                    elif llm_thread is not None and llm_thread.is_alive():
                        print("AI feedback generation is already running.")
                        set_toast(toast, "AI feedback is already generating", "warning")
                    elif confirmed_multimodal_record["safety_status"] == "stop":
                        latest_ai_feedback = (
                            "Coaching is paused. Stop the movement if you feel pain or discomfort."
                        )
                        llm_status = "STOPPED"
                        latest_completed_ai_event = {
                            "type": "result",
                            "request_id": confirmed_multimodal_record["request_id"],
                            "record": copy.deepcopy(confirmed_multimodal_record),
                            "llm_status": "stop",
                            "feedback": latest_ai_feedback,
                            "used_pose_metrics": False,
                            "generation_seconds": 0.0,
                            "model": "deterministic_safety_layer",
                        }
                        print("AI generation skipped because coaching is paused for safety.")
                        set_toast(toast, "Generative AI skipped: coaching is paused", "warning", title="SAFETY")
                    else:
                        llm_status = "STARTING"
                        latest_ai_feedback = ""
                        latest_completed_ai_event = None
                        print("\nStarting AI feedback...")
                        print(f"Request ID: {confirmed_multimodal_record['request_id']}")
                        llm_thread = threading.Thread(
                            target=llm_worker,
                            args=(llm_service, llm_result_queue, copy.deepcopy(confirmed_multimodal_record)),
                            daemon=True,
                        )
                        llm_thread.start()

                elif key == ord("x"):
                    coaching_paused = True
                    speech_status = "STOPPED"
                    print("Manual stop activated. Normal coaching feedback is paused.")
                    set_toast(toast, "Coaching paused", "warning", title="SAFETY")

                elif key == ord("r"):
                    coaching_paused = False
                    if speech_available:
                        speech_status = "READY"
                    print("Coaching resumed.")
                    set_toast(toast, "Coaching resumed", "success")

                elif key == ord("s"):
                    manual_id = str(uuid.uuid4())[:8]
                    filename = save_screenshot(
                        dashboard.copy(),
                        identifier=manual_id,
                        mode_name=MODE_NAMES[current_mode],
                        kind="manual",
                    )
                    if filename:
                        set_toast(toast, "Screenshot saved", "success")
                    else:
                        set_toast(toast, "Screenshot could not be saved", "error")

                elif key == ord("l"):
                    if confirmed_multimodal_record is None:
                        print("No confirmed multimodal record is available to log.")
                        set_toast(toast, "No completed multimodal request is available", "warning")
                    elif latest_completed_ai_event is None:
                        print("Generate AI feedback before logging the multimodal result.")
                        set_toast(toast, "Generate AI feedback before logging", "warning")
                    elif latest_completed_ai_event["request_id"] != confirmed_multimodal_record["request_id"]:
                        print("Log prevented: AI result does not match the request.")
                        set_toast(toast, "Request mismatch: log prevented", "error")
                    elif confirmed_multimodal_record["request_id"] in logged_request_ids:
                        print(
                            "This multimodal request has already been logged: "
                            f"{confirmed_multimodal_record['request_id']}"
                        )
                        set_toast(toast, "This request has already been logged", "warning")
                    else:
                        try:
                            log_multimodal_result(
                                log_file,
                                confirmed_multimodal_record,
                                latest_completed_ai_event,
                            )
                            request_id = confirmed_multimodal_record["request_id"]
                            logged_request_ids.add(request_id)
                            set_toast(
                                toast,
                                f"Observation logged - Request {request_id}",
                                "success",
                                title="SAVED",
                                duration=2.8,
                            )
                        except Exception as error:
                            print(f"Logging failed: {error}")
                            set_toast(toast, "Logging failed", "error")

                elif key == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Prototype closed.")


if __name__ == "__main__":
    main()
