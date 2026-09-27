POSTUREFLOW
===========

PostureFlow is a multimodal AI prototype for general posture awareness.

The system combines:

- MediaPipe Pose for real-time body landmark estimation
- Deterministic posture measurements and safety handling
- faster-whisper for local speech recognition
- Qwen3 4B Instruct through Ollama for constrained AI explanations

This project was developed for the CM3070 Final Project under the Artificial Intelligence project template:
"Orchestrating AI Models to Achieve a Goal".


IMPORTANT
=========

PostureFlow is an educational prototype for general posture awareness.

It is not a medical device, does not diagnose medical conditions or injuries, does not prescribe treatment, and does not replace a physiotherapist or other healthcare professional.


SYSTEM REQUIREMENTS
===================

- Windows
- Python 3.11
- Webcam
- Microphone
- Ollama


INSTALLATION
============

1. Create a Python virtual environment:

python -m venv venv311

2. Activate the virtual environment:

.\venv311\Scripts\Activate.ps1

3. Install the Python dependencies:

pip install -r requirements.txt

4. Install Ollama separately.

5. Download the required Qwen model:

ollama pull qwen3:4b-instruct

complete copy-paste sequence:

py -3.11 -m venv venv311
.\venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull qwen3:4b-instruct
python .\postureflow_v3.py


RUNNING POSTUREFLOW
===================

Ensure Ollama is running, then execute:

python .\postureflow_v3.py

The faster-whisper base.en model is loaded locally by the speech service.


CONTROLS
========

1   Shoulder Alignment Check
2   Head Position Check
3   Bilateral Arm Raise Check

V   Record speech input
C   Confirm displayed transcript
A   Generate AI explanation
L   Log completed multimodal request

X   Pause coaching
R   Resume coaching
S   Save manual screenshot
H   Open help
Q   Quit


MULTIMODAL WORKFLOW
===================

The normal interaction sequence is:

V -> C -> A -> L

1. V creates a new request, freezes the current pose context and records speech.

2. faster-whisper transcribes the spoken input locally.

3. Deterministic safety checking occurs immediately after transcription.

4. For normal speech, the transcript is displayed for user review.

5. C confirms the transcript.

6. A combines the confirmed transcript with the frozen pose context and sends the structured multimodal record to Qwen3 through Ollama.

7. L logs the completed multimodal request.

Recognised stop or pain-related statements pause coaching through deterministic application logic and bypass normal Qwen generation.


POSTURE MODES
=============

1. Shoulder Alignment Check

Compares the vertical positions of the left and right shoulder landmarks.


2. Head Position Check

Uses the nose relative to the shoulder midpoint to provide a narrow head-position observation.


3. Bilateral Arm Raise Check

Uses shoulder, elbow and wrist landmarks to observe bilateral arm height.


These checks describe observable landmark relationships only and are not clinical posture assessments.


SAFETY
======

PostureFlow uses deterministic safety handling before generative AI processing.

Recognised stop or pain-related statements pause coaching without waiting for Qwen.

The X key also provides an independent manual stop control.

The system does not use the language model to determine whether coaching should stop.


PRIVACY
=======

Processing is performed locally.

Raw webcam video is not continuously recorded.

The application may create local:

- screenshots
- audio files
- transcripts
- pose metrics
- CSV logs

Speech and pose data are not sent to an external generative-AI API because Qwen runs locally through Ollama.


PROJECT STRUCTURE
=================

PostureFlow/

    postureflow_v3.py
    speech_service.py
    llm_service.py
    requirements.txt
    README.txt

    tests/
        camera_backend_diagnostic.py
        llm_service_smoke_test.py
        speech_service_test.py


DEVELOPMENT UTILITIES
=====================

The tests folder contains small component-level diagnostic utilities.

camera_backend_diagnostic.py
Checks available webcam indexes and OpenCV capture backends.

speech_service_test.py
Tests microphone input and faster-whisper transcription.

llm_service_smoke_test.py
Sends a representative structured multimodal record directly to the local Qwen service.

These utilities are separate from the controlled evaluation reported in the final project report.


MAIN TECHNOLOGIES
=================

- Python 3.11
- OpenCV
- MediaPipe Pose
- faster-whisper
- Ollama
- Qwen3 4B Instruct
- NumPy
- sounddevice
- soundfile


DISCLAIMER
==========

PostureFlow provides general posture-awareness feedback only.

It is not intended for diagnosis, treatment, rehabilitation prescription or clinical decision-making.

Stop using the system if pain or discomfort occurs and seek appropriate professional advice where necessary.