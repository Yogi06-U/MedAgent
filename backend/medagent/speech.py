"""Speech-to-text audio transcription module."""

import logging
from typing import Dict, Any

logger = logging.getLogger("medagent.speech")


def transcribe_audio(file_path: str) -> Dict[str, Any]:
    """
    Transcribe dictation audio to text using faster-whisper if available.
    Returns status and transcribed text or clear reason.
    """
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, info = model.transcribe(file_path, beam_size=1)
        text = " ".join([segment.text for segment in segments]).strip()
        return {
            "status": "ok",
            "text": text,
            "language": info.language,
            "probability": info.language_probability,
        }
    except ImportError:
        logger.info("faster-whisper not installed; returning explicit unavailable status.")
        return {
            "status": "unavailable",
            "text": "",
            "message": "faster-whisper is not installed in this environment. Voice dictation is unavailable.",
        }
    except Exception as e:
        logger.error("Error transcribing audio: %s", str(e))
        return {
            "status": "error",
            "text": "",
            "message": f"Transcription failed: {str(e)}",
        }
