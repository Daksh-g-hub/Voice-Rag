import os
import io
import time
import httpx
from typing import Optional, Tuple
from ..config import settings

class STTService:
    """
    Speech-To-Text service using Sarvam AI Saaras API with instant Groq Whisper fallback.
    """
    def __init__(self):
        self.api_key = settings.SARVAM_API_KEY
        self.provider = settings.STT_PROVIDER
        self.sarvam_url = "https://api.sarvam.ai/speech-to-text"

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language_code: Optional[str] = None
    ) -> Tuple[str, str, float]:
        """
        Transcribes input audio bytes into text.
        Returns: (transcript_text, detected_language, latency_ms)
        """
        t0 = time.perf_counter()
        lang = language_code or settings.STT_LANGUAGE_CODE

        # Detect audio container format from magic header bytes
        content_type = "audio/wav"
        ext = "wav"
        if audio_bytes.startswith(b"\x1aE\xdf\xa3"):
            content_type = "audio/webm"
            ext = "webm"
        elif audio_bytes.startswith(b"OggS"):
            content_type = "audio/ogg"
            ext = "ogg"
        elif audio_bytes.startswith(b"ID3") or audio_bytes.startswith(b"\xff\xfb"):
            content_type = "audio/mpeg"
            ext = "mp3"

        # 1. If Groq Whisper or Sarvam API key not set, use high-speed Groq Whisper Large v3 Turbo directly!
        if self.provider == "groq_whisper" or (settings.GROQ_API_KEY and (not self.api_key or self.provider != "sarvam")):
            print(f"[STTService] Transcribing real microphone audio ({len(audio_bytes)} bytes) using Groq Whisper Large v3 Turbo...")
            groq_text = await self._transcribe_with_groq(audio_bytes, ext, content_type)
            if groq_text:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                print(f"[STTService] Transcribed successfully in {elapsed_ms:.2f}ms: '{groq_text}'")
                return groq_text, "en-IN", round(elapsed_ms, 2)

        # 2. If Sarvam provider selected and API key is present, call Sarvam AI
        if self.api_key and self.provider == "sarvam":
            headers = {
                "api-subscription-key": self.api_key.strip()
            }
            files = {
                "file": (f"audio.{ext}", audio_bytes, content_type)
            }
            data = {
                "language_code": lang,
                "model": "saaras:v1"
            }

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(self.sarvam_url, headers=headers, files=files, data=data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        transcript = result.get("transcript", "").strip()
                        detected_lang = result.get("language_code", lang)
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        return transcript, detected_lang, round(elapsed_ms, 2)
                    else:
                        print(f"[STTService] Sarvam API status {response.status_code}: {response.text}. Falling back to Groq Whisper...")
            except Exception as e:
                print(f"[STTService] Sarvam exception: {e}. Falling back to Groq Whisper...")

        # 3. Automatic Groq Whisper fallback
        if settings.GROQ_API_KEY:
            groq_text = await self._transcribe_with_groq(audio_bytes, ext, content_type)
            if groq_text:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return groq_text, "en-IN", round(elapsed_ms, 2)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return "Could not transcribe audio. Please verify microphone audio or API keys.", lang, round(elapsed_ms, 2)

    async def _transcribe_with_groq(self, audio_bytes: bytes, ext: str, content_type: str) -> Optional[str]:
        """High-speed backup STT using Groq whisper-large-v3-turbo."""
        try:
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY.strip()}"}
            files = {"file": (f"audio.{ext}", audio_bytes, content_type)}
            data = {"model": "whisper-large-v3-turbo", "language": "en"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(url, headers=headers, files=files, data=data)
                if res.status_code == 200:
                    return res.json().get("text", "").strip()
                else:
                    print(f"[STTService] Groq Whisper API returned {res.status_code}: {res.text}")
        except Exception as err:
            print(f"[STTService] Groq Whisper call failed: {err}")
        return None

stt_service = STTService()
