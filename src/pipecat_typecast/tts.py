#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Typecast text-to-speech service implementation."""

from __future__ import annotations

import os
from typing import AsyncGenerator, Dict, Literal, Optional, Union

import aiohttp
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts
from pydantic import BaseModel, Field, model_validator

AUTHORIZATION_HEADER = "X-API-KEY"
DEFAULT_BASE_URL = "https://api.typecast.ai/v1/text-to-speech"
DEFAULT_MODEL = "ssfm-v30"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_VOICE_ID = "tc_62a8975e695ad26f7fb514d1"

ISO2_TO_ISO3_LANGUAGE_MAP: Dict[str, str] = {
    # Common languages (ssfm-v21 & ssfm-v30)
    "en": "eng",
    "ko": "kor",
    "ja": "jpn",
    "zh": "zho",
    "es": "spa",
    "de": "deu",
    "fr": "fra",
    "it": "ita",
    "ru": "rus",
    "ar": "ara",
    "pt": "por",
    "nl": "nld",
    "pl": "pol",
    "sv": "swe",
    "tr": "tur",
    "hi": "hin",
    "th": "tha",
    "vi": "vie",
    "id": "ind",
    # Additional languages
    "el": "ell",  # Greek
    "ta": "tam",  # Tamil
    "tl": "tgl",  # Tagalog
    "fi": "fin",  # Finnish
    "sk": "slk",  # Slovak
    "hr": "hrv",  # Croatian
    "uk": "ukr",  # Ukrainian
    "da": "dan",  # Danish
    "ms": "msa",  # Malay
    "cs": "ces",  # Czech
    "bg": "bul",  # Bulgarian
    "ro": "ron",  # Romanian
    # ssfm-v30 additional languages
    "bn": "ben",  # Bengali
    "hu": "hun",  # Hungarian
    "no": "nor",  # Norwegian
    "pa": "pan",  # Punjabi
}


def language_to_typecast_language(language: Language) -> Optional[str]:
    """Convert Pipecat language enum values to Typecast ISO-639-3 codes.

    Args:
        language: Pipecat Language enum (e.g., Language.EN, Language.KO).

    Returns:
        ISO-639-3 language code string (e.g., 'eng', 'kor'), or None if unsupported.

    Example:
        >>> language_to_typecast_language(Language.EN)
        'eng'
        >>> language_to_typecast_language(Language.KO)
        'kor'
    """
    if not language:
        return None

    base_code = str(language.value).split("-")[0].lower()
    return ISO2_TO_ISO3_LANGUAGE_MAP.get(base_code)


class PromptOptions(BaseModel):
    """Legacy emotion control options for ssfm-v21 model."""

    emotion_preset: str = Field(default="normal")
    emotion_intensity: float = Field(default=1.0, ge=0.0, le=2.0)


class PresetPromptOptions(BaseModel):
    """Preset-based emotion control for ssfm-v30 model.

    Provides direct control over emotion using predefined presets.
    Available presets for ssfm-v30: normal, happy, sad, angry, whisper, toneup, tonedown
    """

    emotion_type: Literal["preset"] = Field(
        default="preset",
        description="Must be 'preset' for preset-based emotion control",
    )
    emotion_preset: str = Field(
        default="normal",
        description="Emotion preset to apply",
    )
    emotion_intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Intensity of the emotion (0.0-2.0)",
    )


class SmartPromptOptions(BaseModel):
    """Context-aware emotion inference for ssfm-v30 model.

    Automatically infers appropriate emotion based on surrounding text context.
    Useful for natural, context-appropriate emotional delivery.
    """

    emotion_type: Literal["smart"] = Field(
        default="smart",
        description="Must be 'smart' for context-aware emotion inference",
    )
    previous_text: Optional[str] = Field(
        default=None,
        description="Text that comes BEFORE the main text (max 2000 chars)",
        max_length=2000,
    )
    next_text: Optional[str] = Field(
        default=None,
        description="Text that comes AFTER the main text (max 2000 chars)",
        max_length=2000,
    )


# Union type for all prompt options
TypecastPromptOptions = Union[PromptOptions, PresetPromptOptions, SmartPromptOptions]


class OutputOptions(BaseModel):
    """Audio output configuration supported by Typecast.

    `volume` and `target_lufs` are mutually exclusive server-side — any
    request that sets both returns a 4xx. We surface that contract here
    by defaulting `volume` to ``None`` (the server-side default of 100
    kicks in when neither is set) and rejecting explicit combinations
    via a model validator.
    """

    volume: Optional[int] = Field(default=None, ge=0, le=200)
    audio_pitch: int = Field(default=0, ge=-12, le=12)
    audio_tempo: float = Field(default=1.0, ge=0.5, le=2.0)
    audio_format: str = Field(default="wav")
    target_lufs: Optional[float] = Field(
        default=None,
        ge=-70.0,
        le=0.0,
        description=(
            "Absolute loudness normalization target in LUFS. Mutually "
            "exclusive with `volume` — set one or the other, not both."
        ),
    )

    model_config = {"validate_assignment": True}

    @model_validator(mode="after")
    def _check_mutual_exclusion(self) -> "OutputOptions":
        # Both unset (None / None) is the common case — defer to the server
        # default of 100. The model rejects only an *explicit* pairing so the
        # error surfaces locally before the request goes out.
        if self.target_lufs is not None and self.volume is not None:
            raise ValueError(
                "Volume and target_lufs are mutually exclusive. "
                "Set only one of them — leave volume unset when using target_lufs."
            )
        return self


class TypecastInputParams(BaseModel):
    """Input parameters for Typecast TTS configuration.

    Supports both legacy ssfm-v21 and new ssfm-v30 models:
    - For ssfm-v21: Use PromptOptions with emotion_preset and emotion_intensity
    - For ssfm-v30: Use PresetPromptOptions or SmartPromptOptions

    Example (ssfm-v30 with preset):
        >>> params = TypecastInputParams(
        ...     prompt_options=PresetPromptOptions(
        ...         emotion_preset="happy",
        ...         emotion_intensity=1.5,
        ...     )
        ... )

    Example (ssfm-v30 with smart context):
        >>> params = TypecastInputParams(
        ...     prompt_options=SmartPromptOptions(
        ...         previous_text="I just got great news!",
        ...         next_text="I can't wait to share it!",
        ...     )
        ... )
    """

    language: Optional[Language] = Language.EN
    seed: Optional[int] = Field(default=None, ge=0)
    prompt_options: TypecastPromptOptions = Field(default_factory=PresetPromptOptions)
    output_options: OutputOptions = Field(default_factory=OutputOptions)


class TypecastTTSService(TTSService):
    """HTTP-based Typecast TTS service for Pipecat pipelines.

    Provides high-quality neural text-to-speech with emotion control and
    audio customization options.

    Attributes:
        InputParams: Configuration model for TTS parameters.
    """

    InputParams = TypecastInputParams

    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession,
        api_key: str = None,
        voice_id: str = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        sample_rate: Optional[int] = DEFAULT_SAMPLE_RATE,
        params: Optional[TypecastInputParams] = None,
        **kwargs,
    ):
        """Initialize the Typecast TTS service.

        Args:
            aiohttp_session: Active aiohttp client session for API requests.
            api_key: Typecast API key. Falls back to TYPECAST_API_KEY env var.
            voice_id: Voice ID to use. Falls back to TYPECAST_VOICE_ID env var.
            model: Typecast model version (default: ssfm-v30).
            base_url: API endpoint URL.
            sample_rate: Audio sample rate in Hz (default: 44100).
            params: Advanced configuration parameters.
            **kwargs: Additional arguments passed to TTSService.

        Raises:
            ValueError: If api_key is not provided and not in environment.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        api_key = os.getenv("TYPECAST_API_KEY")
        voice_id = os.getenv("TYPECAST_VOICE_ID", DEFAULT_VOICE_ID)

        if not api_key:
            raise ValueError("Typecast API key is required.")

        params = params or TypecastTTSService.InputParams()

        self._session = aiohttp_session
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

        language_code = (
            self.language_to_service_language(params.language) if params.language else None
        )
        prompt_config = params.prompt_options.model_dump(exclude_none=True)
        output_config = params.output_options.model_dump(exclude_none=True)

        self._settings = {
            "base_url": self._base_url,
            "model": model,
            "voice_id": voice_id,
            "language": language_code,
            "seed": params.seed,
            "prompt": prompt_config,
            "output": output_config,
        }

        self.set_model_name(model)
        self.set_voice(voice_id)

    def can_generate_metrics(self) -> bool:
        """Return whether the service can generate metrics."""
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert Pipecat Language enum to Typecast ISO-639-3 code."""
        return language_to_typecast_language(language)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech audio using the Typecast REST API."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        headers = {
            "Content-Type": "application/json",
            AUTHORIZATION_HEADER: self._api_key,
        }

        prompt_options = dict(self._settings.get("prompt") or {})
        output_options = dict(self._settings.get("output") or {})

        audio_format = output_options.get("audio_format", "wav")
        if audio_format != "wav":
            error_message = (
                f"TypecastTTSService only supports 'wav' audio_format, not '{audio_format}'."
            )
            logger.error(error_message)
            yield ErrorFrame(error_message)
            return

        output_options["audio_format"] = "wav"

        payload: Dict[str, object] = {
            "text": text,
            "model": self._settings.get("model", DEFAULT_MODEL),
            "voice_id": self._settings.get("voice_id", DEFAULT_VOICE_ID),
            "prompt": prompt_options,
            "output": output_options,
        }

        language = self._settings.get("language")
        if language:
            payload["language"] = language

        seed = self._settings.get("seed")
        if seed is not None:
            payload["seed"] = seed

        base_url = self._settings.get("base_url", self._base_url)
        ttfb_stopped = False

        try:
            await self.start_ttfb_metrics()

            async with self._session.post(base_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_detail: str
                    try:
                        error_body = await response.json()
                        error_detail = error_body.get("message", str(error_body))
                    except Exception:
                        error_detail = await response.text()

                    logger.error(
                        f"{self}: Typecast API error (status {response.status}): {error_detail}"
                    )
                    yield ErrorFrame(
                        f"Error from Typecast API: status {response.status}, detail: {error_detail}"
                    )
                    return

                await self.start_tts_usage_metrics(text)
                yield TTSStartedFrame()

                # Use default chunk size if sample_rate not set yet
                # (sample_rate * 0.5s * 2 bytes/sample)
                effective_chunk_size = self.chunk_size or int(
                    DEFAULT_SAMPLE_RATE * 0.5 * 2
                )
                chunk_iterator = response.content.iter_chunked(effective_chunk_size)
                first_frame = True

                async for frame in self._stream_audio_frames_from_iterator(
                    chunk_iterator, strip_wav_header=True
                ):
                    if first_frame:
                        await self.stop_ttfb_metrics()
                        ttfb_stopped = True
                        first_frame = False
                    yield frame

                if first_frame and not ttfb_stopped:
                    await self.stop_ttfb_metrics()
                    ttfb_stopped = True

        except Exception as exc:
            logger.exception(f"{self}: Error generating audio: {exc}")
            yield ErrorFrame(str(exc))
        finally:
            if not ttfb_stopped:
                await self.stop_ttfb_metrics()
            logger.debug(f"{self}: Finished TTS [{text}]")
            yield TTSStoppedFrame()
