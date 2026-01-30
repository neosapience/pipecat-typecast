"""End-to-end tests for TypecastTTSService with real API calls.

These tests require valid TYPECAST_API_KEY environment variable.
Run with: pytest -m e2e
"""

import os

import aiohttp
import pytest
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.transcriptions.language import Language

from pipecat_typecast import (
    PresetPromptOptions,
    SmartPromptOptions,
    TypecastInputParams,
    TypecastTTSService,
)


@pytest.fixture
async def aiohttp_session():
    """Create a real aiohttp session for E2E tests."""
    async with aiohttp.ClientSession() as session:
        yield session


@pytest.fixture
def tts_service(real_api_key, real_voice_id, aiohttp_session):
    """Create a TypecastTTSService with real credentials."""
    # Set environment variables for the service
    os.environ["TYPECAST_API_KEY"] = real_api_key
    os.environ["TYPECAST_VOICE_ID"] = real_voice_id

    return TypecastTTSService(aiohttp_session=aiohttp_session)


class TestE2ETTSGeneration:
    """End-to-end tests for TTS audio generation."""

    @pytest.mark.e2e
    async def test_generate_english_speech(self, real_api_key, real_voice_id, aiohttp_session):
        """Test generating English speech with real API."""
        os.environ["TYPECAST_API_KEY"] = real_api_key
        os.environ["TYPECAST_VOICE_ID"] = real_voice_id

        service = TypecastTTSService(aiohttp_session=aiohttp_session)

        frames = []
        async for frame in service.run_tts("Hello, this is a test."):
            frames.append(frame)

        # Verify frame sequence
        frame_types = [type(f).__name__ for f in frames]

        assert "TTSStartedFrame" in frame_types, f"Missing TTSStartedFrame. Got: {frame_types}"
        assert "TTSStoppedFrame" in frame_types, f"Missing TTSStoppedFrame. Got: {frame_types}"

        # Check for errors
        error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
        assert len(error_frames) == 0, f"Unexpected errors: {[f.error for f in error_frames]}"

        # Verify audio data was received
        audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio_frames) > 0, "No audio frames received"

        # Verify audio data is not empty
        total_audio_bytes = sum(len(f.audio) for f in audio_frames)
        assert total_audio_bytes > 0, "Audio data is empty"

    @pytest.mark.e2e
    async def test_generate_korean_speech(self, real_api_key, real_voice_id, aiohttp_session):
        """Test generating Korean speech with real API."""
        os.environ["TYPECAST_API_KEY"] = real_api_key
        os.environ["TYPECAST_VOICE_ID"] = real_voice_id

        params = TypecastInputParams(language=Language.KO)
        service = TypecastTTSService(
            aiohttp_session=aiohttp_session,
            params=params,
        )

        frames = []
        async for frame in service.run_tts("안녕하세요, 테스트입니다."):
            frames.append(frame)

        # Verify no errors
        error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
        assert len(error_frames) == 0, f"Unexpected errors: {[f.error for f in error_frames]}"

        # Verify audio was generated
        audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio_frames) > 0, "No audio frames received"

    @pytest.mark.e2e
    async def test_generate_with_emotion_preset(
        self, real_api_key, real_voice_id, aiohttp_session
    ):
        """Test generating speech with emotion preset."""
        os.environ["TYPECAST_API_KEY"] = real_api_key
        os.environ["TYPECAST_VOICE_ID"] = real_voice_id

        params = TypecastInputParams(
            prompt_options=PresetPromptOptions(
                emotion_preset="happy",
                emotion_intensity=1.5,
            )
        )

        service = TypecastTTSService(
            aiohttp_session=aiohttp_session,
            params=params,
        )

        frames = []
        async for frame in service.run_tts("I am so happy today!"):
            frames.append(frame)

        error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
        assert len(error_frames) == 0, f"Unexpected errors: {[f.error for f in error_frames]}"

        audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio_frames) > 0, "No audio frames received"

    @pytest.mark.e2e
    async def test_generate_with_smart_prompt(
        self, real_api_key, real_voice_id, aiohttp_session
    ):
        """Test generating speech with smart context-aware emotion."""
        os.environ["TYPECAST_API_KEY"] = real_api_key
        os.environ["TYPECAST_VOICE_ID"] = real_voice_id

        params = TypecastInputParams(
            prompt_options=SmartPromptOptions(
                previous_text="I just received amazing news!",
                next_text="I can't believe it!",
            )
        )

        service = TypecastTTSService(
            aiohttp_session=aiohttp_session,
            params=params,
        )

        frames = []
        async for frame in service.run_tts("This is incredible!"):
            frames.append(frame)

        error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
        assert len(error_frames) == 0, f"Unexpected errors: {[f.error for f in error_frames]}"

        audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio_frames) > 0, "No audio frames received"


class TestE2EErrorHandling:
    """End-to-end tests for error handling."""

    @pytest.mark.e2e
    async def test_invalid_api_key(self, aiohttp_session, monkeypatch):
        """Test that invalid API key returns appropriate error."""
        monkeypatch.setenv("TYPECAST_API_KEY", "invalid-api-key")

        service = TypecastTTSService(aiohttp_session=aiohttp_session)

        frames = []
        async for frame in service.run_tts("Hello"):
            frames.append(frame)

        # Should have error frame
        error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
        assert len(error_frames) >= 1, "Expected error frame for invalid API key"

    @pytest.mark.e2e
    async def test_empty_text(self, real_api_key, real_voice_id, aiohttp_session):
        """Test behavior with empty text."""
        os.environ["TYPECAST_API_KEY"] = real_api_key
        os.environ["TYPECAST_VOICE_ID"] = real_voice_id

        service = TypecastTTSService(aiohttp_session=aiohttp_session)

        frames = []
        async for frame in service.run_tts(""):
            frames.append(frame)

        # The API may return an error for empty text
        # This test documents the actual behavior
        frame_types = [type(f).__name__ for f in frames]
        assert len(frames) > 0, "Expected some frames even for empty text"


class TestE2EMultipleLanguages:
    """End-to-end tests for multiple language support."""

    @pytest.mark.e2e
    @pytest.mark.parametrize(
        "language,text",
        [
            (Language.EN, "Hello, how are you?"),
            (Language.KO, "안녕하세요, 어떻게 지내세요?"),
            (Language.JA, "こんにちは、お元気ですか？"),
            (Language.ZH, "你好，你好吗？"),
            (Language.ES, "Hola, ¿cómo estás?"),
            (Language.FR, "Bonjour, comment allez-vous?"),
            (Language.DE, "Hallo, wie geht es Ihnen?"),
        ],
    )
    async def test_multilingual_speech(
        self, real_api_key, real_voice_id, aiohttp_session, language, text
    ):
        """Test generating speech in multiple languages."""
        os.environ["TYPECAST_API_KEY"] = real_api_key
        os.environ["TYPECAST_VOICE_ID"] = real_voice_id

        params = TypecastInputParams(language=language)
        service = TypecastTTSService(
            aiohttp_session=aiohttp_session,
            params=params,
        )

        frames = []
        async for frame in service.run_tts(text):
            frames.append(frame)

        error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
        assert len(error_frames) == 0, f"Error for {language}: {[f.error for f in error_frames]}"

        audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio_frames) > 0, f"No audio for {language}"
