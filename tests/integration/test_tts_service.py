"""Integration tests for TypecastTTSService with mocked API responses."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

from pipecat_typecast import PresetPromptOptions, SmartPromptOptions, TypecastInputParams
from pipecat_typecast.tts import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VOICE_ID,
    TypecastTTSService,
)


class TestTypecastTTSServiceInit:
    """Tests for TypecastTTSService initialization."""

    @pytest.mark.integration
    def test_init_with_env_vars(self, mock_env, mock_aiohttp_session):
        """Test initialization with environment variables."""
        service = TypecastTTSService(aiohttp_session=mock_aiohttp_session)

        assert service._api_key == "test-api-key-12345"
        assert service._settings["voice_id"] == "tc_test_voice_id"
        assert service._settings["model"] == DEFAULT_MODEL
        assert service._settings["base_url"] == DEFAULT_BASE_URL

    @pytest.mark.integration
    def test_init_without_api_key_raises(self, mock_aiohttp_session, monkeypatch):
        """Test that missing API key raises ValueError."""
        monkeypatch.delenv("TYPECAST_API_KEY", raising=False)

        with pytest.raises(ValueError, match="API key is required"):
            TypecastTTSService(aiohttp_session=mock_aiohttp_session)

    @pytest.mark.integration
    def test_init_with_custom_params(self, mock_env, mock_aiohttp_session):
        """Test initialization with custom parameters."""
        params = TypecastInputParams(
            prompt_options=PresetPromptOptions(
                emotion_preset="happy",
                emotion_intensity=1.5,
            )
        )

        service = TypecastTTSService(
            aiohttp_session=mock_aiohttp_session,
            model="ssfm-v21",
            sample_rate=22050,
            params=params,
        )

        assert service._settings["model"] == "ssfm-v21"
        assert service._settings["prompt"]["emotion_preset"] == "happy"
        assert service._settings["prompt"]["emotion_intensity"] == 1.5

    @pytest.mark.integration
    def test_init_with_smart_prompt(self, mock_env, mock_aiohttp_session):
        """Test initialization with SmartPromptOptions."""
        params = TypecastInputParams(
            prompt_options=SmartPromptOptions(
                previous_text="Hello!",
                next_text="Goodbye!",
            )
        )

        service = TypecastTTSService(
            aiohttp_session=mock_aiohttp_session,
            params=params,
        )

        assert service._settings["prompt"]["emotion_type"] == "smart"
        assert service._settings["prompt"]["previous_text"] == "Hello!"

    @pytest.mark.integration
    def test_can_generate_metrics(self, mock_env, mock_aiohttp_session):
        """Test that service can generate metrics."""
        service = TypecastTTSService(aiohttp_session=mock_aiohttp_session)
        assert service.can_generate_metrics() is True


class TestTypecastTTSServiceRunTTS:
    """Tests for TypecastTTSService.run_tts method."""

    @pytest.fixture
    def service(self, mock_env, mock_aiohttp_session):
        """Create a TypecastTTSService instance for testing."""
        return TypecastTTSService(aiohttp_session=mock_aiohttp_session)

    @pytest.mark.integration
    async def test_run_tts_success(self, service, sample_audio_data):
        """Test successful TTS generation."""
        # Create mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content = MagicMock()

        # Create async iterator for chunked content
        async def async_iter():
            yield sample_audio_data

        mock_response.content.iter_chunked = MagicMock(return_value=async_iter())

        # Mock context manager
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        mock_context.__aexit__.return_value = None

        service._session.post = MagicMock(return_value=mock_context)

        # Collect frames
        frames = []
        async for frame in service.run_tts("Hello, world!"):
            frames.append(frame)

        # Verify frames
        frame_types = [type(f) for f in frames]
        assert TTSStartedFrame in frame_types
        assert TTSStoppedFrame in frame_types

        # Verify API was called
        service._session.post.assert_called_once()
        call_args = service._session.post.call_args
        assert call_args[0][0] == DEFAULT_BASE_URL

    @pytest.mark.integration
    async def test_run_tts_api_error(self, service):
        """Test handling of API errors."""
        # Create mock error response
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value={"message": "Invalid request"})

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        mock_context.__aexit__.return_value = None

        service._session.post = MagicMock(return_value=mock_context)

        # Collect frames
        frames = []
        async for frame in service.run_tts("Hello"):
            frames.append(frame)

        # Should yield ErrorFrame
        error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
        assert len(error_frames) >= 1
        assert "400" in str(error_frames[0].error)

    @pytest.mark.integration
    async def test_run_tts_invalid_audio_format(self, mock_env, mock_aiohttp_session):
        """Test that non-WAV audio format yields error."""
        from pipecat_typecast import OutputOptions

        params = TypecastInputParams(
            output_options=OutputOptions(audio_format="mp3")
        )

        service = TypecastTTSService(
            aiohttp_session=mock_aiohttp_session,
            params=params,
        )

        frames = []
        async for frame in service.run_tts("Hello"):
            frames.append(frame)

        # Should yield ErrorFrame for unsupported format
        error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
        assert len(error_frames) >= 1
        assert "wav" in str(error_frames[0].error).lower()

    @pytest.mark.integration
    async def test_run_tts_request_payload(self, service, sample_audio_data):
        """Test that request payload is correctly constructed."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content = MagicMock()

        async def async_iter():
            yield sample_audio_data

        mock_response.content.iter_chunked = MagicMock(return_value=async_iter())

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        mock_context.__aexit__.return_value = None

        service._session.post = MagicMock(return_value=mock_context)

        async for _ in service.run_tts("Test text"):
            pass

        # Verify payload structure
        call_kwargs = service._session.post.call_args[1]
        payload = call_kwargs["json"]

        assert payload["text"] == "Test text"
        assert payload["model"] == DEFAULT_MODEL
        assert payload["voice_id"] == "tc_test_voice_id"
        assert "prompt" in payload
        assert "output" in payload

        # Verify headers
        headers = call_kwargs["headers"]
        assert headers["X-API-KEY"] == "test-api-key-12345"
        assert headers["Content-Type"] == "application/json"


class TestTypecastTTSServiceLanguage:
    """Tests for language handling in TypecastTTSService."""

    @pytest.mark.integration
    def test_language_in_settings(self, mock_env, mock_aiohttp_session):
        """Test that language is correctly set in settings."""
        from pipecat.transcriptions.language import Language

        params = TypecastInputParams(language=Language.KO)

        service = TypecastTTSService(
            aiohttp_session=mock_aiohttp_session,
            params=params,
        )

        assert service._settings["language"] == "kor"

    @pytest.mark.integration
    def test_language_none(self, mock_env, mock_aiohttp_session):
        """Test that None language is handled correctly."""
        params = TypecastInputParams(language=None)

        service = TypecastTTSService(
            aiohttp_session=mock_aiohttp_session,
            params=params,
        )

        assert service._settings["language"] is None
