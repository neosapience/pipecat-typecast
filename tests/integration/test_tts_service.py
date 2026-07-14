"""Integration tests for TypecastTTSService with mocked API responses."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from typecast.exceptions import BadRequestError
from typecast.models import TTSResponse

from pipecat_typecast import (
    OutputOptions,
    PresetPromptOptions,
    SmartPromptOptions,
    TypecastInputParams,
)
from pipecat_typecast.tts import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
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
        prompt = service._settings["prompt"]
        assert prompt.emotion_preset == "happy"
        assert prompt.emotion_intensity == 1.5

    @pytest.mark.integration
    def test_init_prefers_explicit_credentials(self, mock_env, mock_aiohttp_session):
        """Test explicit credentials override environment variables."""
        service = TypecastTTSService(
            aiohttp_session=mock_aiohttp_session,
            api_key="explicit-api-key",
            voice_id="tc_explicit_voice_id",
        )

        assert service._api_key == "explicit-api-key"
        assert service._settings["voice_id"] == "tc_explicit_voice_id"

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

        prompt = service._settings["prompt"]
        assert prompt.emotion_type == "smart"
        assert prompt.previous_text == "Hello!"

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
        async def async_iter():
            yield sample_audio_data

        service._client.text_to_speech_stream = MagicMock(return_value=async_iter())

        # Collect frames
        frames = []
        async for frame in service.run_tts("Hello, world!"):
            frames.append(frame)

        # Verify frames
        frame_types = [type(f) for f in frames]
        assert TTSStartedFrame in frame_types
        assert TTSStoppedFrame in frame_types

        service._client.text_to_speech_stream.assert_called_once()

    @pytest.mark.integration
    async def test_run_tts_api_error(self, service):
        """Test handling of API errors."""
        async def raise_error():
            raise BadRequestError("Bad request: Invalid request")
            yield b""

        service._client.text_to_speech_stream = MagicMock(return_value=raise_error())

        # Collect frames
        frames = []
        async for frame in service.run_tts("Hello"):
            frames.append(frame)

        # Should yield ErrorFrame
        error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
        assert len(error_frames) >= 1
        assert "Invalid request" in str(error_frames[0].error)

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
        async def async_iter():
            yield sample_audio_data

        service._client.text_to_speech_stream = MagicMock(return_value=async_iter())

        async for _ in service.run_tts("Test text"):
            pass

        request = service._client.text_to_speech_stream.call_args.args[0]

        assert request.text == "Test text"
        assert request.model == DEFAULT_MODEL
        assert request.voice_id == "tc_test_voice_id"
        assert request.prompt is not None
        assert request.output is not None

    @pytest.mark.integration
    async def test_run_tts_volume_uses_non_streaming_sdk(self, mock_env, mock_aiohttp_session):
        """Test that volume-based requests use the non-streaming SDK endpoint."""
        params = TypecastInputParams(
            output_options=OutputOptions(volume=110),
        )
        service = TypecastTTSService(
            aiohttp_session=mock_aiohttp_session,
            params=params,
        )
        service._client.text_to_speech = AsyncMock(
            return_value=TTSResponse(audio_data=b"RIFF" + b"\x00" * 80, duration=1.0)
        )

        frames = []
        async for frame in service.run_tts("Test text"):
            frames.append(frame)

        service._client.text_to_speech.assert_awaited_once()
        request = service._client.text_to_speech.call_args.args[0]
        assert request.output.volume == 110
        assert any(isinstance(frame, TTSAudioRawFrame) for frame in frames)

    @pytest.mark.integration
    async def test_run_tts_target_lufs_uses_non_streaming_sdk(
        self, mock_env, mock_aiohttp_session
    ):
        """Test that loudness-normalized requests use the non-streaming SDK endpoint."""
        params = TypecastInputParams(
            output_options=OutputOptions(target_lufs=-16.0),
        )
        service = TypecastTTSService(
            aiohttp_session=mock_aiohttp_session,
            params=params,
        )
        service._client.text_to_speech = AsyncMock(
            return_value=TTSResponse(audio_data=b"RIFF" + b"\x00" * 80, duration=1.0)
        )
        service._client.text_to_speech_stream = MagicMock()

        frames = []
        async for frame in service.run_tts("Test text"):
            frames.append(frame)

        service._client.text_to_speech.assert_awaited_once()
        service._client.text_to_speech_stream.assert_not_called()
        request = service._client.text_to_speech.call_args.args[0]
        assert request.output.target_lufs == -16.0
        assert any(isinstance(frame, TTSAudioRawFrame) for frame in frames)


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
