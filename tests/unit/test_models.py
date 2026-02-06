"""Unit tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from pipecat_typecast import (
    OutputOptions,
    PresetPromptOptions,
    PromptOptions,
    SmartPromptOptions,
    TypecastInputParams,
)


class TestPromptOptions:
    """Tests for legacy PromptOptions model (ssfm-v21)."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values are set correctly."""
        options = PromptOptions()
        assert options.emotion_preset == "normal"
        assert options.emotion_intensity == 1.0

    @pytest.mark.unit
    def test_custom_values(self):
        """Test custom emotion settings."""
        options = PromptOptions(emotion_preset="happy", emotion_intensity=1.5)
        assert options.emotion_preset == "happy"
        assert options.emotion_intensity == 1.5

    @pytest.mark.unit
    def test_emotion_intensity_bounds(self):
        """Test emotion intensity validation bounds (0.0-2.0)."""
        # Valid bounds
        PromptOptions(emotion_intensity=0.0)
        PromptOptions(emotion_intensity=2.0)

        # Invalid bounds
        with pytest.raises(ValidationError):
            PromptOptions(emotion_intensity=-0.1)

        with pytest.raises(ValidationError):
            PromptOptions(emotion_intensity=2.1)


class TestPresetPromptOptions:
    """Tests for PresetPromptOptions model (ssfm-v30)."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values are set correctly."""
        options = PresetPromptOptions()
        assert options.emotion_type == "preset"
        assert options.emotion_preset == "normal"
        assert options.emotion_intensity == 1.0

    @pytest.mark.unit
    def test_emotion_presets(self):
        """Test various emotion presets."""
        presets = ["normal", "happy", "sad", "angry", "whisper", "toneup", "tonedown"]
        for preset in presets:
            options = PresetPromptOptions(emotion_preset=preset)
            assert options.emotion_preset == preset

    @pytest.mark.unit
    def test_model_dump(self):
        """Test model serialization."""
        options = PresetPromptOptions(emotion_preset="happy", emotion_intensity=1.5)
        dumped = options.model_dump()
        assert dumped["emotion_type"] == "preset"
        assert dumped["emotion_preset"] == "happy"
        assert dumped["emotion_intensity"] == 1.5


class TestSmartPromptOptions:
    """Tests for SmartPromptOptions model (ssfm-v30 context-aware)."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values are set correctly."""
        options = SmartPromptOptions()
        assert options.emotion_type == "smart"
        assert options.previous_text is None
        assert options.next_text is None

    @pytest.mark.unit
    def test_with_context(self):
        """Test with context text provided."""
        options = SmartPromptOptions(
            previous_text="I just got great news!",
            next_text="I can't wait to share it!",
        )
        assert options.previous_text == "I just got great news!"
        assert options.next_text == "I can't wait to share it!"

    @pytest.mark.unit
    def test_model_dump_excludes_none(self):
        """Test that None values are excluded from dump."""
        options = SmartPromptOptions(previous_text="Hello")
        dumped = options.model_dump(exclude_none=True)
        assert "previous_text" in dumped
        assert "next_text" not in dumped

    @pytest.mark.unit
    def test_text_max_length(self):
        """Test text length validation (max 2000 chars)."""
        long_text = "a" * 2000
        options = SmartPromptOptions(previous_text=long_text)
        assert len(options.previous_text) == 2000

        # Over limit
        with pytest.raises(ValidationError):
            SmartPromptOptions(previous_text="a" * 2001)


class TestOutputOptions:
    """Tests for OutputOptions model."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values are set correctly."""
        options = OutputOptions()
        assert options.volume == 100
        assert options.audio_pitch == 0
        assert options.audio_tempo == 1.0
        assert options.audio_format == "wav"

    @pytest.mark.unit
    def test_volume_bounds(self):
        """Test volume validation bounds (0-200)."""
        OutputOptions(volume=0)
        OutputOptions(volume=200)

        with pytest.raises(ValidationError):
            OutputOptions(volume=-1)

        with pytest.raises(ValidationError):
            OutputOptions(volume=201)

    @pytest.mark.unit
    def test_audio_pitch_bounds(self):
        """Test audio pitch validation bounds (-12 to 12)."""
        OutputOptions(audio_pitch=-12)
        OutputOptions(audio_pitch=12)

        with pytest.raises(ValidationError):
            OutputOptions(audio_pitch=-13)

        with pytest.raises(ValidationError):
            OutputOptions(audio_pitch=13)

    @pytest.mark.unit
    def test_audio_tempo_bounds(self):
        """Test audio tempo validation bounds (0.5-2.0)."""
        OutputOptions(audio_tempo=0.5)
        OutputOptions(audio_tempo=2.0)

        with pytest.raises(ValidationError):
            OutputOptions(audio_tempo=0.4)

        with pytest.raises(ValidationError):
            OutputOptions(audio_tempo=2.1)


class TestTypecastInputParams:
    """Tests for TypecastInputParams model."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values are set correctly."""
        from pipecat.transcriptions.language import Language

        params = TypecastInputParams()
        assert params.language == Language.EN
        assert params.seed is None
        assert isinstance(params.prompt_options, PresetPromptOptions)
        assert isinstance(params.output_options, OutputOptions)

    @pytest.mark.unit
    def test_with_preset_prompt(self):
        """Test with PresetPromptOptions."""
        params = TypecastInputParams(
            prompt_options=PresetPromptOptions(
                emotion_preset="happy",
                emotion_intensity=1.5,
            )
        )
        assert params.prompt_options.emotion_preset == "happy"

    @pytest.mark.unit
    def test_with_smart_prompt(self):
        """Test with SmartPromptOptions."""
        params = TypecastInputParams(
            prompt_options=SmartPromptOptions(
                previous_text="Hello!",
            )
        )
        assert params.prompt_options.emotion_type == "smart"
        assert params.prompt_options.previous_text == "Hello!"

    @pytest.mark.unit
    def test_seed_validation(self):
        """Test seed validation (must be >= 0)."""
        TypecastInputParams(seed=0)
        TypecastInputParams(seed=12345)

        with pytest.raises(ValidationError):
            TypecastInputParams(seed=-1)
