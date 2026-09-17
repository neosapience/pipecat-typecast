"""Validate the optional remaining-silence setting."""

import pytest
from pydantic import ValidationError

from pipecat_typecast.tts import OutputOptions


def test_optional_silence_contract():
    """Preserve zero and reject values outside the API contract."""
    assert OutputOptions().remove_silence_ms is None
    for value in [0, 300, 1000]:
        assert OutputOptions(remove_silence_ms=value).remove_silence_ms == value
    for value in [True, False, "100", 0.5, -1, 1001]:
        with pytest.raises(ValidationError):
            OutputOptions(remove_silence_ms=value)
