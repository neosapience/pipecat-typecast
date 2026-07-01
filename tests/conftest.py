"""Pytest configuration and shared fixtures."""

import os
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def mock_voice_id():
    """Provide a mock voice ID for testing."""
    return "tc_test_voice_id"


@pytest.fixture
def mock_env(mock_api_key, mock_voice_id, monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("TYPECAST_API_KEY", mock_api_key)
    monkeypatch.setenv("TYPECAST_VOICE_ID", mock_voice_id)


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp ClientSession."""
    session = MagicMock(spec=aiohttp.ClientSession)
    return session


@pytest.fixture
def sample_wav_header():
    """Generate a minimal valid WAV header (44 bytes)."""
    import struct

    # RIFF header
    riff = b"RIFF"
    file_size = struct.pack("<I", 36 + 1000)  # 36 + data size
    wave = b"WAVE"

    # fmt subchunk
    fmt = b"fmt "
    fmt_size = struct.pack("<I", 16)  # PCM
    audio_format = struct.pack("<H", 1)  # PCM
    num_channels = struct.pack("<H", 1)  # Mono
    sample_rate = struct.pack("<I", 44100)
    byte_rate = struct.pack("<I", 44100 * 1 * 2)  # SampleRate * NumChannels * BitsPerSample/8
    block_align = struct.pack("<H", 2)  # NumChannels * BitsPerSample/8
    bits_per_sample = struct.pack("<H", 16)

    # data subchunk
    data = b"data"
    data_size = struct.pack("<I", 1000)

    header = (
        riff
        + file_size
        + wave
        + fmt
        + fmt_size
        + audio_format
        + num_channels
        + sample_rate
        + byte_rate
        + block_align
        + bits_per_sample
        + data
        + data_size
    )

    return header


@pytest.fixture
def sample_audio_data(sample_wav_header):
    """Generate sample audio data with WAV header."""
    # Generate some fake audio samples
    audio_samples = bytes([0x00, 0x80] * 500)  # 1000 bytes of silent audio
    return sample_wav_header + audio_samples


# E2E test fixtures
@pytest.fixture
def real_api_key():
    """Get real API key from environment for E2E tests."""
    api_key = os.getenv("TYPECAST_API_KEY")
    if not api_key:
        pytest.skip("TYPECAST_API_KEY not set - skipping E2E test")
    return api_key


@pytest.fixture
def real_voice_id():
    """Get real voice ID from environment for E2E tests."""
    return os.getenv("TYPECAST_VOICE_ID", "tc_672c5f5ce59fac2a48faeaee")
