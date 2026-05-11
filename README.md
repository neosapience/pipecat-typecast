<div align="center">

# pipecat-ai-typecast

**Typecast TTS Integration for Pipecat AI Pipelines**

[![PyPI version](https://img.shields.io/pypi/v/pipecat-ai-typecast.svg)](https://pypi.org/project/pipecat-ai-typecast/)
[![Python](https://img.shields.io/pypi/pyversions/pipecat-ai-typecast.svg)](https://pypi.org/project/pipecat-ai-typecast/)
[![License](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](LICENSE)

Add high-quality neural voices from [Typecast](https://typecast.ai/) to your Pipecat AI pipelines.

[Installation](#installation) | [Quick Start](#quick-start) | [Configuration](#configuration) | [Examples](#running-the-example)

</div>

---

## Installation

```bash
pip install pipecat-ai-typecast
```

## Prerequisites

| Variable | Required | Description |
|----------|----------|-------------|
| `TYPECAST_API_KEY` | Yes | [Get your API key here](https://typecast.ai/developers/api/api-key) |
| `TYPECAST_VOICE_ID` | No | Voice override (defaults to `tc_672c5f5ce59fac2a48faeaee`) |

## Quick Start

`TypecastTTSService` integrates Typecast's streaming text-to-speech into a Pipecat pipeline. It converts LLM text output into expressive speech while leveraging Pipecat's transport, STT, and turn-taking stack.

```python
import os, aiohttp
from pipecat.pipeline.pipeline import Pipeline
from pipecat_typecast.tts import TypecastTTSService

async with aiohttp.ClientSession() as session:
    llm = ...
    stt = ...
    tts = TypecastTTSService(
        aiohttp_session=session,
        api_key=os.getenv("TYPECAST_API_KEY"),
        voice_id=os.getenv("TYPECAST_VOICE_ID", "tc_672c5f5ce59fac2a48faeaee"),
    )

    pipeline = Pipeline([
        transport.input(),               # audio/user input
        stt,                             # speech to text
        context_aggregator.user(),       # add user text to context
        llm,                             # LLM generates response
        tts,                             # Typecast TTS synthesis
        transport.output(),              # stream audio back to user
        context_aggregator.assistant(),  # store assistant response
    ])
```

See [`example.py`](example.py) for a complete working example including event handlers and transport setup.

---

## Configuration

`TypecastTTSService` exposes structured parameter models for emotion and audio control.

### ssfm-v30 (Default) - Preset Emotion Control

```python
from pipecat_typecast.tts import (
    TypecastTTSService,
    TypecastInputParams,
    PresetPromptOptions,
    OutputOptions,
)

params = TypecastInputParams(
    prompt_options=PresetPromptOptions(
        emotion_preset="happy",      # normal | happy | sad | angry | whisper | toneup | tonedown
        emotion_intensity=1.3,       # 0.0 - 2.0
    ),
    output_options=OutputOptions(
        volume=110,                  # 0 - 200 (percent). Defaults to None
                                     # so the server-side default of 100
                                     # kicks in when neither this nor
                                     # `target_lufs` is set.
        audio_pitch=2,               # -12 to 12 (semitones)
        audio_tempo=1.05,            # 0.5 - 2.0 (playback speed)
        audio_format="wav",          # Only 'wav' supported
        # target_lufs=-16.0,         # Absolute loudness target in LUFS
                                     # (-70.0 to 0.0). Mutually exclusive
                                     # with `volume` — leave `volume` unset
                                     # when using `target_lufs`.
    ),
)

tts = TypecastTTSService(
    aiohttp_session=session,
    api_key=os.getenv("TYPECAST_API_KEY"),
    voice_id="tc_672c5f5ce59fac2a48faeaee",
    model="ssfm-v30",
    params=params,
)
```

### ssfm-v30 - Smart Emotion Control

For context-aware emotional delivery, use `SmartPromptOptions` which infers emotion from surrounding text:

```python
from pipecat_typecast.tts import (
    TypecastTTSService,
    TypecastInputParams,
    SmartPromptOptions,
)

params = TypecastInputParams(
    prompt_options=SmartPromptOptions(
        previous_text="I just got the best news ever!",   # max 2000 chars
        next_text="I can't wait to share this with everyone!",
    ),
)

tts = TypecastTTSService(
    aiohttp_session=session,
    api_key=os.getenv("TYPECAST_API_KEY"),
    params=params,
)
```

### Legacy Model (ssfm-v21)

<details>
<summary>For backward compatibility with ssfm-v21</summary>

```python
from pipecat_typecast.tts import (
    TypecastTTSService,
    TypecastInputParams,
    PromptOptions,
)

params = TypecastInputParams(
    prompt_options=PromptOptions(
        emotion_preset="happy",      # normal | happy | sad | angry
        emotion_intensity=1.3,
    ),
)

tts = TypecastTTSService(
    aiohttp_session=session,
    api_key=os.getenv("TYPECAST_API_KEY"),
    model="ssfm-v21",
    params=params,
)
```

</details>

### Parameter Reference

| Parameter | Range | Description |
|-----------|-------|-------------|
| `emotion_preset` | varies by voice | ssfm-v30 adds: `whisper`, `toneup`, `tonedown` |
| `emotion_intensity` | 0.0 - 2.0 | Values > 1.0 increase expressiveness |
| `audio_pitch` | -12 to 12 | Semitone adjustment |
| `audio_tempo` | 0.5 - 2.0 | Recommended: 0.85 - 1.15 |
| `volume` | 0 - 200 | Defaults to `None` (server-side default of 100 applies). Mutually exclusive with `target_lufs`. |
| `target_lufs` | -70.0 to 0.0 | Absolute loudness normalization target in LUFS. Server rejects requests that set both `target_lufs` and `volume` — leave `volume` unset when using `target_lufs`, or the model raises a `ValidationError` locally. |
| `seed` | integer | Deterministic synthesis for identical text |

---

## Running the Example

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp env.example .env

# 3. Run
uv run python example.py
```

The bot will create a call (e.g. Daily room) and speak responses using Typecast voices.

---

## Compatibility

| Requirement | Version |
|-------------|---------|
| Pipecat | v0.0.94+ |
| Python | 3.10+ |
| Transports | Daily / Twilio / WebRTC |

---

## Support

- **API Documentation**: [typecast.ai](https://typecast.ai)
- **Pipecat Discord**: [discord.gg/pipecat](https://discord.gg/pipecat) (`#community-integrations`)

---

<div align="center">

**Maintainer**: Neosapience / Typecast team ([@neosapience](https://github.com/neosapience))

BSD-2-Clause License

</div>
