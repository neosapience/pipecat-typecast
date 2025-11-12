# Pipecat Typecast TTS Integration

Add high-quality neural voices from [Typecast](https://typecast.ai/) to your Pipecat AI pipelines.

**Maintainer:** Neosapience / Typecast team (@neosapience)

## Installation

```bash
pip install pipecat-ai-typecast
```

## Prerequisites

- Typecast API key (`TYPECAST_API_KEY`)
- Optional: Voice override (`TYPECAST_VOICE_ID`) – defaults to `tc_62a8975e695ad26f7fb514d1`

## Usage with Pipecat Pipeline

`TypecastTTSService` integrates Typecast's streaming text-to-speech into a Pipecat pipeline. It converts LLM text output into expressive speech while leveraging Pipecat's transport, STT, and turn-taking stack.

```python
import os, aiohttp
from pipecat.pipeline.pipeline import Pipeline
from pipecat_typecast.tts import TypecastTTSService

async with aiohttp.ClientSession() as session:
    llm = ...
    sst = ...
    tts = TypecastTTSService(
        aiohttp_session=session,
        api_key=os.getenv("TYPECAST_API_KEY"),
        voice_id=os.getenv("TYPECAST_VOICE_ID", "tc_62a8975e695ad26f7fb514d1"),
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

### Advanced Configuration (Emotion & Audio Controls)

`TypecastTTSService` exposes structured parameter models so you can tune emotion and audio output.

```python
from pipecat_typecast.tts import (
    TypecastTTSService,
    TypecastInputParams,
    PromptOptions,
    OutputOptions,
)

params = TypecastInputParams(
    # Language influences pronunciation model (defaults to English)
    # Language.EN / Language.KO / Language.JA ...
    # If omitted, Typecast auto-detect may apply (depending on voice).
    prompt_options=PromptOptions(
        emotion_preset="happy",      # normal | happy | sad | angry | whisper (voice dependent)
        emotion_intensity=1.3,       # 0.0–2.0 (float)
    ),
    output_options=OutputOptions(
        volume=110,                  # 0–200 (percent)
        audio_pitch=2,               # -12..12 (semitones)
        audio_tempo=1.05,            # 0.5–2.0 (playback speed)
        audio_format="wav",          # Only 'wav' currently supported
    ),
)

tts = TypecastTTSService(
    aiohttp_session=session,
    api_key=os.getenv("TYPECAST_API_KEY"),
    voice_id="tc_62a8975e695ad26f7fb514d1",  # Replace with another voice ID as desired
    model="ssfm-v21",                        # Default model
    params=params,
)
```

Notes:
- `emotion_preset` availability varies by voice. If unsupported, the service falls back to neutral.
- `emotion_intensity` > 1.0 increases expressiveness; extreme values can sound synthetic.
- `audio_pitch` shifts pitch in musical semitone units (use small adjustments for naturalness).
- `audio_tempo` changes speaking speed; keep within 0.85–1.15 for intelligibility.
- `seed` (set in `TypecastInputParams`) provides deterministic synthesis for identical text (when supported by model).
- Unsupported `audio_format` values yield an error frame—keep `wav`.

## Running the Example

1. Install dependencies:
    ```bash
    uv sync
    ```

2. Set up your environment

   ```bash
   cp env.example .env
   ```

3. Run:
    ```bash
    uv run python example.py
    ```

The bot will create a call (e.g. Daily room) and speak responses using Typecast voices.

## Compatibility

**Tested with Pipecat v0.0.89**

- Python 3.10+
- Daily / Twilio / generic WebRTC transports (see `example.py`)

## License

BSD-2-Clause - see [LICENSE](LICENSE)

## Support

- Docs: https://typecast.ai (refer to API docs for voice IDs & parameters)
- Pipecat Discord: https://discord.gg/pipecat (`#community-integrations`)
