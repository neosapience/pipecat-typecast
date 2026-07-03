# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-03

### Added

- `TypecastTTSService` now uses `typecast-python>=0.3.8` with externally injected `aiohttp.ClientSession` support.
- `TypecastInputParams.streaming` defaults to `True` and routes synthesis through Typecast's HTTP streaming endpoint.

### Changed

- Calls that set `OutputOptions.volume` use the non-streaming SDK endpoint so existing volume-based configurations keep working.

## [0.2.1] - 2026-07-01

### Fixed

- Explicit `api_key` and `voice_id` constructor arguments now override environment variables as documented.
- `TypecastTTSService` initialization now works with newer Pipecat versions that no longer expose `set_model_name`.

### Changed

- Added an explicit `websockets` dependency for compatibility with Pipecat versions that import it from `pipecat.services.tts_service`.

## [0.2.0] - 2026-05-11

### Added

- **`OutputOptions.target_lufs`** — absolute loudness normalization target in LUFS (`-70.0` to `0.0`). Maps to the Typecast TTS `output.target_lufs` field. Useful when piping speech into downstream audio pipelines that expect a consistent loudness regardless of voice.

### Changed

- **`OutputOptions.volume` default is now `None`** (was `100`). The server-side default of 100 still applies when neither `volume` nor `target_lufs` is set — `model_dump(exclude_none=True)` simply drops the field from the payload. This change is needed to make the `volume` vs `target_lufs` mutual exclusion explicit: setting `target_lufs` alone is now the supported path, and the dumped payload no longer carries a stray `volume=100` that would make the server reject the request.
- **Mutual exclusion is enforced via a model validator.** `OutputOptions(volume=..., target_lufs=...)` raises `ValidationError("Volume and target_lufs are mutually exclusive...")` so the conflict surfaces at construction time rather than during the HTTP call.

### Backward compatibility

- Callers that built `OutputOptions()` with no arguments are unaffected: previously the request carried `volume=100`, now it carries no `volume` at all and the server applies its own default of 100. End-to-end behavior is identical.
- Callers that explicitly passed `volume=N` continue to work unchanged. Only the combination `volume=N, target_lufs=M` (which the server would have rejected anyway) now raises locally.

## [0.0.1] - 2025-11-12

### Added
- Initial release of Typecast TTS integration for Pipecat
- `TypecastTTSService` for high-quality TTS service

### Documentation
- README with installation and usage instructions
- Foundational example demonstrating complete pipecat bot setup
