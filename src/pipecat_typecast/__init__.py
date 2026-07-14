"""Typecast TTS integration for Pipecat."""

from importlib.metadata import version as _get_version

from pipecat_typecast.tts import (
	OutputOptions,
	PresetPromptOptions,
	PromptOptions,
	SmartPromptOptions,
	TypecastInputParams,
	TypecastPromptOptions,
	TypecastTTSService,
)

try:
	__version__ = _get_version("pipecat-ai-typecast")
except Exception:
	# Fallback for development or when package is not installed
	__version__ = "0.0.1"

__all__ = [
	"TypecastTTSService",
	"TypecastInputParams",
	"PromptOptions",
	"PresetPromptOptions",
	"SmartPromptOptions",
	"TypecastPromptOptions",
	"OutputOptions",
	"__version__",
]
