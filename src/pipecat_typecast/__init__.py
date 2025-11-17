"""Typecast TTS integration for Pipecat."""

from importlib.metadata import version as _get_version

from pipecat_typecast.tts import (
	TypecastTTSService,
	TypecastInputParams,
	PromptOptions,
	OutputOptions,
)

try:
	version = _get_version("pipecat-ai-typecast")
except Exception:
	# Fallback for development or when package is not installed
	version = "0.0.1"

__all__ = [
	"TypecastTTSService",
	"TypecastInputParams",
	"PromptOptions",
	"OutputOptions",
]

