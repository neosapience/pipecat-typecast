"""Unit tests for language conversion functions."""

import pytest
from pipecat.transcriptions.language import Language

from pipecat_typecast.tts import ISO2_TO_ISO3_LANGUAGE_MAP, language_to_typecast_language


class TestLanguageToTypecastLanguage:
    """Tests for the language_to_typecast_language function."""

    @pytest.mark.unit
    def test_common_languages(self):
        """Test conversion of common languages."""
        test_cases = [
            (Language.EN, "eng"),
            (Language.KO, "kor"),
            (Language.JA, "jpn"),
            (Language.ZH, "zho"),
            (Language.ES, "spa"),
            (Language.DE, "deu"),
            (Language.FR, "fra"),
        ]

        for language, expected in test_cases:
            result = language_to_typecast_language(language)
            assert result == expected, f"Expected {expected} for {language}, got {result}"

    @pytest.mark.unit
    def test_none_language(self):
        """Test that None input returns None."""
        result = language_to_typecast_language(None)
        assert result is None

    @pytest.mark.unit
    def test_language_with_region_code(self):
        """Test language codes with region suffixes (e.g., en-US, ko-KR)."""
        # Language enum values like EN_US should be handled
        result = language_to_typecast_language(Language.EN_US)
        assert result == "eng"

        result = language_to_typecast_language(Language.ZH_CN)
        assert result == "zho"

    @pytest.mark.unit
    def test_all_mapped_languages(self):
        """Test that all languages in the mapping are valid."""
        for iso2, iso3 in ISO2_TO_ISO3_LANGUAGE_MAP.items():
            assert len(iso2) == 2, f"ISO-2 code should be 2 chars: {iso2}"
            assert len(iso3) == 3, f"ISO-3 code should be 3 chars: {iso3}"
            assert iso3.islower(), f"ISO-3 code should be lowercase: {iso3}"

    @pytest.mark.unit
    def test_additional_languages(self):
        """Test conversion of additional supported languages."""
        # ssfm-v30 additional languages
        test_cases = [
            (Language.BN, "ben"),  # Bengali
            (Language.HU, "hun"),  # Hungarian
            (Language.NO, "nor"),  # Norwegian
        ]

        for language, expected in test_cases:
            result = language_to_typecast_language(language)
            assert result == expected, f"Expected {expected} for {language}, got {result}"


class TestISO2ToISO3LanguageMap:
    """Tests for the language mapping dictionary."""

    @pytest.mark.unit
    def test_map_contains_common_languages(self):
        """Verify the map contains all common languages."""
        common = ["en", "ko", "ja", "zh", "es", "de", "fr", "it", "ru", "ar", "pt"]
        for code in common:
            assert code in ISO2_TO_ISO3_LANGUAGE_MAP, f"Missing common language: {code}"

    @pytest.mark.unit
    def test_map_values_are_iso3(self):
        """Verify all values are valid ISO-639-3 codes."""
        for iso3 in ISO2_TO_ISO3_LANGUAGE_MAP.values():
            assert isinstance(iso3, str)
            assert len(iso3) == 3
