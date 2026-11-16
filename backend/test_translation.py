import os
import unittest
from unittest.mock import Mock, patch

import requests

from translation import (
    DEFAULT_GROQ_MODEL,
    GROQ_CHAT_COMPLETIONS_URL,
    TranslationConfigurationError,
    TranslationError,
    translate_to_english,
)


class TranslateToEnglishTests(unittest.TestCase):
    @patch("translation.requests.post")
    def test_translates_urdu_using_groq(self, post):
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "Drink while sitting."}}]
        }
        post.return_value = response

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            result = translate_to_english("پانی بیٹھ کر پینا چاہیے۔")

        self.assertEqual(result, "Drink while sitting.")
        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args[0], GROQ_CHAT_COMPLETIONS_URL)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["json"]["model"], DEFAULT_GROQ_MODEL)
        self.assertEqual(kwargs["json"]["messages"][1]["content"], "پانی بیٹھ کر پینا چاہیے۔")
        self.assertEqual(kwargs["timeout"], 30)
        response.raise_for_status.assert_called_once_with()

    @patch("translation.requests.post")
    def test_returns_english_without_calling_groq(self, post):
        result = translate_to_english("The ruling is not available.")

        self.assertEqual(result, "The ruling is not available.")
        post.assert_not_called()

    @patch("translation.requests.post")
    def test_requires_api_key_for_urdu(self, post):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with self.assertRaises(TranslationConfigurationError):
                translate_to_english("یہ اردو متن ہے۔")

        post.assert_not_called()

    @patch("translation.requests.post")
    def test_reports_provider_errors(self, post):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("provider error")
        post.return_value = response

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with self.assertRaises(TranslationError):
                translate_to_english("یہ اردو متن ہے۔")

    @patch("translation.requests.post")
    def test_reports_malformed_provider_responses(self, post):
        response = Mock()
        response.json.return_value = {"choices": []}
        post.return_value = response

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with self.assertRaises(TranslationError):
                translate_to_english("یہ اردو متن ہے۔")

    @patch("translation.requests.post")
    def test_reports_empty_translation(self, post):
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "  "}}]
        }
        post.return_value = response

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with self.assertRaises(TranslationError):
                translate_to_english("یہ اردو متن ہے۔")


if __name__ == "__main__":
    unittest.main()