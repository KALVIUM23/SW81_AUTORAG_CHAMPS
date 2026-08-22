"""
Text Extraction Cleaning and Normalization Engine (Topic 3.20)
"""
import re


class TextCleaner:
    @staticmethod
    def clean_page_text(text: str) -> str:
        if not text:
            return ""

        # Normalize line breaks and tabs
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Strip standard workshop manual running headers/footers
        text = re.sub(r"(?i)Page \d+ of \d+", "", text)
        text = re.sub(r"(?i)Confidential - Authorized Service Bay Only", "", text)
        text = re.sub(r"(?i)Apex Motors Workshop Manual - Rev \d+\.\d+", "", text)

        # Normalize multiple spaces and blank lines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        return text.strip()