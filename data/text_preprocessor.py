"""Text preprocessing for academic graph nodes.

Combines title, keywords, and abstract into unified text representations
suitable for SBERT encoding.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class TextPreprocessor:
    """Preprocesses academic text fields for embedding generation.

    Combines title, keywords, and abstract into a single text string,
    with cleaning and normalization applied.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the text preprocessor.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}

    def preprocess(self, attributes: Dict[str, Any]) -> str:
        """Preprocess node attributes into a unified text representation.

        Combines title, keywords, and abstract in that order, separated
        by periods.

        Args:
            attributes: Dictionary of node attributes. Expected keys:
                        'title', 'keywords', 'abstract'.

        Returns:
            Cleaned and concatenated text string.
        """
        parts = []

        title = attributes.get("title", "")
        if title:
            parts.append(self._clean(str(title)))

        keywords = attributes.get("keywords", [])
        if isinstance(keywords, list) and keywords:
            keyword_str = ", ".join(str(k) for k in keywords if k)
            if keyword_str:
                parts.append(self._clean(keyword_str))
        elif isinstance(keywords, str) and keywords.strip():
            parts.append(self._clean(keywords))

        abstract = attributes.get("abstract", "")
        if abstract:
            parts.append(self._clean(str(abstract)))

        return ". ".join(parts)

    def _clean(self, text: str) -> str:
        """Clean text by normalizing whitespace and removing special characters.

        Args:
            text: Raw text string.

        Returns:
            Cleaned text string.
        """
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
