"""Language detection tool for Qdrant MCP Server."""
import os
from typing import Any, Dict, List, Optional
import logging

from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)

try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    logger.warning("langdetect not installed. Language detection will be unavailable.")
    LANGDETECT_AVAILABLE = False


async def detect_language(
    text: str,
    return_confidence: bool = False,
    return_alternatives: bool = False,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Detect the language of the given text.

    Args:
        text: The text to detect language from
        return_confidence: Whether to return confidence score
        return_alternatives: Whether to return alternative languages with probabilities
        ctx: Optional MCP context

    Returns:
        Dict containing the detected language, and optionally confidence score and alternatives
    """
    if not LANGDETECT_AVAILABLE:
        if ctx:
            ctx.warning("langdetect library is not installed. Please install it with: pip install langdetect")
        return {
            "language": "unknown",
            "error": "langdetect library not installed"
        }

    try:
        if return_alternatives:
            # Get all possible languages with probabilities
            langs = detect_langs(text)
            result = {
                "language": langs[0].lang if langs else "unknown",
                "alternatives": [{"lang": lang.lang, "probability": lang.prob} for lang in langs]
            }
            if return_confidence and langs:
                result["confidence"] = langs[0].prob
            return result
        else:
            # Just get the main language
            language = detect(text)
            result = {"language": language}
            
            if return_confidence:
                # If confidence requested, we need to run detect_langs to get probabilities
                langs = detect_langs(text)
                matching_lang = next((lang for lang in langs if lang.lang == language), None)
                if matching_lang:
                    result["confidence"] = matching_lang.prob
            
            return result
    except LangDetectException as e:
        logger.error(f"Error detecting language: {e}")
        if ctx:
            ctx.error(f"Error detecting language: {e}")
        return {
            "language": "unknown",
            "error": str(e)
        }
