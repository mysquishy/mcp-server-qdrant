"""Translation tool for Qdrant MCP Server."""
import os
from typing import Any, Dict, List, Optional
import logging

from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)

try:
    from deep_translator import (
        GoogleTranslator,
        DeepL,
        MicrosoftTranslator,
        PonsTranslator,
        LingueeTranslator,
        MyMemoryTranslator,
    )
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    logger.warning("deep_translator not installed. Translation will be unavailable.")
    DEEP_TRANSLATOR_AVAILABLE = False

# Define available translation providers
PROVIDERS = {
    "google": GoogleTranslator,
    "deepl": DeepL,
    "microsoft": MicrosoftTranslator,
    "pons": PonsTranslator,
    "linguee": LingueeTranslator,
    "mymemory": MyMemoryTranslator,
}

# Providers requiring API keys and their environment variable names
API_KEY_VARS = {
    "deepl": "DEEPL_API_KEY",
    "microsoft": "MS_TRANSLATOR_KEY",
}


async def translate_text(
    text: str,
    target_language: str,
    source_language: Optional[str] = None,
    provider: str = "google",
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Translate text to the target language.

    Args:
        text: Text to translate
        target_language: Target language code (e.g., 'en', 'fr', 'es')
        source_language: Source language code (optional, auto-detected if not provided)
        provider: Translation provider (default: 'google')
        ctx: Optional MCP context

    Returns:
        Dict containing the translated text and metadata
    """
    if not DEEP_TRANSLATOR_AVAILABLE:
        if ctx:
            ctx.warning("deep_translator library is not installed. Please install it with: pip install deep-translator")
        return {
            "translated_text": text,
            "error": "deep_translator library not installed"
        }

    # Validate provider
    provider = provider.lower()
    if provider not in PROVIDERS:
        available_providers = ", ".join(PROVIDERS.keys())
        error_msg = f"Provider '{provider}' not supported. Available providers: {available_providers}"
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg, "translated_text": text}

    # Check if API key is required and available
    if provider in API_KEY_VARS:
        api_key_var = API_KEY_VARS[provider]
        api_key = os.environ.get(api_key_var)
        if not api_key:
            error_msg = f"API key for {provider} not found. Please set the {api_key_var} environment variable."
            if ctx:
                ctx.error(error_msg)
            return {"error": error_msg, "translated_text": text}

    try:
        # Initialize the translator
        translator_cls = PROVIDERS[provider]
        translator_kwargs = {"target": target_language}
        
        # Add source language if provided
        if source_language:
            translator_kwargs["source"] = source_language
            
        # Add API key if needed
        if provider in API_KEY_VARS:
            translator_kwargs["api_key"] = os.environ.get(API_KEY_VARS[provider])
            
        translator = translator_cls(**translator_kwargs)
        
        # Perform translation
        translated_text = translator.translate(text)
        
        return {
            "translated_text": translated_text,
            "source_language": source_language or "auto",
            "target_language": target_language,
            "provider": provider
        }
    except Exception as e:
        error_msg = f"Translation error: {str(e)}"
        logger.error(error_msg)
        if ctx:
            ctx.error(error_msg)
        return {
            "error": error_msg,
            "translated_text": text,
            "source_language": source_language or "auto",
            "target_language": target_language,
            "provider": provider
        }
