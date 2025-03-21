"""Multilingual search tool for Qdrant MCP Server."""
import os
from typing import Any, Dict, List, Optional, Union
import logging

from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)

# Import local modules for language detection and translation
from ..search.nlq import nlq_search
from .detect_language import detect_language
from .translate_text import translate_text, DEEP_TRANSLATOR_AVAILABLE


async def multilingual_search(
    query: str,
    collection: Optional[str] = None,
    target_languages: Optional[List[str]] = None,
    translate_results: bool = False,
    user_language: Optional[str] = None,
    limit: int = 10,
    filter: Optional[Dict[str, Any]] = None,
    with_payload: Union[bool, List[str]] = True,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Perform multilingual search across documents in different languages.

    Args:
        query: The search query in any language
        collection: The collection to search
        target_languages: List of language codes to search in (if None, searches all languages)
        translate_results: Whether to translate search results to the query language
        user_language: Override detected language and use this as the user's language
        limit: Maximum number of results to return
        filter: Additional filters to apply to the search
        with_payload: Whether to include payload in the results
        ctx: Optional MCP context

    Returns:
        Dictionary containing search results and language information
    """
    # Step 1: Detect the language of the query
    if ctx:
        ctx.info(f"Processing multilingual search for query: '{query}'")
    
    query_language = user_language
    
    if not query_language:
        lang_detection = await detect_language(query, return_confidence=True)
        query_language = lang_detection.get("language", "en")
        
        if ctx and "confidence" in lang_detection:
            ctx.info(f"Detected query language: {query_language} (confidence: {lang_detection.get('confidence', 0):.2f})")
        elif ctx:
            ctx.info(f"Detected query language: {query_language}")
    
    # Step 2: Create translated versions of the query if needed
    translated_queries = {}
    
    if target_languages:
        if not DEEP_TRANSLATOR_AVAILABLE:
            if ctx:
                ctx.warning("Cannot perform multilingual search: deep_translator library not installed")
        else:
            for lang in target_languages:
                if lang != query_language:
                    try:
                        if ctx:
                            ctx.info(f"Translating query to {lang}...")
                        
                        translation = await translate_text(
                            text=query,
                            source_language=query_language,
                            target_language=lang
                        )
                        
                        if "error" not in translation:
                            translated_queries[lang] = translation["translated_text"]
                            if ctx:
                                ctx.info(f"Translated query ({lang}): '{translated_queries[lang]}'")
                    except Exception as e:
                        if ctx:
                            ctx.warning(f"Failed to translate query to {lang}: {e}")
    
    # Step 3: Run searches for original query and all translations
    all_results = []
    
    # Run search with original query
    if ctx:
        ctx.info(f"Searching with original query ({query_language}): '{query}'")
    
    original_results = await nlq_search(
        query=query,
        collection=collection,
        limit=limit,
        filter=filter,
        with_payload=with_payload,
        ctx=ctx
    )
    
    # Add language information to original results
    for result in original_results:
        if isinstance(result, dict):
            result["query_language"] = query_language
            result["result_language"] = result.get("payload", {}).get("language", "unknown")
            all_results.append(result)
    
    # Run searches with translated queries
    for lang, translated_query in translated_queries.items():
        if ctx:
            ctx.info(f"Searching with translated query ({lang}): '{translated_query}'")
        
        lang_results = await nlq_search(
            query=translated_query,
            collection=collection,
            limit=limit,
            filter=filter,
            with_payload=with_payload,
            ctx=ctx
        )
        
        # Add language information to translated results
        for result in lang_results:
            if isinstance(result, dict):
                result["query_language"] = lang
                result["result_language"] = result.get("payload", {}).get("language", "unknown")
                all_results.append(result)
    
    # Step 4: Combine, deduplicate, and sort results
    seen_ids = set()
    unique_results = []
    
    for result in all_results:
        if isinstance(result, dict) and "id" in result:
            result_id = result["id"]
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
    
    # Sort by score
    sorted_results = sorted(unique_results, key=lambda x: x.get("score", 0), reverse=True)
    
    # Limit the number of results
    final_results = sorted_results[:limit]
    
    # Step 5: Translate results if requested
    if translate_results and DEEP_TRANSLATOR_AVAILABLE:
        for result in final_results:
            if isinstance(result, dict) and "payload" in result:
                payload = result.get("payload", {})
                result_language = result.get("result_language", "unknown")
                
                # Only translate if languages differ
                if result_language != "unknown" and result_language != query_language:
                    # Identify text fields to translate
                    for key, value in payload.items():
                        if isinstance(value, str) and len(value) > 0:
                            try:
                                if ctx:
                                    ctx.info(f"Translating result field '{key}' from {result_language} to {query_language}")
                                
                                translation = await translate_text(
                                    text=value,
                                    source_language=result_language,
                                    target_language=query_language
                                )
                                
                                if "error" not in translation:
                                    # Store original text
                                    payload[f"original_{key}"] = value
                                    # Replace with translated text
                                    payload[key] = translation["translated_text"]
                                    
                            except Exception as e:
                                if ctx:
                                    ctx.warning(f"Failed to translate result field '{key}': {e}")
    
    return {
        "results": final_results,
        "query_language": query_language,
        "translated_queries": translated_queries,
        "total_searched_languages": 1 + len(translated_queries),
        "results_translated": translate_results and DEEP_TRANSLATOR_AVAILABLE
    }
