"""Metadata extraction tool for Qdrant MCP server."""
import logging
import re
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio

from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)

# Regular expressions for common metadata patterns
PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "url": r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*',
    "phone_number": r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    "date": r'\b(?:\d{1,2}[-/\s]\d{1,2}[-/\s]\d{2,4})|(?:\d{4}[-/\s]\d{1,2}[-/\s]\d{1,2})\b',
    "time": r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[aApP][mM])?\b',
    "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    "hashtag": r'#\w+',
    "mention": r'@\w+',
    "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    "zip_code": r'\b\d{5}(?:[-\s]\d{4})?\b',
    "currency": r'\$\d+(?:\.\d{2})?',
    "percentage": r'\d+(?:\.\d+)?%',
    "isbn": r'(?:ISBN(?:-1[03])?:?\s)?(?=[0-9X]{10}$|(?=(?:[0-9]+[-\s]){3})[-\s0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[-\s]){4})[-\s0-9]{17}$)(?:97[89][-\s]?)?[0-9]{1,5}[-\s]?[0-9]+[-\s]?[0-9]+[-\s]?[0-9X]',
    "mac_address": r'([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})'
}

# Entity type mapping for common categories
ENTITY_CATEGORIES = {
    "PERSON": "people",
    "ORG": "organizations",
    "GPE": "locations",
    "LOC": "locations",
    "FACILITY": "locations",
    "PRODUCT": "products",
    "EVENT": "events",
    "WORK_OF_ART": "creations",
    "DATE": "dates",
    "TIME": "times",
    "MONEY": "financial",
    "PERCENT": "financial"
}

async def extract_metadata(
    ctx: Context,
    text: str,
    extract_entities: bool = True,
    extract_patterns: bool = True,
    extract_statistics: bool = True,
    custom_patterns: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Extract structured metadata from document text.
    
    This tool analyzes document content and extracts important metadata 
    such as entities (people, organizations, locations), key patterns 
    (emails, dates, URLs), and document statistics.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    text : str
        The text to analyze and extract metadata from
    extract_entities : bool, optional
        Whether to extract named entities (default: True)
    extract_patterns : bool, optional
        Whether to extract common patterns like emails, dates, URLs (default: True)
    extract_statistics : bool, optional
        Whether to extract document statistics (default: True)
    custom_patterns : Dict[str, str], optional
        Additional custom regex patterns to extract
        
    Returns:
    --------
    Dict[str, Any]
        Extracted metadata including entities, patterns, and statistics
    """
    if not text:
        return {"status": "error", "message": "No text provided", "metadata": {}}
    
    await ctx.debug(f"Extracting metadata from text ({len(text)} characters)")
    
    metadata = {}
    tasks = []
    
    # Document statistics - these are quick to calculate
    if extract_statistics:
        await ctx.debug("Extracting document statistics")
        stats = await extract_document_statistics(text)
        metadata["statistics"] = stats
    
    # Extract patterns in parallel with entity extraction
    if extract_patterns:
        await ctx.debug("Extracting patterns")
        patterns_task = asyncio.create_task(extract_text_patterns(text, custom_patterns))
        tasks.append(patterns_task)
    
    # Entity extraction - this is more computationally intensive
    if extract_entities:
        try:
            # We'll import spacy here to avoid overhead if not needed
            import spacy
            await ctx.debug("Extracting named entities")
            entities_task = asyncio.create_task(extract_named_entities(ctx, text))
            tasks.append(entities_task)
        except ImportError:
            await ctx.warning("Spacy not installed. Skipping entity extraction.")
            metadata["entities"] = {"status": "error", "message": "Spacy not installed"}
    
    # Wait for all extraction tasks to complete
    await ctx.report_progress(25, 100)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error in extraction task: {str(result)}", exc_info=True)
            await ctx.debug(f"Error in extraction task: {str(result)}")
        else:
            # Add results to metadata
            if extract_patterns and i == 0:  # First task was pattern extraction
                metadata["patterns"] = result
            elif extract_entities and ((i == 0 and not extract_patterns) or (i == 1 and extract_patterns)):
                metadata["entities"] = result
    
    await ctx.report_progress(100, 100)
    await ctx.info(f"Completed metadata extraction")
    
    return {
        "status": "success",
        "metadata": metadata,
        "timestamp": datetime.now().isoformat()
    }

async def extract_document_statistics(text: str) -> Dict[str, Any]:
    """Extract basic document statistics."""
    # Split by whitespace for word count
    words = text.split()
    
    # Split by periods, exclamation, question marks for sentence count
    # This is a simplified approach - a more sophisticated sentence tokenizer would be better
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Split by double newlines for paragraph count
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "average_word_length": sum(len(word) for word in words) / max(len(words), 1),
        "average_sentence_length": sum(len(sentence.split()) for sentence in sentences) / max(len(sentences), 1)
    }

async def extract_text_patterns(
    text: str, 
    custom_patterns: Optional[Dict[str, str]] = None
) -> Dict[str, List[str]]:
    """Extract common patterns like emails, dates, URLs using regex."""
    # Combine built-in patterns with custom patterns
    all_patterns = PATTERNS.copy()
    if custom_patterns:
        all_patterns.update(custom_patterns)
    
    # Extract patterns
    results = {}
    for pattern_name, regex in all_patterns.items():
        matches = re.findall(regex, text)
        if matches:
            # Remove duplicates while preserving order
            unique_matches = []
            seen = set()
            for match in matches:
                if match not in seen:
                    seen.add(match)
                    unique_matches.append(match)
            results[pattern_name] = unique_matches
    
    return results

async def extract_named_entities(ctx: Context, text: str) -> Dict[str, Any]:
    """Extract named entities using spaCy."""
    try:
        import spacy
        
        # Try to load the model, downloading if necessary
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            await ctx.info("Downloading spaCy model (this may take a moment)")
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        
        # Process the text in chunks to avoid memory issues with very large texts
        max_length = 1000000  # spaCy's default max length
        if len(text) > max_length:
            await ctx.warning(f"Text is very large ({len(text)} chars). Processing in chunks.")
            
            # Create chunks that respect sentence boundaries where possible
            chunks = []
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < max_length:
                    current_chunk += sentence + " "
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Process each chunk
            all_entities = {}
            for i, chunk in enumerate(chunks):
                await ctx.debug(f"Processing chunk {i+1}/{len(chunks)}")
                await ctx.report_progress(i, len(chunks))
                
                doc = nlp(chunk)
                
                # Extract entities from this chunk
                chunk_entities = process_entities(doc.ents)
                
                # Merge with overall entities
                for category, entities in chunk_entities.items():
                    if category in all_entities:
                        for entity, count in entities.items():
                            all_entities[category][entity] = all_entities[category].get(entity, 0) + count
                    else:
                        all_entities[category] = entities.copy()
            
            entities_by_category = all_entities
        else:
            # Process the whole text at once
            doc = nlp(text)
            entities_by_category = process_entities(doc.ents)
        
        # Sort entities by frequency
        for category, entities in entities_by_category.items():
            entities_by_category[category] = {
                k: v for k, v in sorted(
                    entities.items(),
                    key=lambda item: item[1],
                    reverse=True
                )
            }
        
        return {
            "by_category": entities_by_category,
            "top_entities": get_top_entities(entities_by_category, limit=10)
        }
        
    except Exception as e:
        error_msg = f"Error extracting entities: {str(e)}"
        logger.error(error_msg, exc_info=True)
        await ctx.debug(error_msg)
        return {"status": "error", "message": error_msg}

def process_entities(entities):
    """Process and categorize spaCy entities."""
    entities_by_category = {}
    
    for ent in entities:
        category = ENTITY_CATEGORIES.get(ent.label_, "other")
        
        if category not in entities_by_category:
            entities_by_category[category] = {}
        
        # Count entity occurrences
        entity_text = ent.text.strip()
        if entity_text:
            entities_by_category[category][entity_text] = entities_by_category[category].get(entity_text, 0) + 1
    
    return entities_by_category

def get_top_entities(entities_by_category, limit=10):
    """Get the top entities across all categories."""
    all_entities = []
    
    for category, entities in entities_by_category.items():
        for entity, count in entities.items():
            all_entities.append({
                "text": entity,
                "category": category,
                "count": count
            })
    
    # Sort by count and take top entities
    top_entities = sorted(all_entities, key=lambda x: x["count"], reverse=True)[:limit]
    
    return top_entities
