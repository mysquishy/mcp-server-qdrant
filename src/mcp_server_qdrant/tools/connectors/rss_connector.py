"""RSS feed connector for Qdrant MCP Server.

This tool allows monitoring RSS feeds and storing new items in Qdrant collections.
"""
import os
import asyncio
from typing import Any, Dict, List, Optional, Set, Union
import logging
import json
from datetime import datetime, timedelta
import time
import hashlib
import re
from urllib.parse import urlparse

from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)

try:
    import feedparser
    import httpx
    from bs4 import BeautifulSoup
    FEED_DEPENDENCIES_AVAILABLE = True
except ImportError:
    logger.warning("RSS feed dependencies not installed. Install with: pip install feedparser httpx beautifulsoup4")
    FEED_DEPENDENCIES_AVAILABLE = False

# Import web tools for processing feed content
from ..web.content_processor import process_content
from ..data_processing.chunk_and_process import chunk_and_process


async def setup_rss_connector(
    feed_url: str,
    collection: str,
    update_interval: int = 3600,  # Default: 1 hour
    max_items_per_update: int = 10,
    extract_full_content: bool = True,
    remove_html_tags: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    start_immediately: bool = True,
    store_feed_metadata: bool = True,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Set up an RSS feed connector that periodically checks for new items and stores them in Qdrant.

    Args:
        feed_url: URL of the RSS feed to monitor
        collection: Qdrant collection to store feed items
        update_interval: Time between feed checks in seconds (default: 1 hour)
        max_items_per_update: Maximum number of items to process per update
        extract_full_content: Whether to extract full content from item links
        remove_html_tags: Whether to remove HTML tags from content
        chunk_size: Size of each chunk when storing in Qdrant
        chunk_overlap: Overlap between chunks
        start_immediately: Whether to perform initial check immediately
        store_feed_metadata: Whether to store feed metadata with each item
        ctx: Optional MCP context

    Returns:
        Dictionary containing setup status and feed information
    """
    if not FEED_DEPENDENCIES_AVAILABLE:
        error_msg = "RSS feed dependencies not installed. Install with: pip install feedparser httpx beautifulsoup4"
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg}
    
    if ctx:
        ctx.info(f"Setting up RSS connector for feed: {feed_url}")
    
    # Validate feed URL
    try:
        parsed_url = urlparse(feed_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            error_msg = f"Invalid feed URL format: {feed_url}"
            if ctx:
                ctx.error(error_msg)
            return {"error": error_msg}
    except Exception as e:
        error_msg = f"Error parsing feed URL: {str(e)}"
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg}
    
    # Set up configuration
    config = {
        "feed_url": feed_url,
        "collection": collection,
        "update_interval": update_interval,
        "max_items_per_update": max_items_per_update,
        "extract_full_content": extract_full_content,
        "remove_html_tags": remove_html_tags,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "last_updated": None,
        "last_etag": None,
        "last_modified": None,
        "items_processed": 0,
        "is_active": True,
        "created_at": datetime.now().isoformat()
    }
    
    # Fetch initial feed data
    feed_info = {}
    
    if start_immediately:
        try:
            if ctx:
                ctx.info(f"Performing initial fetch of feed: {feed_url}")
            
            # Get feed data
            feed_data = feedparser.parse(feed_url)
            
            if feed_data.get('bozo', 0) == 1:
                # Feed parsing had an error
                error_msg = f"Warning: Feed has errors: {feed_data.get('bozo_exception')}"
                if ctx:
                    ctx.warning(error_msg)
            
            # Extract feed metadata
            feed_info = {
                "title": feed_data.feed.get('title', 'Unknown Feed'),
                "description": feed_data.feed.get('description', ''),
                "link": feed_data.feed.get('link', feed_url),
                "language": feed_data.feed.get('language', 'unknown'),
                "updated": feed_data.feed.get('updated', ''),
                "entries_count": len(feed_data.entries)
            }
            
            # Process initial items
            if feed_data.entries:
                processed_count = await process_feed_items(
                    feed_data=feed_data,
                    collection=collection,
                    max_items=max_items_per_update,
                    extract_full_content=extract_full_content,
                    remove_html_tags=remove_html_tags,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    store_feed_metadata=store_feed_metadata,
                    ctx=ctx
                )
                
                config["items_processed"] = processed_count
                config["last_updated"] = datetime.now().isoformat()
                config["last_etag"] = feed_data.get('etag')
                config["last_modified"] = feed_data.get('modified')
                
                if ctx:
                    ctx.info(f"Processed {processed_count} items from feed")
        
        except Exception as e:
            error_msg = f"Error during initial feed fetch: {str(e)}"
            logger.error(error_msg)
            if ctx:
                ctx.error(error_msg)
            feed_info = {"error": error_msg}
    
    # Return configuration and feed info
    return {
        "status": "success",
        "message": "RSS connector set up successfully" + (" with initial fetch" if start_immediately else ""),
        "connector_id": hashlib.md5(feed_url.encode()).hexdigest(),
        "config": config,
        "feed_info": feed_info
    }


async def process_feed_items(
    feed_data: Any,
    collection: str,
    max_items: int = 10,
    extract_full_content: bool = True,
    remove_html_tags: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    store_feed_metadata: bool = True,
    ctx: Optional[Context] = None
) -> int:
    """
    Process items from an RSS feed and store them in Qdrant.

    Args:
        feed_data: Feed data from feedparser
        collection: Qdrant collection to store items
        max_items: Maximum number of items to process
        extract_full_content: Whether to extract full content from item links
        remove_html_tags: Whether to remove HTML tags from content
        chunk_size: Size of each chunk when storing in Qdrant
        chunk_overlap: Overlap between chunks
        store_feed_metadata: Whether to store feed metadata with each item
        ctx: Optional MCP context

    Returns:
        Number of items processed
    """
    if not feed_data.entries:
        if ctx:
            ctx.info("No feed entries found")
        return 0
    
    # Limit to max_items
    entries = feed_data.entries[:max_items]
    
    if ctx:
        ctx.info(f"Processing {len(entries)} feed entries")
    
    processed_count = 0
    
    for i, entry in enumerate(entries):
        try:
            if ctx:
                ctx.info(f"Processing feed entry {i+1}/{len(entries)}: {entry.get('title', 'Untitled')}")
                ctx.report_progress(i, len(entries))
            
            # Extract entry metadata
            entry_metadata = {
                "feed_title": feed_data.feed.get('title', 'Unknown Feed'),
                "feed_link": feed_data.feed.get('link', ''),
                "title": entry.get('title', 'Untitled'),
                "link": entry.get('link', ''),
                "published": entry.get('published', ''),
                "author": entry.get('author', ''),
                "id": entry.get('id', hashlib.md5(entry.get('link', '').encode()).hexdigest()),
                "tags": [tag.get('term', '') for tag in entry.get('tags', [])],
                "source": "rss_connector",
                "processed_at": datetime.now().isoformat()
            }
            
            # Get entry content
            entry_content = ""
            
            # Try to get content from summary or content fields
            if 'summary' in entry:
                entry_content = entry.summary
            elif 'content' in entry:
                entry_content = entry.content[0].value
            elif 'description' in entry:
                entry_content = entry.description
            
            # Extract full content if requested and link is available
            if extract_full_content and entry.get('link'):
                try:
                    if ctx:
                        ctx.info(f"Extracting full content from: {entry.get('link')}")
                    
                    result = await process_content(
                        source=entry.get('link'),
                        collection=None,  # Don't store directly, we'll combine with metadata
                        extract_metadata=True,
                        store_in_qdrant=False,
                        remove_html_tags=remove_html_tags,
                        ctx=ctx
                    )
                    
                    if 'error' not in result:
                        full_content = result.get('content_preview', '')
                        if full_content and len(full_content) > len(entry_content):
                            # Use the full content if it's longer
                            entry_content = full_content
                            
                            # Merge metadata
                            if result.get('metadata'):
                                for key, value in result.get('metadata', {}).items():
                                    if key not in entry_metadata:
                                        entry_metadata[f"page_{key}"] = value
                
                except Exception as e:
                    if ctx:
                        ctx.warning(f"Error extracting full content: {str(e)}")
            
            # Remove HTML if requested
            if remove_html_tags and entry_content:
                soup = BeautifulSoup(entry_content, 'lxml')
                entry_content = soup.get_text(separator='\n', strip=True)
            
            # Store in Qdrant
            if entry_content:
                try:
                    result = await chunk_and_process(
                        text=entry_content,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        collection=collection,
                        metadata=entry_metadata,
                        ctx=ctx
                    )
                    
                    if isinstance(result, dict) and "chunks" in result:
                        chunks_stored = len(result["chunks"])
                        if ctx:
                            ctx.info(f"Stored {chunks_stored} chunks for entry")
                        processed_count += 1
                
                except Exception as e:
                    if ctx:
                        ctx.error(f"Error storing entry in Qdrant: {str(e)}")
        
        except Exception as e:
            if ctx:
                ctx.error(f"Error processing feed entry: {str(e)}")
    
    return processed_count


async def check_rss_updates(
    feed_url: str,
    collection: str,
    last_etag: Optional[str] = None,
    last_modified: Optional[str] = None,
    max_items: int = 10,
    extract_full_content: bool = True,
    remove_html_tags: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    store_feed_metadata: bool = True,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Check an RSS feed for updates and process new items.

    Args:
        feed_url: URL of the RSS feed to check
        collection: Qdrant collection to store new items
        last_etag: Last ETag value for conditional requests
        last_modified: Last modified time for conditional requests
        max_items: Maximum number of items to process
        extract_full_content: Whether to extract full content from item links
        remove_html_tags: Whether to remove HTML tags from content
        chunk_size: Size of each chunk when storing in Qdrant
        chunk_overlap: Overlap between chunks
        store_feed_metadata: Whether to store feed metadata with each item
        ctx: Optional MCP context

    Returns:
        Dictionary containing check status and update information
    """
    if not FEED_DEPENDENCIES_AVAILABLE:
        error_msg = "RSS feed dependencies not installed. Install with: pip install feedparser httpx beautifulsoup4"
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg}
    
    if ctx:
        ctx.info(f"Checking RSS feed for updates: {feed_url}")
    
    try:
        # Set up conditional request parameters
        request_params = {}
        if last_etag:
            request_params['etag'] = last_etag
        if last_modified:
            request_params['modified'] = last_modified
        
        # Parse feed with conditional parameters
        feed_data = feedparser.parse(feed_url, **request_params)
        
        # Check status
        status = feed_data.get('status', 200)
        
        if status == 304:  # Not modified
            if ctx:
                ctx.info("Feed has not been modified since last check")
            return {
                "status": "not_modified",
                "message": "Feed has not been modified since last check",
                "items_processed": 0,
                "new_etag": last_etag,
                "new_modified": last_modified
            }
        
        # Process feed items
        processed_count = await process_feed_items(
            feed_data=feed_data,
            collection=collection,
            max_items=max_items,
            extract_full_content=extract_full_content,
            remove_html_tags=remove_html_tags,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            store_feed_metadata=store_feed_metadata,
            ctx=ctx
        )
        
        # Get new ETag and Modified values
        new_etag = feed_data.get('etag', last_etag)
        new_modified = feed_data.get('modified', last_modified)
        
        return {
            "status": "success",
            "message": f"Successfully checked feed and processed {processed_count} items",
            "items_processed": processed_count,
            "feed_title": feed_data.feed.get('title', 'Unknown Feed'),
            "feed_link": feed_data.feed.get('link', feed_url),
            "entries_count": len(feed_data.entries),
            "new_etag": new_etag,
            "new_modified": new_modified
        }
    
    except Exception as e:
        error_msg = f"Error checking RSS feed: {str(e)}"
        logger.error(error_msg)
        if ctx:
            ctx.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "items_processed": 0
        }


async def list_rss_feeds(
    collection: str,
    limit: int = 100,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    List all RSS feeds configured for a collection.

    Args:
        collection: Qdrant collection to query
        limit: Maximum number of feeds to return
        ctx: Optional MCP context

    Returns:
        Dictionary containing list of configured feeds
    """
    # In a real implementation, this would query a database or storage for connector configs
    # For this demo, we'll return a placeholder message
    if ctx:
        ctx.info(f"Listing RSS feeds for collection: {collection}")
        ctx.warning("This is a placeholder implementation. In a real deployment, feed configurations would be stored in a database.")
    
    return {
        "status": "success",
        "message": "This is a placeholder implementation. In a real deployment, feed configurations would be stored in a database.",
        "feeds": []
    }
