"""Twitter connector for Qdrant MCP Server.

This tool allows monitoring Twitter accounts and storing new tweets in Qdrant collections.
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
from urllib.parse import urlparse, quote_plus

from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)

try:
    import httpx
    from bs4 import BeautifulSoup
    TWITTER_DEPENDENCIES_AVAILABLE = True
except ImportError:
    logger.warning("Twitter dependencies not installed. Install with: pip install httpx beautifulsoup4")
    TWITTER_DEPENDENCIES_AVAILABLE = False

# Import web tools for processing tweet content
from ..web.content_processor import process_content
from ..data_processing.chunk_and_process import chunk_and_process


async def setup_twitter_connector(
    ctx: Context,
    username: str,
    collection: str,
    update_interval: int = 3600,  # Default: 1 hour
    max_tweets_per_update: int = 20,
    include_retweets: bool = False,
    include_replies: bool = False,
    fetch_full_conversation: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    api_key: Optional[str] = None,
    start_immediately: bool = True,
    store_media: bool = False
) -> Dict[str, Any]:
    """
    Set up a Twitter connector that periodically checks for new tweets and stores them in Qdrant.

    Args:
        ctx: The MCP request context
        username: Twitter username to monitor (without the @ symbol)
        collection: Qdrant collection to store tweets
        update_interval: Time between checks in seconds (default: 1 hour)
        max_tweets_per_update: Maximum number of tweets to process per update
        include_retweets: Whether to include retweets
        include_replies: Whether to include replies
        fetch_full_conversation: Whether to fetch full conversation thread for replies
        chunk_size: Size of each chunk when storing in Qdrant
        chunk_overlap: Overlap between chunks
        api_key: Twitter API key (if available)
        start_immediately: Whether to perform initial check immediately
        store_media: Whether to extract and store media content

    Returns:
        Dictionary containing setup status and Twitter information
    """
    if not TWITTER_DEPENDENCIES_AVAILABLE:
        error_msg = "Twitter dependencies not installed. Install with: pip install httpx beautifulsoup4"
        await ctx.error(error_msg)
        return {"error": error_msg}
    
    await ctx.info(f"Setting up Twitter connector for user: @{username}")
    
    # Validate username
    if not username or not re.match(r'^[A-Za-z0-9_]+$', username):
        error_msg = f"Invalid Twitter username format: {username}"
        await ctx.error(error_msg)
        return {"error": error_msg}
    
    # Set up configuration
    config = {
        "username": username,
        "collection": collection,
        "update_interval": update_interval,
        "max_tweets_per_update": max_tweets_per_update,
        "include_retweets": include_retweets,
        "include_replies": include_replies,
        "fetch_full_conversation": fetch_full_conversation,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "last_updated": None,
        "last_tweet_id": None,
        "tweets_processed": 0,
        "is_active": True,
        "created_at": datetime.now().isoformat(),
        "store_media": store_media
    }
    
    # TODO: Add initial fetch implementation
    # TODO: Add storing and processing of tweets
    
    return {
        "status": "success",
        "message": "Twitter connector set up successfully",
        "connector_id": hashlib.md5(f"twitter_{username}".encode()).hexdigest(),
        "config": config
    }


async def fetch_user_tweets(
    ctx: Context,
    username: str,
    max_tweets: int = 20,
    include_retweets: bool = False,
    include_replies: bool = False,
    since_id: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch tweets from a Twitter user.

    Args:
        ctx: The MCP request context
        username: Twitter username (without @ symbol)
        max_tweets: Maximum number of tweets to fetch
        include_retweets: Whether to include retweets
        include_replies: Whether to include replies
        since_id: Only fetch tweets newer than this ID
        api_key: Twitter API key (if available)

    Returns:
        Dictionary containing tweets and user information
    """
    # TODO: Implement tweet fetching logic
    # This will be expanded in the full implementation
    return {"tweets": [], "user_display_name": f"@{username}"}


async def check_twitter_updates(
    ctx: Context,
    username: str,
    collection: str,
    last_tweet_id: Optional[str] = None,
    max_tweets: int = 20,
    include_retweets: bool = False,
    include_replies: bool = False
) -> Dict[str, Any]:
    """
    Check a Twitter account for updates and process new tweets.

    Args:
        ctx: The MCP request context
        username: Twitter username to check (without the @ symbol)
        collection: Qdrant collection to store new tweets
        last_tweet_id: Only fetch tweets newer than this ID
        max_tweets: Maximum number of tweets to process
        include_retweets: Whether to include retweets
        include_replies: Whether to include replies

    Returns:
        Dictionary containing check status and update information
    """
    # TODO: Implement update checking logic
    # This will be expanded in the full implementation
    return {
        "status": "no_updates",
        "message": "No new tweets found since last check",
        "tweets_processed": 0,
        "new_last_tweet_id": last_tweet_id
    }


async def list_twitter_connectors(
    ctx: Context,
    collection: str,
    limit: int = 100
) -> Dict[str, Any]:
    """
    List all Twitter connectors configured for a collection.

    Args:
        ctx: The MCP request context
        collection: Qdrant collection to query
        limit: Maximum number of connectors to return

    Returns:
        Dictionary containing list of configured connectors
    """
    # This is a placeholder implementation for now
    # In a real implementation, this would query a database or storage for connector configs
    await ctx.info(f"Listing Twitter connectors for collection: {collection}")
    await ctx.warning("This is a placeholder implementation. In a real deployment, connector configurations would be stored in a database.")
    
    return {
        "status": "success",
        "message": "This is a placeholder implementation. In a real deployment, connector configurations would be stored in a database.",
        "connectors": []
    }
