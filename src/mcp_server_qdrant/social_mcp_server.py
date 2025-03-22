"""
Enhanced Qdrant MCP server with real-time data connectors.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union

from mcp.server.fastmcp import FastMCP, Context
from .tools.connectors import (
    twitter_connector,
    mastodon_connector
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server
mcp = FastMCP(
    "Qdrant Vector Database",
    description="Vector similarity search and real-time social media content indexing for LLMs",
    dependencies=[
        "qdrant-client",
        "fastembed",
        "httpx",
        "beautifulsoup4",
        "mastodon.py",
    ],
)


# ------------------------------
# Twitter Connector Tools
# ------------------------------

@mcp.tool()
async def setup_twitter_connector(
    ctx: Context,
    username: str,
    collection_name: str,
    include_retweets: bool = True,
    include_replies: bool = True,
    fetch_limit: int = 50,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    bearer_token: Optional[str] = None,
    update_interval_minutes: int = 15,
) -> Dict[str, Any]:
    """
    Set up a Twitter connector to automatically fetch and index tweets from a specific user.
    
    Args:
        username: Twitter username to monitor (without the @ symbol)
        collection_name: Name of the Qdrant collection to store tweets
        include_retweets: Whether to include retweets
        include_replies: Whether to include replies
        fetch_limit: Maximum number of tweets to fetch initially and in each update
        api_key: Twitter API key (optional)
        api_secret: Twitter API secret (optional)
        bearer_token: Twitter API bearer token (optional)
        update_interval_minutes: How often to check for updates (in minutes)
    
    Returns:
        Dict with connector configuration and status info
    """
    return await twitter_connector.setup_twitter_connector(
        username=username,
        collection_name=collection_name,
        include_retweets=include_retweets,
        include_replies=include_replies,
        fetch_limit=fetch_limit,
        api_key=api_key,
        api_secret=api_secret,
        bearer_token=bearer_token,
        update_interval_minutes=update_interval_minutes,
    )


@mcp.tool()
async def check_twitter_updates(
    ctx: Context,
    connector_id: str
) -> Dict[str, Any]:
    """
    Manually check for new tweets from a configured Twitter connector.
    
    Args:
        connector_id: ID of the connector to check
        
    Returns:
        Dict with update status
    """
    return await twitter_connector.check_twitter_updates(connector_id)


@mcp.tool()
async def list_twitter_connectors(ctx: Context) -> List[Dict[str, Any]]:
    """
    List all active Twitter connectors.
    
    Returns:
        List of connector configurations
    """
    return await twitter_connector.list_twitter_connectors()


@mcp.tool()
async def delete_twitter_connector(
    ctx: Context,
    connector_id: str
) -> Dict[str, Any]:
    """
    Delete a Twitter connector.
    
    Args:
        connector_id: ID of the connector to delete
        
    Returns:
        Status message
    """
    return await twitter_connector.delete_twitter_connector(connector_id)


# ------------------------------
# Mastodon Connector Tools
# ------------------------------

@mcp.tool()
async def setup_mastodon_connector(
    ctx: Context,
    account: str,
    instance_url: str,
    collection_name: str,
    include_boosts: bool = True,
    include_replies: bool = True,
    fetch_limit: int = 50,
    api_access_token: Optional[str] = None,
    update_interval_minutes: int = 15,
) -> Dict[str, Any]:
    """
    Set up a Mastodon connector to automatically fetch and index posts from a specific user.
    
    Args:
        account: Mastodon account username (without the @ symbol)
        instance_url: URL of the Mastodon instance (e.g., "https://mastodon.social")
        collection_name: Name of the Qdrant collection to store posts
        include_boosts: Whether to include boosted posts
        include_replies: Whether to include replies
        fetch_limit: Maximum number of posts to fetch initially and in each update
        api_access_token: Mastodon API access token (optional)
        update_interval_minutes: How often to check for updates (in minutes)
    
    Returns:
        Dict with connector configuration and status info
    """
    return await mastodon_connector.setup_mastodon_connector(
        account=account,
        instance_url=instance_url,
        collection_name=collection_name,
        include_boosts=include_boosts,
        include_replies=include_replies,
        fetch_limit=fetch_limit,
        api_access_token=api_access_token,
        update_interval_minutes=update_interval_minutes,
    )


@mcp.tool()
async def check_mastodon_updates(
    ctx: Context,
    connector_id: str
) -> Dict[str, Any]:
    """
    Manually check for new posts from a configured Mastodon connector.
    
    Args:
        connector_id: ID of the connector to check
        
    Returns:
        Dict with update status
    """
    return await mastodon_connector.check_mastodon_updates(connector_id)


@mcp.tool()
async def list_mastodon_connectors(ctx: Context) -> List[Dict[str, Any]]:
    """
    List all active Mastodon connectors.
    
    Returns:
        List of connector configurations
    """
    return await mastodon_connector.list_mastodon_connectors()


@mcp.tool()
async def delete_mastodon_connector(
    ctx: Context,
    connector_id: str
) -> Dict[str, Any]:
    """
    Delete a Mastodon connector.
    
    Args:
        connector_id: ID of the connector to delete
        
    Returns:
        Status message
    """
    return await mastodon_connector.delete_mastodon_connector(connector_id)


# ------------------------------
# Combined Social Media Tools
# ------------------------------

@mcp.tool()
async def list_all_social_connectors(ctx: Context) -> Dict[str, List[Dict[str, Any]]]:
    """
    List all active social media connectors of all types.
    
    Returns:
        Dict with lists of connectors by type
    """
    twitter_connectors = await twitter_connector.list_twitter_connectors()
    mastodon_connectors = await mastodon_connector.list_mastodon_connectors()
    
    return {
        "twitter": twitter_connectors,
        "mastodon": mastodon_connectors,
        "total_count": len(twitter_connectors) + len(mastodon_connectors),
    }


@mcp.tool()
async def search_social_content(
    ctx: Context,
    query: str,
    collection_name: str,
    source_types: Optional[List[str]] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search for social media content across all sources using vector similarity.
    
    Args:
        query: Search query
        collection_name: Name of the Qdrant collection to search
        source_types: List of source types to include (e.g., ["twitter", "mastodon"])
        limit: Maximum number of results to return
        
    Returns:
        List of matching social media posts with scores
    """
    from .search import nlq_search
    
    # Build filter based on source types
    filter_query = None
    if source_types:
        filter_query = {
            "should": [
                {"key": "source", "match": {"value": source_type}}
                for source_type in source_types
            ],
        }
    
    # Search using vector similarity
    results = await nlq_search(
        query=query,
        collection=collection_name,
        filter=filter_query,
        limit=limit,
    )
    
    return results


# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
