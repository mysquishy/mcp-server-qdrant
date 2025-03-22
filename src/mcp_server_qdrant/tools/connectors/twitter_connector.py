"""
Twitter connector for Qdrant MCP server.
This module provides functions for fetching and indexing Twitter content.
"""

import asyncio
import datetime
import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from bs4 import BeautifulSoup

# Import Qdrant client functions
from ...qdrant import QdrantConnector
from ...embeddings.factory import create_embedding_provider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for active connectors (in a production environment, use persistent storage)
active_twitter_connectors = {}

async def setup_twitter_connector(
    username: str,
    collection_name: str,
    include_retweets: bool = True,
    include_replies: bool = True,
    fetch_limit: int = 50,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    bearer_token: Optional[str] = None,
    update_interval_minutes: int = 15,
    metadata_prefix: str = "twitter",
) -> Dict[str, Any]:
    """
    Set up a Twitter connector to fetch and index tweets from a specific user.
    
    Args:
        username: Twitter username to monitor (without the @ symbol)
        collection_name: Name of the Qdrant collection to store tweets
        include_retweets: Whether to include retweets
        include_replies: Whether to include replies
        fetch_limit: Maximum number of tweets to fetch initially
        api_key: Twitter API key (optional)
        api_secret: Twitter API secret (optional)
        bearer_token: Twitter API bearer token (optional)
        update_interval_minutes: How often to check for updates
        metadata_prefix: Prefix for metadata fields in the collection
        
    Returns:
        Dict with connector configuration and status info
    """
    # Ensure username is properly formatted
    username = username.lstrip('@')
    
    # Generate a unique ID for this connector
    connector_id = f"twitter_{username}_{uuid.uuid4().hex[:8]}"
    
    # Get Qdrant client and collection
    qdrant_client = await get_qdrant_client()
    collection = await get_collection(collection_name)
    
    # Initialize connector configuration
    connector_config = {
        "id": connector_id,
        "type": "twitter",
        "username": username,
        "collection_name": collection_name,
        "include_retweets": include_retweets,
        "include_replies": include_replies,
        "fetch_limit": fetch_limit,
        "has_api_access": bool(bearer_token or (api_key and api_secret)),
        "update_interval_minutes": update_interval_minutes,
        "metadata_prefix": metadata_prefix,
        "last_updated": None,
        "last_tweet_id": None,
        "active": True,
        "error": None,
    }
    
    # Store API credentials securely (not in the main config that gets returned)
    if api_key and api_secret:
        connector_config["_api_key"] = api_key
        connector_config["_api_secret"] = api_secret
    
    if bearer_token:
        connector_config["_bearer_token"] = bearer_token
    
    try:
        # Initial fetch of tweets
        tweets, last_id = await fetch_tweets(
            username=username,
            limit=fetch_limit,
            include_retweets=include_retweets,
            include_replies=include_replies,
            bearer_token=bearer_token,
            api_key=api_key,
            api_secret=api_secret,
        )
        
        if tweets:
            # Process and store tweets
            await process_and_store_tweets(
                tweets=tweets,
                collection_name=collection_name,
                metadata_prefix=metadata_prefix,
            )
            
            # Update connector status
            connector_config["last_updated"] = datetime.datetime.now().isoformat()
            connector_config["last_tweet_id"] = last_id
            connector_config["initial_count"] = len(tweets)
            logger.info(f"Successfully set up Twitter connector for @{username} with {len(tweets)} tweets")
        else:
            logger.warning(f"No tweets found for @{username}")
            connector_config["error"] = "No tweets found"
    
    except Exception as e:
        logger.error(f"Error setting up Twitter connector for @{username}: {str(e)}")
        connector_config["error"] = str(e)
        connector_config["active"] = False
    
    # Store connector configuration
    active_twitter_connectors[connector_id] = connector_config
    
    # Schedule automatic updates if no errors occurred
    if connector_config["active"] and not connector_config["error"]:
        asyncio.create_task(
            schedule_updates(
                connector_id=connector_id,
                interval_minutes=update_interval_minutes,
            )
        )
    
    # Return a copy of the configuration (without sensitive API credentials)
    return {k: v for k, v in connector_config.items() if not k.startswith('_')}


async def fetch_tweets(
    username: str,
    limit: int = 50,
    include_retweets: bool = True,
    include_replies: bool = True,
    bearer_token: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    since_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Fetch tweets from a user's timeline.
    
    Will use Twitter API if credentials are provided, otherwise falls back to web scraping.
    
    Args:
        username: Twitter username
        limit: Maximum number of tweets to fetch
        include_retweets: Whether to include retweets
        include_replies: Whether to include replies
        bearer_token: Twitter API bearer token (optional)
        api_key: Twitter API key (optional)
        api_secret: Twitter API secret (optional)
        since_id: Only fetch tweets newer than this ID
        
    Returns:
        Tuple of (tweets, last_tweet_id)
    """
    # Check if API credentials are available
    if bearer_token or (api_key and api_secret):
        # Use Twitter API
        try:
            return await fetch_tweets_api(
                username=username,
                limit=limit,
                include_retweets=include_retweets,
                include_replies=include_replies,
                bearer_token=bearer_token,
                api_key=api_key,
                api_secret=api_secret,
                since_id=since_id,
            )
        except Exception as e:
            logger.warning(f"Error using Twitter API, falling back to web scraping: {str(e)}")
    
    # Fall back to web scraping if API fails or credentials not available
    return await fetch_tweets_web(
        username=username,
        limit=limit,
        include_retweets=include_retweets,
        include_replies=include_replies,
        since_id=since_id,
    )


async def fetch_tweets_api(
    username: str,
    limit: int = 50,
    include_retweets: bool = True,
    include_replies: bool = True,
    bearer_token: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    since_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Fetch tweets using the Twitter API.
    
    Args:
        username: Twitter username
        limit: Maximum number of tweets to fetch
        include_retweets: Whether to include retweets
        include_replies: Whether to include replies
        bearer_token: Twitter API bearer token (optional)
        api_key: Twitter API key (optional)
        api_secret: Twitter API secret (optional)
        since_id: Only fetch tweets newer than this ID
        
    Returns:
        Tuple of (tweets, last_tweet_id)
    """
    # Determine authorization method
    if bearer_token:
        headers = {"Authorization": f"Bearer {bearer_token}"}
        auth = None
    elif api_key and api_secret:
        # For API v1.1, get bearer token using OAuth flow
        auth_url = "https://api.twitter.com/oauth2/token"
        auth = (api_key, api_secret)
        data = {"grant_type": "client_credentials"}
        headers = {"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"}
        
        async with httpx.AsyncClient() as client:
            auth_response = await client.post(auth_url, auth=auth, data=data, headers=headers)
            auth_data = auth_response.json()
            
            if "access_token" not in auth_data:
                raise ValueError(f"Failed to get bearer token: {auth_data}")
            
            bearer_token = auth_data["access_token"]
            headers = {"Authorization": f"Bearer {bearer_token}"}
            auth = None
    else:
        raise ValueError("Either bearer_token or (api_key and api_secret) must be provided")
    
    # Construct API URL
    api_url = f"https://api.twitter.com/2/users/by/username/{username}/tweets"
    
    # Add query parameters
    params = {
        "max_results": min(100, limit),  # API maximum is 100
        "tweet.fields": "created_at,public_metrics,referenced_tweets,entities,attachments,author_id",
        "expansions": "attachments.media_keys,referenced_tweets.id,author_id",
        "media.fields": "url,preview_image_url,type",
        "user.fields": "name,username,profile_image_url",
    }
    
    if since_id:
        params["since_id"] = since_id
    
    async with httpx.AsyncClient() as client:
        response = await client.get(api_url, headers=headers, params=params)
        
        if response.status_code != 200:
            raise ValueError(f"Twitter API error: {response.status_code} - {response.text}")
        
        data = response.json()
        
        if "data" not in data:
            # No tweets found or error
            return [], None
        
        # Process tweets
        processed_tweets = []
        last_id = None
        
        for tweet in data["data"]:
            # Filter retweets if needed
            is_retweet = False
            if "referenced_tweets" in tweet:
                for ref in tweet["referenced_tweets"]:
                    if ref["type"] == "retweeted":
                        is_retweet = True
                        break
            
            if is_retweet and not include_retweets:
                continue
                
            # Filter replies if needed
            is_reply = "referenced_tweets" in tweet and any(ref["type"] == "replied_to" for ref in tweet["referenced_tweets"])
            if is_reply and not include_replies:
                continue
            
            # Process tweet into our standard format
            processed_tweet = {
                "id": tweet["id"],
                "text": tweet["text"],
                "created_at": tweet["created_at"],
                "author": {
                    "id": tweet["author_id"],
                    "username": next(
                        user["username"] 
                        for user in data.get("includes", {}).get("users", []) 
                        if user["id"] == tweet["author_id"]
                    ),
                },
                "metrics": tweet.get("public_metrics", {}),
                "is_retweet": is_retweet,
                "is_reply": is_reply,
                "media": [],
            }
            
            # Process media
            if "attachments" in tweet and "media_keys" in tweet["attachments"]:
                for media_key in tweet["attachments"]["media_keys"]:
                    media_item = next(
                        (m for m in data.get("includes", {}).get("media", []) if m["media_key"] == media_key),
                        None
                    )
                    if media_item:
                        processed_tweet["media"].append({
                            "type": media_item["type"],
                            "url": media_item.get("url", ""),
                            "preview_url": media_item.get("preview_image_url", ""),
                        })
            
            processed_tweets.append(processed_tweet)
            
            # Track newest tweet ID
            if last_id is None or tweet["id"] > last_id:
                last_id = tweet["id"]
        
        return processed_tweets, last_id


async def fetch_tweets_web(
    username: str,
    limit: int = 50,
    include_retweets: bool = True,
    include_replies: bool = True,
    since_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Fetch tweets by scraping Nitter (a Twitter web frontend).
    
    Used as a fallback when API credentials aren't available.
    
    Args:
        username: Twitter username
        limit: Maximum number of tweets to fetch
        include_retweets: Whether to include retweets
        include_replies: Whether to include replies
        since_id: Only fetch tweets newer than this ID
        
    Returns:
        Tuple of (tweets, last_tweet_id)
    """
    # Use Nitter instances for scraping (more reliable than Twitter web)
    nitter_instances = [
        "https://nitter.net",
        "https://nitter.kavin.rocks",
        "https://nitter.unixfox.eu",
    ]
    
    # Try each instance until we get a successful response
    tweets = []
    last_id = None
    success = False
    
    for instance in nitter_instances:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{instance}/{username}"
                response = await client.get(url)
                
                if response.status_code != 200:
                    continue
                
                # Parse HTML
                soup = BeautifulSoup(response.text, "html.parser")
                tweet_elements = soup.select(".timeline-item")
                
                if not tweet_elements:
                    continue
                
                # We got a valid response with tweets
                success = True
                
                # Process tweets
                for tweet_elem in tweet_elements[:limit]:
                    # Skip pinned tweets
                    if "pinned" in tweet_elem.get("class", []):
                        continue
                    
                    # Extract tweet ID
                    tweet_id = tweet_elem.get("data-tweet-id")
                    if not tweet_id:
                        continue
                    
                    # Skip if we only want tweets after a certain ID
                    if since_id and tweet_id <= since_id:
                        continue
                    
                    # Check if it's a retweet
                    is_retweet = bool(tweet_elem.select_one(".retweet-header"))
                    if is_retweet and not include_retweets:
                        continue
                    
                    # Check if it's a reply
                    is_reply = bool(tweet_elem.select_one(".replying-to"))
                    if is_reply and not include_replies:
                        continue
                    
                    # Extract tweet content
                    content_elem = tweet_elem.select_one(".tweet-content")
                    text = content_elem.get_text() if content_elem else ""
                    
                    # Extract timestamp
                    time_elem = tweet_elem.select_one(".tweet-date")
                    timestamp = time_elem.find("a").get("title") if time_elem else ""
                    
                    # Extract metrics
                    metrics = {
                        "retweet_count": extract_metric(tweet_elem, ".retweet-count"),
                        "reply_count": extract_metric(tweet_elem, ".reply-count"),
                        "like_count": extract_metric(tweet_elem, ".like-count"),
                    }
                    
                    # Extract media
                    media = []
                    media_elements = tweet_elem.select(".attachments img, .attachments video")
                    for media_elem in media_elements:
                        media_type = "photo" if media_elem.name == "img" else "video"
                        media_url = media_elem.get("src", "")
                        
                        # Convert relative URLs to absolute
                        if media_url.startswith("/"):
                            media_url = f"{instance}{media_url}"
                        
                        media.append({
                            "type": media_type,
                            "url": media_url,
                            "preview_url": media_url,
                        })
                    
                    # Create tweet object
                    tweet = {
                        "id": tweet_id,
                        "text": text,
                        "created_at": timestamp,
                        "author": {
                            "username": username,
                        },
                        "metrics": metrics,
                        "is_retweet": is_retweet,
                        "is_reply": is_reply,
                        "media": media,
                    }
                    
                    tweets.append(tweet)
                    
                    # Track newest tweet ID
                    if last_id is None or tweet_id > last_id:
                        last_id = tweet_id
                
                # We got what we needed from this instance
                break
                
        except Exception as e:
            logger.warning(f"Error scraping tweets from {instance}: {str(e)}")
    
    if not success:
        raise ValueError("Failed to fetch tweets from all Nitter instances")
    
    return tweets, last_id


def extract_metric(tweet_elem, selector):
    """Helper function to extract numerical metrics from tweet elements."""
    elem = tweet_elem.select_one(selector)
    if not elem:
        return 0
    
    text = elem.get_text().strip()
    if not text:
        return 0
    
    # Convert K/M suffixes to actual numbers
    if text.endswith("K"):
        return int(float(text[:-1]) * 1000)
    elif text.endswith("M"):
        return int(float(text[:-1]) * 1000000)
    
    try:
        return int(text)
    except ValueError:
        return 0


async def process_and_store_tweets(
    tweets: List[Dict[str, Any]],
    collection_name: str,
    metadata_prefix: str = "twitter",
) -> int:
    """
    Process tweets and store them in Qdrant.
    
    Args:
        tweets: List of tweets to process
        collection_name: Qdrant collection name
        metadata_prefix: Prefix for metadata fields
        
    Returns:
        Number of tweets stored
    """
    # Skip if no tweets
    if not tweets:
        return 0
    
    # Get Qdrant client and collection
    qdrant_client = await get_qdrant_client()
    collection = await get_collection(collection_name)
    
    # Process tweets into points for Qdrant
    points = []
    
    for tweet in tweets:
        # Generate a unique ID for this tweet
        point_id = f"{metadata_prefix}_{tweet['id']}"
        
        # Create payload with all metadata
        payload = {
            f"{metadata_prefix}_id": tweet["id"],
            f"{metadata_prefix}_text": tweet["text"],
            f"{metadata_prefix}_created_at": tweet["created_at"],
            f"{metadata_prefix}_author": tweet["author"]["username"],
            f"{metadata_prefix}_is_retweet": tweet["is_retweet"],
            f"{metadata_prefix}_is_reply": tweet["is_reply"],
            f"{metadata_prefix}_has_media": bool(tweet["media"]),
            f"{metadata_prefix}_retweet_count": tweet["metrics"].get("retweet_count", 0),
            f"{metadata_prefix}_reply_count": tweet["metrics"].get("reply_count", 0),
            f"{metadata_prefix}_like_count": tweet["metrics"].get("like_count", 0),
            "source": "twitter",
            "source_type": "social_media",
            "content_type": "tweet",
        }
        
        # Add media information if present
        if tweet["media"]:
            payload[f"{metadata_prefix}_media_types"] = [m["type"] for m in tweet["media"]]
            payload[f"{metadata_prefix}_media_urls"] = [m["url"] for m in tweet["media"]]
        
        # Generate embedding for the tweet text
        embedding = await embed_text(tweet["text"])
        
        # Create point
        point = {
            "id": point_id,
            "vector": embedding,
            "payload": payload,
        }
        
        points.append(point)
    
    # Batch upsert points to Qdrant
    if points:
        try:
            await qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
            )
            logger.info(f"Successfully stored {len(points)} tweets in {collection_name}")
            return len(points)
        except Exception as e:
            logger.error(f"Error storing tweets in Qdrant: {str(e)}")
            raise
    
    return 0


async def check_twitter_updates(connector_id: str) -> Dict[str, Any]:
    """
    Check for new tweets from a configured Twitter connector.
    
    Args:
        connector_id: ID of the connector to check
        
    Returns:
        Dict with update status
    """
    # Fetch connector configuration
    if connector_id not in active_twitter_connectors:
        raise ValueError(f"Twitter connector not found: {connector_id}")
    
    connector = active_twitter_connectors[connector_id]
    
    # Skip if connector is not active
    if not connector["active"]:
        return {
            "status": "skipped",
            "reason": "Connector is not active",
            "connector_id": connector_id,
        }
    
    # Prepare API credentials
    api_key = connector.get("_api_key")
    api_secret = connector.get("_api_secret")
    bearer_token = connector.get("_bearer_token")
    
    try:
        # Fetch new tweets since last update
        tweets, last_id = await fetch_tweets(
            username=connector["username"],
            limit=connector["fetch_limit"],
            include_retweets=connector["include_retweets"],
            include_replies=connector["include_replies"],
            bearer_token=bearer_token,
            api_key=api_key,
            api_secret=api_secret,
            since_id=connector["last_tweet_id"],
        )
        
        # Process and store new tweets
        if tweets:
            stored_count = await process_and_store_tweets(
                tweets=tweets,
                collection_name=connector["collection_name"],
                metadata_prefix=connector["metadata_prefix"],
            )
            
            # Update connector with new info
            connector["last_updated"] = datetime.datetime.now().isoformat()
            if last_id:
                connector["last_tweet_id"] = last_id
            
            logger.info(f"Updated Twitter connector for @{connector['username']} with {len(tweets)} new tweets")
            
            return {
                "status": "success",
                "connector_id": connector_id,
                "new_tweets": len(tweets),
                "stored_count": stored_count,
                "last_id": last_id,
            }
        else:
            # No new tweets found
            connector["last_updated"] = datetime.datetime.now().isoformat()
            
            logger.info(f"No new tweets found for @{connector['username']}")
            
            return {
                "status": "success",
                "connector_id": connector_id,
                "new_tweets": 0,
                "stored_count": 0,
            }
    
    except Exception as e:
        # Handle errors
        error_message = str(e)
        logger.error(f"Error updating Twitter connector for @{connector['username']}: {error_message}")
        
        # Detect rate limiting errors
        if "rate limit" in error_message.lower():
            # Temporary error, don't disable connector
            return {
                "status": "error",
                "connector_id": connector_id,
                "error": f"Rate limit exceeded: {error_message}",
                "retry_after": "15 minutes",  # Default retry after rate limit
            }
        
        # Mark other errors in the connector
        connector["error"] = error_message
        
        # Don't disable connector on temporary errors
        if "timeout" in error_message.lower() or "connection" in error_message.lower():
            return {
                "status": "error",
                "connector_id": connector_id,
                "error": f"Temporary error: {error_message}",
                "retry": True,
            }
        
        # Disable connector on persistent errors
        connector["active"] = False
        
        return {
            "status": "error",
            "connector_id": connector_id,
            "error": error_message,
            "connector_disabled": True,
        }


async def schedule_updates(connector_id: str, interval_minutes: int = 15):
    """
    Schedule periodic updates for a Twitter connector.
    
    Args:
        connector_id: ID of the connector to update
        interval_minutes: Time between updates in minutes
    """
    interval_seconds = max(1, interval_minutes) * 60
    
    while True:
        # Check if connector still exists and is active
        if connector_id not in active_twitter_connectors:
            logger.info(f"Stopping updates for removed connector: {connector_id}")
            break
        
        connector = active_twitter_connectors[connector_id]
        if not connector["active"]:
            logger.info(f"Stopping updates for inactive connector: {connector_id}")
            break
        
        # Wait for the specified interval
        await asyncio.sleep(interval_seconds)
        
        # Check for updates
        try:
            await check_twitter_updates(connector_id)
        except Exception as e:
            logger.error(f"Error in scheduled update for connector {connector_id}: {str(e)}")


async def list_twitter_connectors() -> List[Dict[str, Any]]:
    """
    List all active Twitter connectors.
    
    Returns:
        List of connector configurations (without sensitive API credentials)
    """
    # Return public information about all connectors (strip API credentials)
    return [
        {k: v for k, v in connector.items() if not k.startswith('_')}
        for connector in active_twitter_connectors.values()
    ]


async def delete_twitter_connector(connector_id: str) -> Dict[str, Any]:
    """
    Delete a Twitter connector.
    
    Args:
        connector_id: ID of the connector to delete
        
    Returns:
        Status message
    """
    if connector_id not in active_twitter_connectors:
        raise ValueError(f"Twitter connector not found: {connector_id}")
    
    # Get connector info for the response
    connector = active_twitter_connectors[connector_id]
    username = connector["username"]
    
    # Remove the connector
    del active_twitter_connectors[connector_id]
    
    logger.info(f"Deleted Twitter connector for @{username} (ID: {connector_id})")
    
    return {
        "status": "success",
        "message": f"Twitter connector for @{username} deleted",
        "connector_id": connector_id,
    }


# Missing helper functions - need to implement
async def get_qdrant_client():
    """Get Qdrant client singleton."""
    # This would typically be implemented to return a singleton QdrantConnector instance
    # For now, we'll return None as a placeholder
    return None


async def get_collection(collection_name: str):
    """Get or create a Qdrant collection."""
    # This would typically be implemented to get or create a collection
    # For now, we'll return None as a placeholder
    return None


async def embed_text(text: str):
    """Generate embedding for text using the configured provider."""
    # This would typically use the embedding provider to generate embeddings
    # For now, we'll return a placeholder empty vector
    return []
