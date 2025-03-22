"""
Mastodon connector for Qdrant MCP server.
This module provides functions for fetching and indexing Mastodon content.
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

try:
    from mastodon import Mastodon as MastodonAPI
except ImportError:
    MastodonAPI = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for active connectors (in a production environment, use persistent storage)
active_mastodon_connectors = {}

async def setup_mastodon_connector(
    account: str,
    instance_url: str,
    collection_name: str,
    include_boosts: bool = True,
    include_replies: bool = True,
    fetch_limit: int = 50,
    api_access_token: Optional[str] = None,
    update_interval_minutes: int = 15,
    metadata_prefix: str = "mastodon",
) -> Dict[str, Any]:
    """
    Set up a Mastodon connector to fetch and index posts from a specific user.
    
    Args:
        account: Mastodon account username (without the @ symbol)
        instance_url: URL of the Mastodon instance (e.g., "https://mastodon.social")
        collection_name: Name of the Qdrant collection to store posts
        include_boosts: Whether to include boosted posts
        include_replies: Whether to include replies
        fetch_limit: Maximum number of posts to fetch initially
        api_access_token: Mastodon API access token (optional)
        update_interval_minutes: How often to check for updates
        metadata_prefix: Prefix for metadata fields in the collection
        
    Returns:
        Dict with connector configuration and status info
    """
    # Ensure account is properly formatted
    account = account.lstrip('@')
    
    # Ensure instance URL is properly formatted
    if not instance_url.startswith("http"):
        instance_url = f"https://{instance_url}"
    
    if instance_url.endswith("/"):
        instance_url = instance_url[:-1]
    
    # Generate a unique ID for this connector
    connector_id = f"mastodon_{account}_{uuid.uuid4().hex[:8]}"
    
    # Initialize connector configuration
    connector_config = {
        "id": connector_id,
        "type": "mastodon",
        "account": account,
        "instance_url": instance_url,
        "collection_name": collection_name,
        "include_boosts": include_boosts,
        "include_replies": include_replies,
        "fetch_limit": fetch_limit,
        "has_api_access": bool(api_access_token and MastodonAPI),
        "update_interval_minutes": update_interval_minutes,
        "metadata_prefix": metadata_prefix,
        "last_updated": None,
        "last_post_id": None,
        "active": True,
        "error": None,
    }
    
    # Store API credentials securely (not in the main config that gets returned)
    if api_access_token:
        connector_config["_api_access_token"] = api_access_token
    
    try:
        # Check if the mastodon.py library is available
        if api_access_token and not MastodonAPI:
            logger.warning("mastodon.py library not installed. Falling back to web scraping.")
            connector_config["has_api_access"] = False
        
        # Initial fetch of posts
        posts, last_id = await fetch_posts(
            account=account,
            instance_url=instance_url,
            limit=fetch_limit,
            include_boosts=include_boosts,
            include_replies=include_replies,
            api_access_token=api_access_token,
        )
        
        if posts:
            # Process and store posts
            await process_and_store_posts(
                posts=posts,
                collection_name=collection_name,
                metadata_prefix=metadata_prefix,
            )
            
            # Update connector status
            connector_config["last_updated"] = datetime.datetime.now().isoformat()
            connector_config["last_post_id"] = last_id
            connector_config["initial_count"] = len(posts)
            logger.info(f"Successfully set up Mastodon connector for @{account}@{instance_url.replace('https://', '')} with {len(posts)} posts")
        else:
            logger.warning(f"No posts found for @{account}@{instance_url.replace('https://', '')}")
            connector_config["error"] = "No posts found"
    
    except Exception as e:
        logger.error(f"Error setting up Mastodon connector for @{account}@{instance_url.replace('https://', '')}: {str(e)}")
        connector_config["error"] = str(e)
        connector_config["active"] = False
    
    # Store connector configuration
    active_mastodon_connectors[connector_id] = connector_config
    
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


async def fetch_posts(
    account: str,
    instance_url: str,
    limit: int = 50,
    include_boosts: bool = True,
    include_replies: bool = True,
    api_access_token: Optional[str] = None,
    since_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Fetch posts from a Mastodon user's timeline.
    
    Will use Mastodon API if credentials are provided, otherwise falls back to web scraping.
    
    Args:
        account: Mastodon account username
        instance_url: URL of the Mastodon instance
        limit: Maximum number of posts to fetch
        include_boosts: Whether to include boosted posts
        include_replies: Whether to include replies
        api_access_token: Mastodon API access token (optional)
        since_id: Only fetch posts newer than this ID
        
    Returns:
        Tuple of (posts, last_post_id)
    """
    # Check if API credentials are available and mastodon.py is installed
    if api_access_token and MastodonAPI:
        # Use Mastodon API
        try:
            return await fetch_posts_api(
                account=account,
                instance_url=instance_url,
                limit=limit,
                include_boosts=include_boosts,
                include_replies=include_replies,
                api_access_token=api_access_token,
                since_id=since_id,
            )
        except Exception as e:
            logger.warning(f"Error using Mastodon API, falling back to web scraping: {str(e)}")
    
    # Fall back to web scraping if API fails or credentials not available
    return await fetch_posts_web(
        account=account,
        instance_url=instance_url,
        limit=limit,
        include_boosts=include_boosts,
        include_replies=include_replies,
        since_id=since_id,
    )


async def fetch_posts_api(
    account: str,
    instance_url: str,
    limit: int = 50,
    include_boosts: bool = True,
    include_replies: bool = True,
    api_access_token: str = None,
    since_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Fetch posts using the Mastodon API.
    
    Args:
        account: Mastodon account username
        instance_url: URL of the Mastodon instance
        limit: Maximum number of posts to fetch
        include_boosts: Whether to include boosted posts
        include_replies: Whether to include replies
        api_access_token: Mastodon API access token
        since_id: Only fetch posts newer than this ID
        
    Returns:
        Tuple of (posts, last_post_id)
    """
    # This function uses synchronous Mastodon.py API, so we run it in a thread pool
    async def run_in_thread():
        # Create Mastodon API client
        mastodon = MastodonAPI(
            access_token=api_access_token,
            api_base_url=instance_url
        )
        
        # Look up account ID
        account_info = mastodon.account_lookup(account)
        account_id = account_info["id"]
        
        # Fetch account's posts
        statuses = mastodon.account_statuses(
            id=account_id,
            limit=limit,
            since_id=since_id
        )
        
        # Process posts
        processed_posts = []
        last_id = None
        
        for status in statuses:
            # Filter boosts if needed
            is_boost = "reblog" in status and status["reblog"]
            if is_boost and not include_boosts:
                continue
                
            # Filter replies if needed
            is_reply = status["in_reply_to_id"] is not None
            if is_reply and not include_replies:
                continue
            
            # Process content with CWs
            content = status["content"]
            spoiler_text = status.get("spoiler_text", "")
            if spoiler_text:
                content = f"[CW: {spoiler_text}]\n{content}"
            
            # Process post into our standard format
            processed_post = {
                "id": status["id"],
                "text": BeautifulSoup(content, "html.parser").get_text(),
                "html": content,
                "created_at": status["created_at"].isoformat() if isinstance(status["created_at"], datetime.datetime) else status["created_at"],
                "url": status["url"],
                "author": {
                    "id": status["account"]["id"],
                    "username": status["account"]["username"],
                    "display_name": status["account"]["display_name"],
                    "url": status["account"]["url"],
                },
                "metrics": {
                    "replies_count": status["replies_count"],
                    "reblogs_count": status["reblogs_count"],
                    "favourites_count": status["favourites_count"],
                },
                "is_boost": is_boost,
                "is_reply": is_reply,
                "visibility": status["visibility"],
                "media": [],
            }
            
            # Process media
            if "media_attachments" in status and status["media_attachments"]:
                for media in status["media_attachments"]:
                    processed_post["media"].append({
                        "type": media["type"],
                        "url": media.get("url", ""),
                        "preview_url": media.get("preview_url", ""),
                        "description": media.get("description", ""),
                    })
            
            processed_posts.append(processed_post)
            
            # Track newest post ID
            post_id_str = str(status["id"])
            if last_id is None or post_id_str > last_id:
                last_id = post_id_str
        
        return processed_posts, last_id
    
    # Run in thread since Mastodon.py is synchronous
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: asyncio.run(run_in_thread()))


async def fetch_posts_web(
    account: str,
    instance_url: str,
    limit: int = 50,
    include_boosts: bool = True,
    include_replies: bool = True,
    since_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Fetch posts by scraping a Mastodon instance's web interface.
    
    Used as a fallback when API credentials aren't available.
    
    Args:
        account: Mastodon account username
        instance_url: URL of the Mastodon instance
        limit: Maximum number of posts to fetch
        include_boosts: Whether to include boosted posts
        include_replies: Whether to include replies
        since_id: Only fetch posts newer than this ID
        
    Returns:
        Tuple of (posts, last_post_id)
    """
    # Build the profile URL
    profile_url = f"{instance_url}/@{account}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(profile_url)
            
            if response.status_code != 200:
                raise ValueError(f"Error accessing {profile_url}: HTTP {response.status_code}")
            
            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract post elements
            post_elements = soup.select(".status-card, .detailed-status")
            
            if not post_elements:
                logger.warning(f"No posts found at {profile_url} via web scraping")
                return [], None
            
            # Process posts
            posts = []
            last_id = None
            
            for i, post_elem in enumerate(post_elements[:limit]):
                try:
                    # Extract post ID
                    post_url = post_elem.select_one("a.status__relative-time").get("href")
                    post_id = post_url.split("/")[-1] if post_url else None
                    
                    if not post_id:
                        continue
                    
                    # Skip if we only want posts after a certain ID
                    if since_id and post_id <= since_id:
                        continue
                    
                    # Check if it's a boost
                    is_boost = bool(post_elem.select_one(".status__prepend"))
                    if is_boost and not include_boosts:
                        continue
                    
                    # Check if it's a reply
                    is_reply = bool(post_elem.select_one(".conversation__status__header"))
                    if is_reply and not include_replies:
                        continue
                    
                    # Extract content
                    content_elem = post_elem.select_one(".status__content")
                    text = content_elem.get_text().strip() if content_elem else ""
                    
                    # Extract content warnings if present
                    cw_elem = post_elem.select_one(".status__content-warning-text")
                    if cw_elem:
                        cw_text = cw_elem.get_text().strip()
                        text = f"[CW: {cw_text}]\n{text}"
                    
                    # Extract timestamp
                    time_elem = post_elem.select_one("time")
                    timestamp = time_elem.get("datetime") if time_elem else ""
                    
                    # Extract metrics
                    metrics = {
                        "replies_count": extract_count(post_elem, ".status__action-bar__counter--replies"),
                        "reblogs_count": extract_count(post_elem, ".status__action-bar__counter--reblog"),
                        "favourites_count": extract_count(post_elem, ".status__action-bar__counter--favourite"),
                    }
                    
                    # Extract media
                    media = []
                    media_elements = post_elem.select(".media-gallery img, .status__attachment video")
                    for media_elem in media_elements:
                        media_type = "image" if media_elem.name == "img" else "video"
                        media_url = media_elem.get("src", "") or media_elem.get("data-src", "")
                        
                        # Convert relative URLs to absolute
                        if media_url.startswith("/"):
                            media_url = f"{instance_url}{media_url}"
                        
                        media.append({
                            "type": media_type,
                            "url": media_url,
                            "preview_url": media_url,
                        })
                    
                    # Create post object
                    post = {
                        "id": post_id,
                        "text": text,
                        "created_at": timestamp,
                        "url": f"{instance_url}{post_url}",
                        "author": {
                            "username": account,
                            "instance": instance_url.replace("https://", ""),
                        },
                        "metrics": metrics,
                        "is_boost": is_boost,
                        "is_reply": is_reply,
                        "visibility": "public",  # Assume public since we could scrape it
                        "media": media,
                    }
                    
                    posts.append(post)
                    
                    # Track newest post ID
                    if last_id is None or post_id > last_id:
                        last_id = post_id
                        
                except Exception as e:
                    logger.warning(f"Error processing post {i}: {str(e)}")
            
            return posts, last_id
            
    except Exception as e:
        logger.error(f"Error scraping posts from {profile_url}: {str(e)}")
        raise


def extract_count(post_elem, selector):
    """Helper function to extract numerical metrics from post elements."""
    elem = post_elem.select_one(selector)
    if not elem:
        return 0
    
    text = elem.get_text().strip()
    if not text or text == "0":
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


async def process_and_store_posts(
    posts: List[Dict[str, Any]],
    collection_name: str,
    metadata_prefix: str = "mastodon",
) -> int:
    """
    Process Mastodon posts and store them in Qdrant.
    
    Args:
        posts: List of posts to process
        collection_name: Qdrant collection name
        metadata_prefix: Prefix for metadata fields
        
    Returns:
        Number of posts stored
    """
    # Skip if no posts
    if not posts:
        return 0
    
    # Get Qdrant client and collection
    qdrant_client = await get_qdrant_client()
    collection = await get_collection(collection_name)
    
    # Process posts into points for Qdrant
    points = []
    
    for post in posts:
        # Generate a unique ID for this post
        point_id = f"{metadata_prefix}_{post['id']}"
        
        # Get the instance domain from the author or URL
        instance = None
        if "author" in post and "instance" in post["author"]:
            instance = post["author"]["instance"]
        elif post.get("url"):
            try:
                instance = post["url"].split("/")[2]
            except (IndexError, ValueError):
                pass
        
        # Create payload with all metadata
        payload = {
            f"{metadata_prefix}_id": post["id"],
            f"{metadata_prefix}_text": post["text"],
            f"{metadata_prefix}_created_at": post["created_at"],
            f"{metadata_prefix}_author": post["author"]["username"],
            f"{metadata_prefix}_instance": instance,
            f"{metadata_prefix}_is_boost": post["is_boost"],
            f"{metadata_prefix}_is_reply": post["is_reply"],
            f"{metadata_prefix}_has_media": bool(post["media"]),
            f"{metadata_prefix}_visibility": post.get("visibility", "unknown"),
            f"{metadata_prefix}_url": post.get("url", ""),
            f"{metadata_prefix}_replies_count": post["metrics"].get("replies_count", 0),
            f"{metadata_prefix}_reblogs_count": post["metrics"].get("reblogs_count", 0),
            f"{metadata_prefix}_favourites_count": post["metrics"].get("favourites_count", 0),
            "source": "mastodon",
            "source_type": "social_media",
            "content_type": "post",
        }
        
        # Add content warning flag if present
        if post["text"].startswith("[CW:"):
            payload[f"{metadata_prefix}_has_cw"] = True
        
        # Add media information if present
        if post["media"]:
            payload[f"{metadata_prefix}_media_types"] = [m["type"] for m in post["media"]]
            payload[f"{metadata_prefix}_media_urls"] = [m["url"] for m in post["media"]]
        
        # Add HTML content if available
        if "html" in post:
            payload[f"{metadata_prefix}_html"] = post["html"]
        
        # Generate embedding for the post text
        embedding = await embed_text(post["text"])
        
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
            logger.info(f"Successfully stored {len(points)} Mastodon posts in {collection_name}")
            return len(points)
        except Exception as e:
            logger.error(f"Error storing Mastodon posts in Qdrant: {str(e)}")
            raise
    
    return 0


async def check_mastodon_updates(connector_id: str) -> Dict[str, Any]:
    """
    Check for new posts from a configured Mastodon connector.
    
    Args:
        connector_id: ID of the connector to check
        
    Returns:
        Dict with update status
    """
    # Fetch connector configuration
    if connector_id not in active_mastodon_connectors:
        raise ValueError(f"Mastodon connector not found: {connector_id}")
    
    connector = active_mastodon_connectors[connector_id]
    
    # Skip if connector is not active
    if not connector["active"]:
        return {
            "status": "skipped",
            "reason": "Connector is not active",
            "connector_id": connector_id,
        }
    
    # Prepare API credentials
    api_access_token = connector.get("_api_access_token")
    
    try:
        # Fetch new posts since last update
        posts, last_id = await fetch_posts(
            account=connector["account"],
            instance_url=connector["instance_url"],
            limit=connector["fetch_limit"],
            include_boosts=connector["include_boosts"],
            include_replies=connector["include_replies"],
            api_access_token=api_access_token,
            since_id=connector["last_post_id"],
        )
        
        # Process and store new posts
        if posts:
            stored_count = await process_and_store_posts(
                posts=posts,
                collection_name=connector["collection_name"],
                metadata_prefix=connector["metadata_prefix"],
            )
            
            # Update connector with new info
            connector["last_updated"] = datetime.datetime.now().isoformat()
            if last_id:
                connector["last_post_id"] = last_id
            
            logger.info(f"Updated Mastodon connector for @{connector['account']}@{connector['instance_url'].replace('https://', '')} with {len(posts)} new posts")
            
            return {
                "status": "success",
                "connector_id": connector_id,
                "new_posts": len(posts),
                "stored_count": stored_count,
                "last_id": last_id,
            }
        else:
            # No new posts found
            connector["last_updated"] = datetime.datetime.now().isoformat()
            
            logger.info(f"No new posts found for @{connector['account']}@{connector['instance_url'].replace('https://', '')}")
            
            return {
                "status": "success",
                "connector_id": connector_id,
                "new_posts": 0,
                "stored_count": 0,
            }
    
    except Exception as e:
        # Handle errors
        error_message = str(e)
        logger.error(f"Error updating Mastodon connector for @{connector['account']}@{connector['instance_url'].replace('https://', '')}: {error_message}")
        
        # Detect rate limiting errors
        if "rate limit" in error_message.lower() or "429" in error_message:
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
    Schedule periodic updates for a Mastodon connector.
    
    Args:
        connector_id: ID of the connector to update
        interval_minutes: Time between updates in minutes
    """
    interval_seconds = max(1, interval_minutes) * 60
    
    while True:
        # Check if connector still exists and is active
        if connector_id not in active_mastodon_connectors:
            logger.info(f"Stopping updates for removed connector: {connector_id}")
            break
        
        connector = active_mastodon_connectors[connector_id]
        if not connector["active"]:
            logger.info(f"Stopping updates for inactive connector: {connector_id}")
            break
        
        # Wait for the specified interval
        await asyncio.sleep(interval_seconds)
        
        # Check for updates
        try:
            await check_mastodon_updates(connector_id)
        except Exception as e:
            logger.error(f"Error in scheduled update for connector {connector_id}: {str(e)}")


async def list_mastodon_connectors() -> List[Dict[str, Any]]:
    """
    List all active Mastodon connectors.
    
    Returns:
        List of connector configurations (without sensitive API credentials)
    """
    # Return public information about all connectors (strip API credentials)
    return [
        {k: v for k, v in connector.items() if not k.startswith('_')}
        for connector in active_mastodon_connectors.values()
    ]


async def delete_mastodon_connector(connector_id: str) -> Dict[str, Any]:
    """
    Delete a Mastodon connector.
    
    Args:
        connector_id: ID of the connector to delete
        
    Returns:
        Status message
    """
    if connector_id not in active_mastodon_connectors:
        raise ValueError(f"Mastodon connector not found: {connector_id}")
    
    # Get connector info for the response
    connector = active_mastodon_connectors[connector_id]
    account = connector["account"]
    instance = connector["instance_url"].replace("https://", "")
    
    # Remove the connector
    del active_mastodon_connectors[connector_id]
    
    logger.info(f"Deleted Mastodon connector for @{account}@{instance} (ID: {connector_id})")
    
    return {
        "status": "success",
        "message": f"Mastodon connector for @{account}@{instance} deleted",
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
