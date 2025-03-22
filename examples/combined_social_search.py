"""
Example script demonstrating how to use both Twitter and Mastodon connectors
together and search across all social media content.
"""

import asyncio
import os
import logging
from pprint import pprint

from mcp_server_qdrant.tools.connectors.twitter_connector import setup_twitter_connector
from mcp_server_qdrant.tools.connectors.mastodon_connector import setup_mastodon_connector
from mcp_server_qdrant.search import nlq_search as search_social_content

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
COLLECTION_NAME = "social_media"
TWITTER_USERNAME = "anthropic"
MASTODON_ACCOUNT = "Gargron"
MASTODON_INSTANCE = "https://mastodon.social"

# API credentials (optional)
TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY")
TWITTER_API_SECRET = os.environ.get("TWITTER_API_SECRET")
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")
MASTODON_ACCESS_TOKEN = os.environ.get("MASTODON_ACCESS_TOKEN")


async def list_all_social_connectors():
    """List all active social media connectors of all types."""
    from mcp_server_qdrant.tools.connectors.twitter_connector import list_twitter_connectors
    from mcp_server_qdrant.tools.connectors.mastodon_connector import list_mastodon_connectors
    
    twitter_connectors = await list_twitter_connectors()
    mastodon_connectors = await list_mastodon_connectors()
    
    return {
        "twitter": twitter_connectors,
        "mastodon": mastodon_connectors,
        "total_count": len(twitter_connectors) + len(mastodon_connectors),
    }


async def main():
    """Run the combined social media connectors example."""
    logger.info("Setting up connectors for both Twitter and Mastodon")
    
    # Set up Twitter connector
    twitter_result = await setup_twitter_connector(
        username=TWITTER_USERNAME,
        collection_name=COLLECTION_NAME,
        include_retweets=True,
        include_replies=False,
        fetch_limit=30,
        api_key=TWITTER_API_KEY,
        api_secret=TWITTER_API_SECRET,
        bearer_token=TWITTER_BEARER_TOKEN,
    )
    
    logger.info("Twitter connector setup complete:")
    pprint(twitter_result)
    
    # Set up Mastodon connector
    mastodon_result = await setup_mastodon_connector(
        account=MASTODON_ACCOUNT,
        instance_url=MASTODON_INSTANCE,
        collection_name=COLLECTION_NAME,
        include_boosts=True,
        include_replies=False,
        fetch_limit=30,
        api_access_token=MASTODON_ACCESS_TOKEN,
    )
    
    logger.info("Mastodon connector setup complete:")
    pprint(mastodon_result)
    
    # Wait for initial data collection
    logger.info("Waiting 10 seconds for initial data collection...")
    await asyncio.sleep(10)
    
    # List all social connectors
    logger.info("Listing all social media connectors:")
    all_connectors = await list_all_social_connectors()
    pprint(all_connectors)
    
    # Search across all social media
    logger.info("Searching for content about 'AI ethics' across all social media platforms:")
    results = await search_social_content(
        query="AI ethics",
        collection_name=COLLECTION_NAME,
        filter=None,  # No filter means search all sources
        limit=10,
    )
    
    logger.info(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        source = result["payload"]["source"]
        logger.info(f"Result {i} [{source}]:")
        logger.info(f"  Score: {result['score']}")
        
        if source == "twitter":
            logger.info(f"  Text: {result['payload']['twitter_text']}")
            logger.info(f"  Author: @{result['payload']['twitter_author']}")
            logger.info(f"  Created: {result['payload']['twitter_created_at']}")
            logger.info(f"  Metrics: {result['payload']['twitter_like_count']} likes, "
                        f"{result['payload']['twitter_retweet_count']} retweets")
        elif source == "mastodon":
            logger.info(f"  Text: {result['payload']['mastodon_text']}")
            logger.info(f"  Author: @{result['payload']['mastodon_author']}@{result['payload']['mastodon_instance']}")
            logger.info(f"  Created: {result['payload']['mastodon_created_at']}")
            logger.info(f"  URL: {result['payload']['mastodon_url']}")
        
        logger.info("")
    
    # Search with source filtering
    logger.info("Searching only Twitter content:")
    twitter_results = await search_social_content(
        query="AI ethics",
        collection_name=COLLECTION_NAME,
        filter={
            "must": [
                {"key": "source", "match": {"value": "twitter"}}
            ]
        },
        limit=5,
    )
    
    logger.info(f"Found {len(twitter_results)} Twitter results")
    
    # Search with source filtering
    logger.info("Searching only Mastodon content:")
    mastodon_results = await search_social_content(
        query="AI ethics",
        collection_name=COLLECTION_NAME,
        filter={
            "must": [
                {"key": "source", "match": {"value": "mastodon"}}
            ]
        },
        limit=5,
    )
    
    logger.info(f"Found {len(mastodon_results)} Mastodon results")


if __name__ == "__main__":
    asyncio.run(main())
