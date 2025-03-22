"""
Example script demonstrating how to use the Twitter connector 
to index and search tweets.
"""

import asyncio
import os
import logging
from pprint import pprint

from mcp_server_qdrant.tools.connectors.twitter_connector import (
    setup_twitter_connector,
    check_twitter_updates,
    list_twitter_connectors,
    delete_twitter_connector,
)
from mcp_server_qdrant.search import nlq_search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
COLLECTION_NAME = "twitter_data"
TWITTER_USERNAME = "anthropic"

# API credentials (optional)
TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY")
TWITTER_API_SECRET = os.environ.get("TWITTER_API_SECRET")
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")


async def main():
    """Run the Twitter connector example."""
    logger.info(f"Setting up Twitter connector for @{TWITTER_USERNAME}")
    
    # Set up Twitter connector
    result = await setup_twitter_connector(
        username=TWITTER_USERNAME,
        collection_name=COLLECTION_NAME,
        include_retweets=True,
        include_replies=True,
        fetch_limit=50,
        api_key=TWITTER_API_KEY,
        api_secret=TWITTER_API_SECRET,
        bearer_token=TWITTER_BEARER_TOKEN,
        update_interval_minutes=30,
    )
    
    logger.info("Twitter connector setup complete:")
    pprint(result)
    
    # Get the connector ID for future reference
    connector_id = result["id"]
    
    # Wait for initial data collection
    logger.info("Waiting 10 seconds for initial data collection...")
    await asyncio.sleep(10)
    
    # List all Twitter connectors
    logger.info("Listing all Twitter connectors:")
    connectors = await list_twitter_connectors()
    pprint(connectors)
    
    # Manually check for updates
    logger.info("Manually checking for new tweets:")
    update_result = await check_twitter_updates(connector_id)
    pprint(update_result)
    
    # Search for tweets about AI
    logger.info("Searching for tweets about AI:")
    search_results = await nlq_search(
        query="AI",
        collection_name=COLLECTION_NAME,
        filter={
            "must": [
                {"key": "source", "match": {"value": "twitter"}}
            ]
        },
        limit=5,
    )
    
    logger.info(f"Found {len(search_results)} results:")
    for i, result in enumerate(search_results, 1):
        logger.info(f"Result {i}:")
        logger.info(f"  Score: {result['score']}")
        logger.info(f"  Text: {result['payload']['twitter_text']}")
        logger.info(f"  Author: @{result['payload']['twitter_author']}")
        logger.info(f"  Created: {result['payload']['twitter_created_at']}")
        logger.info(f"  Metrics: {result['payload']['twitter_like_count']} likes, "
                    f"{result['payload']['twitter_retweet_count']} retweets")
        logger.info("")
    
    # Search for tweets with media
    logger.info("Searching for tweets with media:")
    media_results = await nlq_search(
        query="AI",
        collection_name=COLLECTION_NAME,
        filter={
            "must": [
                {"key": "source", "match": {"value": "twitter"}},
                {"key": "twitter_has_media", "match": {"value": True}}
            ]
        },
        limit=5,
    )
    
    logger.info(f"Found {len(media_results)} results with media")
    
    # Delete the connector (uncomment to test)
    # logger.info("Deleting the Twitter connector:")
    # delete_result = await delete_twitter_connector(connector_id)
    # pprint(delete_result)


if __name__ == "__main__":
    asyncio.run(main())
