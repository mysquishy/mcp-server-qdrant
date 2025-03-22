"""
Example script demonstrating how to use the Mastodon connector 
to index and search Mastodon posts.
"""

import asyncio
import os
import logging
from pprint import pprint

from mcp_server_qdrant.tools.connectors.mastodon_connector import (
    setup_mastodon_connector,
    check_mastodon_updates,
    list_mastodon_connectors,
    delete_mastodon_connector,
)
from mcp_server_qdrant.search import nlq_search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
COLLECTION_NAME = "mastodon_data"
MASTODON_ACCOUNT = "Gargron"
MASTODON_INSTANCE = "https://mastodon.social"

# API credentials (optional)
MASTODON_ACCESS_TOKEN = os.environ.get("MASTODON_ACCESS_TOKEN")


async def main():
    """Run the Mastodon connector example."""
    logger.info(f"Setting up Mastodon connector for @{MASTODON_ACCOUNT}@{MASTODON_INSTANCE.replace('https://', '')}")
    
    # Set up Mastodon connector
    result = await setup_mastodon_connector(
        account=MASTODON_ACCOUNT,
        instance_url=MASTODON_INSTANCE,
        collection_name=COLLECTION_NAME,
        include_boosts=True,
        include_replies=True,
        fetch_limit=50,
        api_access_token=MASTODON_ACCESS_TOKEN,
        update_interval_minutes=30,
    )
    
    logger.info("Mastodon connector setup complete:")
    pprint(result)
    
    # Get the connector ID for future reference
    connector_id = result["id"]
    
    # Wait for initial data collection
    logger.info("Waiting 10 seconds for initial data collection...")
    await asyncio.sleep(10)
    
    # List all Mastodon connectors
    logger.info("Listing all Mastodon connectors:")
    connectors = await list_mastodon_connectors()
    pprint(connectors)
    
    # Manually check for updates
    logger.info("Manually checking for new posts:")
    update_result = await check_mastodon_updates(connector_id)
    pprint(update_result)
    
    # Search for posts about open source
    logger.info("Searching for posts about open source:")
    search_results = await nlq_search(
        query="open source",
        collection_name=COLLECTION_NAME,
        filter={
            "must": [
                {"key": "source", "match": {"value": "mastodon"}}
            ]
        },
        limit=5,
    )
    
    logger.info(f"Found {len(search_results)} results:")
    for i, result in enumerate(search_results, 1):
        logger.info(f"Result {i}:")
        logger.info(f"  Score: {result['score']}")
        logger.info(f"  Text: {result['payload']['mastodon_text']}")
        logger.info(f"  Author: @{result['payload']['mastodon_author']}@{result['payload']['mastodon_instance']}")
        logger.info(f"  Created: {result['payload']['mastodon_created_at']}")
        logger.info(f"  URL: {result['payload']['mastodon_url']}")
        logger.info("")
    
    # Search for posts with media
    logger.info("Searching for posts with media:")
    media_results = await nlq_search(
        query="open source",
        collection_name=COLLECTION_NAME,
        filter={
            "must": [
                {"key": "source", "match": {"value": "mastodon"}},
                {"key": "mastodon_has_media", "match": {"value": True}}
            ]
        },
        limit=5,
    )
    
    logger.info(f"Found {len(media_results)} results with media")
    
    # Delete the connector (uncomment to test)
    # logger.info("Deleting the Mastodon connector:")
    # delete_result = await delete_mastodon_connector(connector_id)
    # pprint(delete_result)


if __name__ == "__main__":
    asyncio.run(main())
