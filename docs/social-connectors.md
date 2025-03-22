# Real-Time Social Media Connectors for Qdrant MCP Server

This extension to the Qdrant MCP server provides real-time data connectors for automatically fetching, processing, and indexing content from social media platforms for semantic search and analysis.

## Available Connectors

- **Twitter Connector**: Automatically fetch and index tweets from specific Twitter users
- **Mastodon Connector**: Automatically fetch and index posts from specific Mastodon accounts

## Features

- **Graceful Degradation**: Each connector works with or without API credentials
- **Automatic Updates**: Periodic checking for new content based on configurable intervals
- **Rich Metadata**: Comprehensive metadata storage for advanced filtering and retrieval
- **Media Support**: Handling of images, videos, and other media attachments
- **Content Filtering**: Options to include or exclude specific content types (retweets, replies, etc.)
- **Error Handling**: Robust error handling with graceful recovery for temporary issues

## Installation

The social media connectors are included in the main Qdrant MCP server package. To use all features, install with the optional dependencies:

```bash
# Install with all optional dependencies
pip install "mcp-server-qdrant[mastodon]"
```

## Twitter Connector

The Twitter connector allows you to automatically fetch and index tweets from specific Twitter accounts.

### Setting Up a Twitter Connector

```python
# Example using the Twitter connector
result = await setup_twitter_connector(
    username="anthropic",
    collection_name="social_media",
    include_retweets=True,
    include_replies=False,
    fetch_limit=100,
    # Optional API credentials (if available)
    bearer_token="your_bearer_token",
    update_interval_minutes=30,
)

# Get the connector ID for future reference
connector_id = result["id"]
```

### Twitter Connector Authentication Options

The Twitter connector supports three authentication methods, in order of preference:

1. **Bearer Token**: Simplest method, pass a valid bearer token
2. **API Key & Secret**: Authenticate with API key and secret pair
3. **No Authentication**: Falls back to web scraping using Nitter (limited capabilities)

### Managing Twitter Connectors

```python
# List all active Twitter connectors
connectors = await list_twitter_connectors()

# Manually check for new tweets
update_result = await check_twitter_updates(connector_id)

# Delete a Twitter connector
delete_result = await delete_twitter_connector(connector_id)
```

## Mastodon Connector

The Mastodon connector allows you to automatically fetch and index posts from specific Mastodon accounts across different instances.

### Setting Up a Mastodon Connector

```python
# Example using the Mastodon connector
result = await setup_mastodon_connector(
    account="anthropic",
    instance_url="https://mastodon.social",
    collection_name="social_media",
    include_boosts=True,
    include_replies=False,
    fetch_limit=100,
    # Optional API access token (if available)
    api_access_token="your_access_token",
    update_interval_minutes=30,
)

# Get the connector ID for future reference
connector_id = result["id"]
```

### Mastodon Connector Authentication Options

The Mastodon connector supports two authentication methods:

1. **API Access Token**: Use an access token from the Mastodon instance
2. **No Authentication**: Falls back to web scraping of public profiles (limited capabilities)

### Managing Mastodon Connectors

```python
# List all active Mastodon connectors
connectors = await list_mastodon_connectors()

# Manually check for new posts
update_result = await check_mastodon_updates(connector_id)

# Delete a Mastodon connector
delete_result = await delete_mastodon_connector(connector_id)
```

## Combined Tools

The enhanced MCP server also provides tools for working with multiple social media connectors together:

```python
# List all social media connectors of all types
all_connectors = await list_all_social_connectors()

# Search across all social media content
search_results = await search_social_content(
    query="AI safety",
    collection_name="social_media",
    source_types=["twitter", "mastodon"],
    limit=20,
)
```

## Payload Schema

### Twitter Payload Schema

The Twitter connector stores the following metadata in Qdrant for each tweet:

```json
{
  "twitter_id": "1234567890",
  "twitter_text": "Tweet content here",
  "twitter_created_at": "2023-05-25T12:34:56Z",
  "twitter_author": "username",
  "twitter_is_retweet": false,
  "twitter_is_reply": false,
  "twitter_has_media": true,
  "twitter_retweet_count": 42,
  "twitter_reply_count": 7,
  "twitter_like_count": 123,
  "twitter_media_types": ["photo", "video"],
  "twitter_media_urls": ["https://example.com/image.jpg"],
  "source": "twitter",
  "source_type": "social_media",
  "content_type": "tweet"
}
```

### Mastodon Payload Schema

The Mastodon connector stores the following metadata in Qdrant for each post:

```json
{
  "mastodon_id": "1234567890",
  "mastodon_text": "Post content here",
  "mastodon_created_at": "2023-05-25T12:34:56Z",
  "mastodon_author": "username",
  "mastodon_instance": "mastodon.social",
  "mastodon_is_boost": false,
  "mastodon_is_reply": false,
  "mastodon_has_media": true,
  "mastodon_has_cw": false,
  "mastodon_visibility": "public",
  "mastodon_url": "https://mastodon.social/@user/1234567890",
  "mastodon_replies_count": 7,
  "mastodon_reblogs_count": 42,
  "mastodon_favourites_count": 123,
  "mastodon_media_types": ["image", "video"],
  "mastodon_media_urls": ["https://example.com/image.jpg"],
  "mastodon_html": "<p>Original HTML content</p>",
  "source": "mastodon",
  "source_type": "social_media",
  "content_type": "post"
}
```

## Advanced Usage

### Custom Filtering

You can use the rich metadata to build custom filters when searching:

```python
from mcp_server_qdrant.search import nlq_search

# Search for tweets with media that mention AI and have at least 10 likes
results = await nlq_search(
    query="AI",
    collection="social_media",
    filter={
        "must": [
            {"key": "source", "match": {"value": "twitter"}},
            {"key": "twitter_has_media", "match": {"value": True}},
            {"key": "twitter_like_count", "range": {"gte": 10}},
        ]
    },
    limit=20,
)
```

### Combining with Other Data Sources

You can use the social media connectors alongside other data sources in the same collection:

```python
# Setup connectors for different sources to the same collection
await setup_twitter_connector(
    username="anthropic",
    collection_name="knowledge_base",
)

await setup_mastodon_connector(
    account="anthropic",
    instance_url="https://mastodon.social",
    collection_name="knowledge_base",
)

# You can also add documents, webpages, etc. to the same collection
# ...

# Then search across all sources
results = await search_social_content(
    query="Latest AI research",
    collection_name="knowledge_base",
)
```

## Error Handling

The connectors are designed to handle various error conditions gracefully:

- **API Rate Limits**: Temporary pause with automatic retry
- **Network Issues**: Automatic retry with exponential backoff
- **API Changes**: Fallback to web scraping when API fails
- **Deleted Content**: Skips over deleted or unavailable content

## Dependencies

- **httpx**: For async HTTP requests
- **beautifulsoup4**: For web scraping when API access is unavailable
- **mastodon.py**: Optional dependency for Mastodon API access

## Notes

- The in-memory storage for connectors is suitable for development but should be replaced with persistent storage in production
- Web scraping methods are provided as fallbacks and might be affected by changes to the websites' structure
- Consider implementing a rate limiting mechanism when working with multiple connectors to avoid API limits

## Future Improvements

Planned enhancements for future releases:

1. **RSS Feed Connector**: Support for RSS/Atom feeds
2. **Persistent Storage**: Database-backed connector configuration
3. **Content Filtering**: Advanced filtering based on keywords or sentiment
4. **Real-time Notifications**: Webhooks for new content alerts
5. **Custom Embedding Models**: Support for domain-specific embedding models
