# Social Media Connectors Integration - Next Session

## Overview

We have successfully migrated and integrated the social media connectors from the incorrect location (`/Users/squishy64/mcp_server_qdrant`) to the proper project directory structure (`/Users/squishy64/mcp-server-qdrant`). This integration allows the Qdrant MCP server to automatically fetch, process, and index content from social media platforms for semantic search and analysis.

## Completed Tasks

1. **Implementation Files Migration**:
   - Migrated Twitter connector to `/src/mcp_server_qdrant/tools/connectors/twitter_connector.py`
   - Migrated Mastodon connector to `/src/mcp_server_qdrant/tools/connectors/mastodon_connector.py`
   - Created proper `__init__.py` to expose connector functions

2. **Documentation and Examples**:
   - Added documentation to `/docs/social-connectors.md`
   - Migrated example files to `/examples/` directory:
     - `twitter_example.py`
     - `mastodon_example.py`
     - `combined_social_search.py`

3. **Dependency Management**:
   - Added new `social` optional dependency group in `pyproject.toml`
   - Added required dependencies: `httpx`, `beautifulsoup4`, `mastodon.py`

4. **Integration Patch**:
   - Created `integration_patch.diff` with changes needed for server integration
   - Added tool descriptions in `settings.py`
   - Added tool registrations in `server.py`

5. **Integration Documentation**:
   - Created a comprehensive summary document: `social_connectors_integration.md`
   - Documented the changes made and steps to complete integration

## Features Added

The integration adds the following capabilities to the Qdrant MCP Server:

### Twitter Connector
- Automatic fetching and indexing of tweets from specific Twitter users
- Support for API-based access with fallback to web scraping
- Options to include/exclude retweets and replies
- Automatic periodic updates with configurable intervals
- Comprehensive metadata extraction and storage

### Mastodon Connector
- Automatic fetching and indexing of posts from Mastodon accounts
- Support for any Mastodon instance with cross-instance compatibility
- Options to include/exclude boosts and replies
- API access with web scraping fallback
- Full metadata storage for filtering and retrieval

## Next Steps

To complete the integration, follow these final steps:

1. **Apply Integration Patch**:
   ```bash
   patch -p1 < integration_patch.diff
   ```
   Or manually apply the changes to:
   - `src/mcp_server_qdrant/settings.py`
   - `src/mcp_server_qdrant/server.py`

2. **Install Dependencies**:
   ```bash
   pip install -e ".[social]"  # For social media connectors only
   pip install -e ".[all]"     # For all optional dependencies
   ```

3. **Test Integration**:
   ```bash
   # Start Qdrant server
   docker run -p 6333:6333 qdrant/qdrant

   # Run the MCP server
   python -m mcp_server_qdrant.server --qdrant-url http://localhost:6333 --collection-name social_media

   # In another terminal, run one of the examples
   python examples/twitter_example.py
   ```

4. **Verify Tools in Claude Desktop**:
   - Check that social media connector tools are available in Claude Desktop
   - Test the tools with a sample query

## Tool Usage Examples

### Twitter Connector
```python
await setup_twitter_connector(
    username="anthropic",
    collection_name="social_media",
    include_retweets=True,
    include_replies=False,
    fetch_limit=100,
    bearer_token="your_bearer_token",  # Optional
    update_interval_minutes=30,
)
```

### Mastodon Connector
```python
await setup_mastodon_connector(
    account="Gargron",
    instance_url="https://mastodon.social",
    collection_name="social_media",
    include_boosts=True,
    include_replies=False,
    fetch_limit=100,
    api_access_token="your_access_token",  # Optional
    update_interval_minutes=30,
)
```

### Searching Across Social Media
```python
# Search across all social media sources
results = await nlq_search(
    query="AI ethics",
    collection_name="social_media",
    filter=None,  # No filter means search all sources
    limit=10,
)

# Search only Twitter content
twitter_results = await nlq_search(
    query="AI ethics",
    collection_name="social_media",
    filter={
        "must": [
            {"key": "source", "match": {"value": "twitter"}}
        ]
    },
    limit=5,
)
```