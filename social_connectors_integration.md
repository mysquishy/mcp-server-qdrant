# Social Media Connectors Integration

This document summarizes the integration of social media connectors into the main Qdrant MCP Server project.

## Files Migrated

The following files were migrated from the incorrect location to the proper project directory:

1. **Twitter Connector**: 
   - `/Users/squishy64/mcp-server-qdrant/src/mcp_server_qdrant/tools/connectors/twitter_connector.py`

2. **Mastodon Connector**: 
   - `/Users/squishy64/mcp-server-qdrant/src/mcp_server_qdrant/tools/connectors/mastodon_connector.py`

3. **Connectors Package**:
   - `/Users/squishy64/mcp-server-qdrant/src/mcp_server_qdrant/tools/connectors/__init__.py`

4. **Documentation**:
   - `/Users/squishy64/mcp-server-qdrant/docs/social-connectors.md`

5. **Example Files**:
   - `/Users/squishy64/mcp-server-qdrant/examples/combined_social_search.py`
   - `/Users/squishy64/mcp-server-qdrant/examples/twitter_example.py`
   - `/Users/squishy64/mcp-server-qdrant/examples/mastodon_example.py`

## Integration Steps

1. **Create Required Directories**:
   - Created the `/tools/connectors` directory structure
   - Created the proper directories for examples and documentation

2. **Migrate Implementation Files**:
   - Moved the Twitter connector implementation to the proper location
   - Moved the Mastodon connector implementation to the proper location
   - Created an appropriate `__init__.py` to expose the connector functions

3. **Migrate Documentation and Examples**:
   - Moved the documentation to the `docs` directory
   - Moved the example files to the `examples` directory

4. **Update Dependencies**:
   - Added a new `social` optional dependency group in `pyproject.toml`
   - Added required dependencies like `httpx`, `beautifulsoup4`, and `mastodon.py`

5. **Create Integration Patch**:
   - Created `integration_patch.diff` showing changes needed to integrate with `server.py`
   - Added tool descriptions in `settings.py`
   - Added tool registrations in `server.py`

## Tools Provided

The integration adds the following tools to the Qdrant MCP Server:

### Twitter Connector Tools

1. **setup-twitter-connector**: Set up a Twitter connector to automatically fetch and index tweets
2. **check-twitter-updates**: Manually check for new tweets from a configured connector
3. **list-twitter-connectors**: List all active Twitter connectors
4. **delete-twitter-connector**: Delete a Twitter connector

### Mastodon Connector Tools

1. **setup-mastodon-connector**: Set up a Mastodon connector to automatically fetch and index posts
2. **check-mastodon-updates**: Manually check for new posts from a configured connector
3. **list-mastodon-connectors**: List all active Mastodon connectors
4. **delete-mastodon-connector**: Delete a Mastodon connector

## Installation

To use the social media connectors, install the package with the social media dependencies:

```bash
pip install "mcp-server-qdrant[social]"
```

Or to install all optional dependencies:

```bash
pip install "mcp-server-qdrant[all]"
```

## Usage Examples

See the example files in the `examples` directory for detailed usage examples:

- `twitter_example.py`: Example of using the Twitter connector
- `mastodon_example.py`: Example of using the Mastodon connector
- `combined_social_search.py`: Example of using both connectors together

## Final Integration Steps

To complete the integration, apply the changes in `integration_patch.diff` to the following files:

1. Update `src/mcp_server_qdrant/settings.py` to add tool descriptions
2. Update `src/mcp_server_qdrant/server.py` to import and register the tools

This can be done manually by editing these files, or by using the patch command:

```bash
patch -p1 < integration_patch.diff
```
