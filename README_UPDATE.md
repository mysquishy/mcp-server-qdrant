# Real-Time Data Connectors Update

This update adds initial support for real-time data connectors to the Qdrant MCP Server. These connectors enable automatic collection and indexing of content from external sources.

## Files Added

1. `/real-time-connectors-progress.md` - A detailed documentation of implementation progress and plans
2. `/src/mcp_server_qdrant/tools/connectors/twitter_connector.py` - Initial Twitter connector implementation (stub)

## Committing Changes

To commit these changes to your repository, run:

```bash
bash commit_real_time_connectors.sh
```

## Next Steps

In the next development session, we'll:

1. Complete the Twitter connector implementation
2. Create the Mastodon connector
3. Register both connectors in the MCP server

Refer to `real-time-connectors-progress.md` for detailed implementation plans.
