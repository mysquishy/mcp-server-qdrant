"""
Social media connectors for the MCP Server.

This package provides connectors for integrating with various social media platforms,
allowing automated fetching, processing, and indexing of social media content.
"""

# Make connectors available at package level
from .mastodon_connector import (
    setup_mastodon_connector,
    check_mastodon_updates,
    list_mastodon_connectors,
    delete_mastodon_connector,
)

from .twitter_connector import (
    setup_twitter_connector,
    check_twitter_updates,
    list_twitter_connectors,
    delete_twitter_connector,
)

from .combined_social_search import (
    list_all_social_connectors,
    search_social_content,
)

__all__ = [
    # Mastodon connector
    "setup_mastodon_connector",
    "check_mastodon_updates",
    "list_mastodon_connectors",
    "delete_mastodon_connector",
    
    # Twitter connector
    "setup_twitter_connector",
    "check_twitter_updates",
    "list_twitter_connectors",
    "delete_twitter_connector",
    
    # Combined social search
    "list_all_social_connectors",
    "search_social_content",
]

