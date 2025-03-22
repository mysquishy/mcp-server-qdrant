#!/bin/bash

set -e

echo "====== Completing Social Media Connector Integration ======"

# 1. Install dependencies
echo -e "\n\n=> Installing dependencies for social media connectors..."
pip install -e ".[social]"

# 2. Check if tools are properly imported
echo -e "\n\n=> Verifying that connectors are properly imported..."
if python -c "from mcp_server_qdrant.tools.connectors import setup_twitter_connector, setup_mastodon_connector"; then
    echo "✅ Connectors successfully imported"
else
    echo "❌ Error importing connectors"
    exit 1
fi

# 3. Run a test example
echo -e "\n\n=> Testing social media connector functionality..."
if [ -f "examples/twitter_example.py" ]; then
    echo "Example file exists: examples/twitter_example.py"
    echo "To run the example, execute: python examples/twitter_example.py"
else
    echo "Example file not found: examples/twitter_example.py"
    echo "Check in the /examples directory for available examples."
fi

# 4. Output success message with usage instructions
echo -e "\n\n====== Social Media Connector Integration Complete! ======"
echo ""
echo "Usage Instructions:"
echo "-------------------"
echo "1. Start the Qdrant server:"
echo "   docker run -p 6333:6333 qdrant/qdrant"
echo ""
echo "2. Start the MCP server with social connectors:"
echo "   python src/mcp_server_qdrant/server.py --qdrant-url http://localhost:6333 --collection-name social_media"
echo ""
echo "Twitter Connector Example:"
echo "-------------------------"
echo "await setup_twitter_connector("
echo "    username=\"anthropic\","
echo "    collection_name=\"social_media\","
echo "    include_retweets=True,"
echo "    include_replies=False,"
echo "    fetch_limit=100,"
echo "    bearer_token=\"your_bearer_token\","  # Optional
echo "    update_interval_minutes=30,"
echo ")"
echo ""
echo "Mastodon Connector Example:"
echo "--------------------------"
echo "await setup_mastodon_connector("
echo "    account=\"Gargron\","
echo "    instance_url=\"https://mastodon.social\","
echo "    collection_name=\"social_media\","
echo "    include_boosts=True,"
echo "    include_replies=False,"
echo "    fetch_limit=100,"
echo "    api_access_token=\"your_access_token\","  # Optional
echo "    update_interval_minutes=30,"
echo ")"
echo ""
echo "For more information, see the documentation at:"
echo "/docs/social-connectors.md"
