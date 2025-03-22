#!/bin/bash

# Commit the real-time connectors implementation progress
echo "Adding files to git..."
git add src/mcp_server_qdrant/tools/connectors/twitter_connector.py real-time-connectors-progress.md

# Commit the changes
git commit -m "Add Twitter connector stub and implementation progress doc"

echo "Changes committed successfully. You can now push them to your repository with 'git push'."
