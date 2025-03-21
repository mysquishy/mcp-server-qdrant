#!/bin/bash
# Script to run the enhanced MCP server with the correct Python path

# Get the site-packages directory
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# Print debug info
echo "Setting PYTHONPATH to include: $SITE_PACKAGES"

# Run the server with PYTHONPATH set
PYTHONPATH="$SITE_PACKAGES:$PYTHONPATH" python enhanced_mcp_server.py
