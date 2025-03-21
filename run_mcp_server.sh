#!/bin/bash
# Script to run the enhanced MCP server with the correct Python path

# Get the absolute path to the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the site-packages directory
SITE_PACKAGES=$(python -c "import site; print(':'.join(site.getsitepackages()))")

# Print debug info
echo "Setting PYTHONPATH to include:"
echo "  - $SCRIPT_DIR (script directory)"
echo "  - $SITE_PACKAGES (Python site-packages)"

# Set PYTHONPATH to include both the script directory and site-packages
export PYTHONPATH="$SCRIPT_DIR:$SITE_PACKAGES:$PYTHONPATH"

# Run the launcher script
python "$SCRIPT_DIR/run_mcp_server.py"
