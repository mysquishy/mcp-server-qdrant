#!/usr/bin/env python
"""
Script to add the missing tools to enhanced_mcp_server.py
"""
import sys
import os
from typing import Dict, Any, List, Optional, Union

# Define the code to add
additions = """
# Import the missing visualization and metadata tools
from mcp_server_qdrant.tools.metadata.extract_metadata import extract_metadata as extract_metadata_impl
from mcp_server_qdrant.tools.visualization.visualize_vectors import visualize_vectors as visualize_vectors_impl
from mcp_server_qdrant.tools.visualization.cluster_visualization import cluster_visualization as cluster_visualization_impl

# Metadata Extraction Tool
@fast_mcp.tool(name="extract_metadata")
async def extract_metadata_tool(
    ctx: Context,
    text: str,
    extract_entities: bool = True,
    extract_patterns: bool = True,
    extract_statistics: bool = True,
    custom_patterns: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    \"\"\"
    Extract structured metadata from document text.
    
    This tool analyzes document content and extracts important metadata 
    such as entities (people, organizations, locations), key patterns 
    (emails, dates, URLs), and document statistics.
    
    Args:
        text: The text to analyze and extract metadata from
        extract_entities: Whether to extract named entities (default: True)
        extract_patterns: Whether to extract common patterns like emails, dates, URLs (default: True)
        extract_statistics: Whether to extract document statistics (default: True)
        custom_patterns: Additional custom regex patterns to extract
        
    Returns:
        Extracted metadata including entities, patterns, and statistics
    \"\"\"
    return await extract_metadata_impl(ctx, text, extract_entities, extract_patterns, extract_statistics, custom_patterns)

# Vector Visualization Tool
@fast_mcp.tool(name="visualize_vectors")
async def visualize_vectors_tool(
    ctx: Context,
    collection: str,
    label_field: Optional[str] = "text",
    limit: int = 1000,
    method: str = "umap",
    dimensions: int = 2,
    filter: Optional[Dict[str, Any]] = None,
    category_field: Optional[str] = None,
    custom_colors: Optional[Dict[str, str]] = None,
    width: int = 800,
    height: int = 600,
    title: str = "Vector Visualization",
) -> Union[Image, Dict[str, Any]]:
    \"\"\"
    Generate 2D or 3D projections of vectors for visualization.
    
    This tool retrieves vectors from a Qdrant collection and creates
    a visual representation using dimensionality reduction techniques.
    
    Args:
        collection: Name of the Qdrant collection containing vectors
        label_field: Field in the payload to use for point labels (default: "text")
        limit: Maximum number of points to visualize (default: 1000)
        method: Dimensionality reduction method: "umap", "tsne", or "pca" (default: "umap")
        dimensions: Number of dimensions for the projection: 2 or 3 (default: 2)
        filter: Filter to apply when retrieving points
        category_field: Field in the payload to use for categorizing points
        custom_colors: Custom color mapping for categories
        width: Width of the visualization in pixels (default: 800)
        height: Height of the visualization in pixels (default: 600)
        title: Title for the visualization (default: "Vector Visualization")
        
    Returns:
        Visual representation of the vectors as an image or plot data
    \"\"\"
    return await visualize_vectors_impl(
        ctx, collection, label_field, limit, method, dimensions, filter, 
        category_field, custom_colors, width, height, title
    )

# Cluster Visualization Tool
@fast_mcp.tool(name="cluster_visualization")
async def cluster_visualization_tool(
    ctx: Context,
    collection: str,
    n_clusters: Optional[int] = None,
    text_field: str = "text",
    method: str = "hdbscan",
    min_cluster_size: int = 5,
    limit: int = 5000,
    filter: Optional[Dict[str, Any]] = None,
    width: int = 900,
    height: int = 700,
    title: str = "Semantic Clusters",
    label_clusters: bool = True,
    extract_topics: bool = True,
    topic_n_words: int = 5,
    include_raw_data: bool = False,
) -> Union[Image, Dict[str, Any]]:
    \"\"\"
    Visualize semantic clusters within a Qdrant collection.
    
    This tool applies clustering algorithms to vectors in a collection,
    visualizes the clusters, and optionally extracts representative topics.
    
    Args:
        collection: Name of the Qdrant collection to analyze
        n_clusters: Number of clusters (for KMeans, required if method='kmeans')
        text_field: Field containing text for topic extraction (default: "text")
        method: Clustering method: "hdbscan", "kmeans", or "dbscan" (default: "hdbscan")
        min_cluster_size: Minimum cluster size for HDBSCAN (default: 5)
        limit: Maximum number of points to analyze (default: 5000)
        filter: Filter to apply when retrieving points
        width: Width of the visualization (default: 900)
        height: Height of the visualization (default: 700)
        title: Title for the visualization (default: "Semantic Clusters")
        label_clusters: Whether to label clusters with topic keywords (default: True)
        extract_topics: Whether to extract topics from clusters (default: True)
        topic_n_words: Number of keywords to extract per topic (default: 5)
        include_raw_data: Include raw data in the response (default: False)
        
    Returns:
        Visual representation of clusters as an image, 
        with optional raw data and topic information
    \"\"\"
    return await cluster_visualization_impl(
        ctx, collection, n_clusters, text_field, method, min_cluster_size, 
        limit, filter, width, height, title, label_clusters, extract_topics, 
        topic_n_words, include_raw_data
    )
"""

# Function to insert the code before a specific line
def insert_code_before_line(file_path, line_marker, code_to_insert):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the line number where we want to insert the code
    marker_index = None
    for i, line in enumerate(lines):
        if line_marker in line:
            marker_index = i
            break
    
    if marker_index is not None:
        # Insert the new code before the marker line
        lines.insert(marker_index, code_to_insert)
        
        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)
        
        print(f"Successfully updated {file_path}")
        return True
    else:
        print(f"Marker line '{line_marker}' not found in {file_path}")
        return False

# Update the enhanced_mcp_server.py file
enhanced_server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'enhanced_mcp_server.py')

# First, add the imports to the top of the file
# Check if the imports are already there
import_added = False
with open(enhanced_server_path, 'r') as file:
    content = file.read()
    if "from mcp_server_qdrant.tools.metadata.extract_metadata" not in content:
        # Add the imports after the standard imports
        insert_code_before_line(enhanced_server_path, "# Define a simple memory storage tool", additions)
        import_added = True

if import_added:
    print("The missing tools have been added to enhanced_mcp_server.py")
    print("You can now restart the server to use the new tools.")
else:
    print("The tools are already in the file or the file structure couldn't be recognized.")
    print("Please check the enhanced_mcp_server.py file manually.")
