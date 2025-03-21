"""Vector visualization tool for Qdrant MCP server."""
import logging
import json
import base64
from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
from io import BytesIO

from mcp.server.fastmcp import Context, Image

logger = logging.getLogger(__name__)

# Default colors for different categories
DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

async def visualize_vectors(
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
    """
    Generate 2D or 3D projections of vectors for visualization.
    
    This tool retrieves vectors from a Qdrant collection and creates
    a visual representation using dimensionality reduction techniques.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    collection : str
        Name of the Qdrant collection containing the vectors
    label_field : str, optional
        Field in the payload to use for point labels (default: "text")
    limit : int, optional
        Maximum number of points to visualize (default: 1000)
    method : str, optional
        Dimensionality reduction method: "umap", "tsne", or "pca" (default: "umap")
    dimensions : int, optional
        Number of dimensions for the projection: 2 or 3 (default: 2)
    filter : Dict[str, Any], optional
        Filter to apply when retrieving points
    category_field : str, optional
        Field in the payload to use for categorizing points
    custom_colors : Dict[str, str], optional
        Custom color mapping for categories
    width : int, optional
        Width of the visualization in pixels (default: 800)
    height : int, optional
        Height of the visualization in pixels (default: 600)
    title : str, optional
        Title for the visualization (default: "Vector Visualization")
        
    Returns:
    --------
    Image or Dict[str, Any]
        Visual representation of the vectors as an image or plot data
    """
    await ctx.info(f"Generating {dimensions}D vector visualization using {method}")
    
    # Validate parameters
    if dimensions not in (2, 3):
        raise ValueError("Dimensions must be either 2 or 3")
    
    if method not in ("umap", "tsne", "pca"):
        raise ValueError("Method must be one of: umap, tsne, pca")
    
    # Import necessary libraries
    try:
        import numpy as np
        import plotly.graph_objects as go
        from qdrant_client import QdrantClient
        
        # Get Qdrant client from the context
        client = ctx.request_context.lifespan_context.get("qdrant_client")
        if client is None:
            await ctx.warning("Qdrant client not found in context, creating a new one")
            from qdrant_client import QdrantClient
            
            # Get Qdrant settings from context
            settings = ctx.request_context.lifespan_context.get("settings")
            if settings and settings.qdrant_url:
                client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
            elif settings and settings.qdrant_local_path:
                client = QdrantClient(path=settings.qdrant_local_path)
            else:
                raise ValueError("Qdrant connection settings not found")
        
        # First, retrieve the points from Qdrant
        await ctx.debug(f"Retrieving up to {limit} points from collection '{collection}'")
        
        scroll_result = client.scroll(
            collection_name=collection,
            limit=limit,
            with_payload=True,
            with_vectors=True,
            filter=filter
        )
        
        points = scroll_result[0]  # First element contains the points
        
        if not points:
            await ctx.warning(f"No points found in collection '{collection}'")
            return {"status": "error", "message": "No points found in the collection"}
        
        await ctx.debug(f"Retrieved {len(points)} points")
        await ctx.report_progress(25, 100)
        
        # Extract vectors
        vectors = np.array([point.vector for point in points if point.vector is not None])
        
        if vectors.size == 0:
            await ctx.warning("No valid vectors found in the retrieved points")
            return {"status": "error", "message": "No valid vectors found"}
        
        # Extract labels and categories
        labels = []
        categories = []
        for point in points:
            # Extract label
            if label_field and point.payload and label_field in point.payload:
                # If label_field contains nested fields (e.g., "metadata.title")
                label_parts = label_field.split(".")
                label_value = point.payload
                for part in label_parts:
                    if isinstance(label_value, dict) and part in label_value:
                        label_value = label_value[part]
                    else:
                        label_value = str(point.id)
                        break
                
                # Truncate long labels
                if isinstance(label_value, str) and len(label_value) > 50:
                    label_value = label_value[:47] + "..."
                
                labels.append(str(label_value))
            else:
                labels.append(str(point.id))
            
            # Extract category
            if category_field and point.payload and category_field in point.payload:
                # Handle nested category fields
                category_parts = category_field.split(".")
                category_value = point.payload
                for part in category_parts:
                    if isinstance(category_value, dict) and part in category_value:
                        category_value = category_value[part]
                    else:
                        category_value = "unknown"
                        break
                
                categories.append(str(category_value))
            else:
                categories.append("default")
        
        # Perform dimensionality reduction
        await ctx.debug(f"Performing {method} dimensionality reduction to {dimensions}D")
        await ctx.report_progress(50, 100)
        
        if method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=dimensions, random_state=42)
                embedding = reducer.fit_transform(vectors)
            except ImportError:
                await ctx.warning("UMAP not installed, falling back to PCA")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=dimensions)
                embedding = reducer.fit_transform(vectors)
        
        elif method == "tsne":
            try:
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=dimensions, random_state=42)
                embedding = reducer.fit_transform(vectors)
            except ImportError:
                await ctx.warning("TSNE not installed, falling back to PCA")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=dimensions)
                embedding = reducer.fit_transform(vectors)
        
        elif method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=dimensions)
            embedding = reducer.fit_transform(vectors)
        
        await ctx.debug("Dimensionality reduction completed")
        await ctx.report_progress(75, 100)
        
        # Create color mapping for categories
        unique_categories = list(set(categories))
        if custom_colors:
            color_map = {cat: custom_colors.get(cat, DEFAULT_COLORS[i % len(DEFAULT_COLORS)]) 
                        for i, cat in enumerate(unique_categories)}
        else:
            color_map = {cat: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] 
                        for i, cat in enumerate(unique_categories)}
        
        # Generate colors for each point
        colors = [color_map[cat] for cat in categories]
        
        # Create the visualization
        await ctx.debug(f"Creating {dimensions}D visualization")
        
        if dimensions == 2:
            # 2D plot
            fig = go.Figure()
            
            # Create scatter traces for each category
            for category in unique_categories:
                indices = [i for i, cat in enumerate(categories) if cat == category]
                
                fig.add_trace(go.Scatter(
                    x=embedding[indices, 0],
                    y=embedding[indices, 1],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=color_map[category],
                        opacity=0.7,
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    text=[labels[i] for i in indices],
                    name=category
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                width=width,
                height=height,
                showlegend=True,
                legend=dict(itemsizing='constant', font=dict(size=10)),
                margin=dict(l=0, r=0, b=0, t=40),
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        
        else:  # 3D plot
            # Create a single 3D scatter plot
            fig = go.Figure()
            
            for category in unique_categories:
                indices = [i for i, cat in enumerate(categories) if cat == category]
                
                fig.add_trace(go.Scatter3d(
                    x=embedding[indices, 0],
                    y=embedding[indices, 1],
                    z=embedding[indices, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=color_map[category],
                        opacity=0.7,
                        line=dict(width=0.5, color='DarkSlateGrey')
                    ),
                    text=[labels[i] for i in indices],
                    name=category
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                width=width,
                height=height,
                showlegend=True,
                legend=dict(itemsizing='constant', font=dict(size=10)),
                margin=dict(l=0, r=0, b=0, t=40),
                scene=dict(
                    xaxis=dict(showticklabels=False, title=''),
                    yaxis=dict(showticklabels=False, title=''),
                    zaxis=dict(showticklabels=False, title='')
                )
            )
        
        # Convert plot to image
        await ctx.report_progress(90, 100)
        img_bytes = fig.to_image(format="png")
        
        # Return as an Image object
        await ctx.info("Vector visualization completed")
        await ctx.report_progress(100, 100)
        
        return Image(data=img_bytes, format="png")
        
    except Exception as e:
        error_msg = f"Error generating vector visualization: {str(e)}"
        logger.error(error_msg, exc_info=True)
        await ctx.error(error_msg)
        return {"status": "error", "message": error_msg}
