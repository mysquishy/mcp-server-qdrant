"""Cluster visualization tool for Qdrant MCP server."""
import logging
import json
import base64
from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
from io import BytesIO

from mcp.server.fastmcp import Context, Image

logger = logging.getLogger(__name__)

# Default colors for clusters
CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#ad494a"
]

async def cluster_visualization(
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
    """
    Visualize semantic clusters within a Qdrant collection.
    
    This tool applies clustering algorithms to vectors in a collection,
    visualizes the clusters, and optionally extracts representative topics.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    collection : str
        Name of the Qdrant collection to analyze
    n_clusters : int, optional
        Number of clusters (for KMeans, required if method='kmeans')
    text_field : str, optional
        Field containing text for topic extraction (default: "text")
    method : str, optional
        Clustering method: "hdbscan", "kmeans", or "dbscan" (default: "hdbscan")
    min_cluster_size : int, optional
        Minimum cluster size for HDBSCAN (default: 5)
    limit : int, optional
        Maximum number of points to analyze (default: 5000)
    filter : Dict[str, Any], optional
        Filter to apply when retrieving points
    width : int, optional
        Width of the visualization (default: 900)
    height : int, optional
        Height of the visualization (default: 700)
    title : str, optional
        Title for the visualization (default: "Semantic Clusters")
    label_clusters : bool, optional
        Whether to label clusters with topic keywords (default: True)
    extract_topics : bool, optional
        Whether to extract topics from clusters (default: True)
    topic_n_words : int, optional
        Number of keywords to extract per topic (default: 5)
    include_raw_data : bool, optional
        Include raw data in the response (default: False)
        
    Returns:
    --------
    Image or Dict[str, Any]
        Visual representation of clusters as an image, 
        with optional raw data and topic information
    """
    await ctx.info(f"Visualizing semantic clusters in collection '{collection}'")
    
    # Validate parameters
    if method == "kmeans" and not n_clusters:
        raise ValueError("n_clusters must be specified when using KMeans")
    
    if method not in ("hdbscan", "kmeans", "dbscan"):
        raise ValueError("method must be one of: hdbscan, kmeans, dbscan")
    
    # Import necessary libraries
    try:
        import numpy as np
        import plotly.graph_objects as go
        from qdrant_client import QdrantClient
        import umap
        
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
        
        # Retrieve points from Qdrant
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
        await ctx.report_progress(20, 100)
        
        # Extract vectors and text
        vectors = []
        texts = []
        ids = []
        
        for point in points:
            if point.vector is not None:
                vectors.append(point.vector)
                
                # Extract text for topic modeling
                if text_field in point.payload:
                    # Handle nested fields
                    field_parts = text_field.split(".")
                    text_value = point.payload
                    for part in field_parts:
                        if isinstance(text_value, dict) and part in text_value:
                            text_value = text_value[part]
                        else:
                            text_value = ""
                            break
                    
                    texts.append(str(text_value))
                else:
                    texts.append("")
                
                ids.append(point.id)
        
        vectors = np.array(vectors)
        
        if vectors.shape[0] == 0:
            await ctx.warning("No valid vectors found in the retrieved points")
            return {"status": "error", "message": "No valid vectors found"}
        
        await ctx.debug(f"Processing {vectors.shape[0]} vectors")
        
        # Dimension reduction for visualization
        await ctx.debug("Performing UMAP dimension reduction")
        await ctx.report_progress(30, 100)
        
        # 2D projection for visualization
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = umap_reducer.fit_transform(vectors)
        
        # Perform clustering
        await ctx.debug(f"Performing clustering with method: {method}")
        await ctx.report_progress(50, 100)
        
        if method == "hdbscan":
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    gen_min_span_tree=True,
                    prediction_data=True
                )
                cluster_labels = clusterer.fit_predict(vectors)
            except ImportError:
                await ctx.warning("HDBSCAN not installed, falling back to KMeans")
                from sklearn.cluster import KMeans
                n_clusters = n_clusters or min(20, int(np.sqrt(vectors.shape[0] / 2)))
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = clusterer.fit_predict(vectors)
        
        elif method == "kmeans":
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(vectors)
        
        elif method == "dbscan":
            from sklearn.cluster import DBSCAN
            # Estimate eps based on data
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min_cluster_size)
            nn.fit(vectors)
            distances, _ = nn.kneighbors(vectors)
            eps = np.percentile(distances[:, -1], 90)  # 90th percentile of distances
            
            clusterer = DBSCAN(eps=eps, min_samples=min_cluster_size)
            cluster_labels = clusterer.fit_predict(vectors)
        
        # Count clusters, excluding noise points (label -1)
        unique_clusters = np.unique(cluster_labels)
        n_clusters_found = len([c for c in unique_clusters if c >= 0])
        
        await ctx.debug(f"Found {n_clusters_found} clusters")
        await ctx.report_progress(70, 100)
        
        # Extract topics if requested
        topics = {}
        if extract_topics and texts:
            try:
                await ctx.debug("Extracting topics from clusters")
                
                # Import text processing modules
                from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
                import nltk
                
                try:
                    nltk.data.find('corpora/stopwords')
                except (LookupError, ImportError):
                    await ctx.debug("Downloading NLTK stopwords")
                    nltk.download('stopwords', quiet=True)
                
                from nltk.corpus import stopwords
                
                # Get stopwords
                stop_words = set(stopwords.words('english'))
                
                # Create vectorizer for topic extraction
                vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words=stop_words,
                    max_df=0.8,
                    min_df=3
                )
                
                # Only vectorize if we have enough texts
                if len(texts) >= 3:
                    # Fit vectorizer on all texts
                    try:
                        X = vectorizer.fit_transform(texts)
                        feature_names = np.array(vectorizer.get_feature_names_out())
                        
                        # Extract topics for each cluster
                        for cluster_id in unique_clusters:
                            if cluster_id >= 0:  # Skip noise cluster
                                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                                
                                if len(cluster_indices) >= 3:  # Need at least a few docs for meaningful topics
                                    cluster_X = X[cluster_indices]
                                    
                                    # Sum TF-IDF scores across documents in the cluster
                                    cluster_sum = np.asarray(cluster_X.sum(axis=0)).flatten()
                                    
                                    # Get top words
                                    top_indices = cluster_sum.argsort()[-topic_n_words:][::-1]
                                    top_words = feature_names[top_indices]
                                    
                                    topics[int(cluster_id)] = top_words.tolist()
                    except Exception as e:
                        await ctx.warning(f"Error in topic extraction: {str(e)}")
            
            except Exception as e:
                await ctx.warning(f"Could not extract topics: {str(e)}")
        
        # Create visualization
        await ctx.debug("Creating cluster visualization")
        await ctx.report_progress(80, 100)
        
        # Create a colormap for clusters
        n_colors_needed = max(1, n_clusters_found)
        colors = CLUSTER_COLORS[:n_colors_needed]
        if n_colors_needed > len(CLUSTER_COLORS):
            # Generate additional colors if needed
            import colorsys
            for i in range(len(CLUSTER_COLORS), n_colors_needed):
                h = (i * 0.618033988749895) % 1.0  # Golden ratio method
                rgb = colorsys.hsv_to_rgb(h, 0.8, 0.95)
                colors.append(f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})")
        
        # Create the plot
        fig = go.Figure()
        
        # Add traces for each cluster
        cluster_points = {}
        for cluster_id in unique_clusters:
            cluster_idx = np.where(cluster_labels == cluster_id)[0]
            cluster_points[int(cluster_id)] = cluster_idx.tolist()
            
            # Determine cluster name/label
            if cluster_id == -1:
                cluster_name = "Noise"
                cluster_color = "#7f7f7f"  # Gray for noise
            else:
                if label_clusters and cluster_id in topics:
                    topic_words = topics[cluster_id]
                    cluster_name = f"Cluster {cluster_id}: {', '.join(topic_words)}"
                else:
                    cluster_name = f"Cluster {cluster_id}"
                
                cluster_color = colors[cluster_id % len(colors)]
            
            # Add scatter plot for this cluster
            fig.add_trace(go.Scatter(
                x=embedding[cluster_idx, 0],
                y=embedding[cluster_idx, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=cluster_color,
                    opacity=0.7,
                    line=dict(width=0.5, color='DarkSlateGrey')
                ),
                text=[f"ID: {ids[i]}<br>Text: {texts[i][:100]}..." if len(texts[i]) > 100 else f"ID: {ids[i]}<br>Text: {texts[i]}" for i in cluster_idx],
                name=cluster_name
            ))
            
            # Add cluster label if we have topics
            if label_clusters and cluster_id in topics and cluster_id >= 0:
                # Find centroid of this cluster
                centroid_x = np.mean(embedding[cluster_idx, 0])
                centroid_y = np.mean(embedding[cluster_idx, 1])
                
                # Add text annotation
                fig.add_trace(go.Scatter(
                    x=[centroid_x],
                    y=[centroid_y],
                    mode='text',
                    text=[f"Cluster {cluster_id}"],
                    textfont=dict(
                        size=12,
                        color='black',
                        family="Arial, sans-serif",
                        weight="bold"
                    ),
                    hoverinfo='none',
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            showlegend=True,
            legend=dict(
                itemsizing='constant',
                font=dict(size=10),
                title=f"Clusters ({n_clusters_found})"
            ),
            margin=dict(l=10, r=10, b=10, t=40),
            hovermode='closest',
            plot_bgcolor='rgb(250,250,250)',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=""
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=""
            )
        )
        
        # Add cluster statistics annotation
        stats_text = f"Found {n_clusters_found} clusters in {vectors.shape[0]} points"
        fig.add_annotation(
            x=0.01,
            y=0.01,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
            borderpad=4,
            align="left"
        )
        
        # Convert plot to image
        await ctx.report_progress(90, 100)
        img_bytes = fig.to_image(format="png")
        
        # Prepare result
        result = {}
        if include_raw_data:
            result = {
                "status": "success",
                "n_clusters": n_clusters_found,
                "n_points": vectors.shape[0],
                "clusters": {
                    str(cluster_id): {
                        "size": len(cluster_points[cluster_id]),
                        "points": cluster_points[cluster_id],
                        "topics": topics.get(cluster_id, []) if cluster_id >= 0 else []
                    } for cluster_id in unique_clusters
                },
                "timestamp": asyncio.get_event_loop().time()
            }
        
        # Return as an Image object
        await ctx.info(f"Cluster visualization completed with {n_clusters_found} clusters")
        await ctx.report_progress(100, 100)
        
        if include_raw_data:
            # If raw data requested, return both
            return {
                "status": "success",
                "image": Image(data=img_bytes, format="png"),
                "data": result
            }
        else:
            # Otherwise just return the image
            return Image(data=img_bytes, format="png")
        
    except Exception as e:
        error_msg = f"Error generating cluster visualization: {str(e)}"
        logger.error(error_msg, exc_info=True)
        await ctx.error(error_msg)
        return {"status": "error", "message": error_msg}
