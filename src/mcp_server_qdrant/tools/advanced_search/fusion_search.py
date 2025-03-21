"""Fusion search implementation for Qdrant MCP Server.

This tool combines results from multiple embedding models for more robust search.
"""
import os
from typing import Any, Dict, List, Optional, Union
import logging
import json
from collections import defaultdict

from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)

# Import necessary modules for search
from ...embeddings.factory import get_embedding_provider
from ...qdrant import QdrantClient
from ..search.nlq import nlq_search


async def fusion_search(
    query: str,
    collection: Optional[str] = None,
    embedding_models: Optional[List[str]] = None,
    fusion_method: str = "rrf",  # Options: "rrf", "max", "avg"
    rrf_k: int = 60,  # Parameter for RRF fusion
    limit: int = 10,
    filter: Optional[Dict[str, Any]] = None,
    with_payload: Union[bool, List[str]] = True,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Perform fusion search combining results from multiple embedding models.

    Args:
        query: The search query
        collection: The collection to search in
        embedding_models: List of embedding models to use (if None, uses default model and model+)
        fusion_method: Method for fusing results - "rrf" (Reciprocal Rank Fusion),
                      "max" (maximum score), or "avg" (average score)
        rrf_k: Parameter for RRF fusion (higher = less importance to top ranks)
        limit: Maximum number of results to return after fusion
        filter: Additional filters to apply to the search
        with_payload: Whether to include payload in the results
        ctx: Optional MCP context

    Returns:
        Dictionary containing fused search results and metadata
    """
    if ctx:
        ctx.info(f"Performing fusion search for query: '{query}'")
    
    # Step 1: Determine which embedding models to use
    if not embedding_models or len(embedding_models) == 0:
        # Use default model from config and fallback second model
        default_model = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        fallback_model = "all-mpnet-base-v2" if "MiniLM" in default_model else "all-MiniLM-L6-v2"
        embedding_models = [default_model, fallback_model]
    
    if ctx:
        ctx.info(f"Using embedding models: {', '.join(embedding_models)}")
    
    # Step 2: Perform search with each model
    model_results = {}
    all_ids = set()
    
    for model_name in embedding_models:
        if ctx:
            ctx.info(f"Searching with model: {model_name}")
        
        # Create a temporary embedding provider for this model
        try:
            # Get QdrantClient singleton
            qdrant_client = QdrantClient()
            
            # Generate embedding using the specified model
            embedding_provider = get_embedding_provider(model_name=model_name)
            embeddings = await embedding_provider.embed([query])
            
            if not embeddings or len(embeddings) == 0:
                if ctx:
                    ctx.warning(f"Failed to generate embedding with model {model_name}")
                continue
                
            query_vector = embeddings[0]
            
            # Perform search with higher limit to allow for better fusion
            higher_limit = limit * 3
            
            # Use search directly with Qdrant client
            search_results = await qdrant_client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=higher_limit,
                query_filter=filter,
                with_payload=with_payload
            )
            
            # Process results
            processed_results = []
            for i, result in enumerate(search_results):
                processed_result = {
                    "id": result.id,
                    "score": result.score,
                    "rank": i + 1,  # 1-based rank
                    "payload": result.payload
                }
                processed_results.append(processed_result)
                all_ids.add(result.id)
            
            model_results[model_name] = processed_results
            
            if ctx:
                ctx.info(f"Found {len(processed_results)} results with {model_name}")
                
        except Exception as e:
            error_msg = f"Error searching with model {model_name}: {str(e)}"
            logger.error(error_msg)
            if ctx:
                ctx.error(error_msg)
    
    if len(model_results) == 0:
        error_msg = "No search results found with any embedding model"
        if ctx:
            ctx.warning(error_msg)
        return {"error": error_msg, "results": []}
    
    # Step 3: Fuse results based on the selected method
    if ctx:
        ctx.info(f"Fusing results using method: {fusion_method}")
    
    # Create a map of document ID to combined score
    doc_scores = defaultdict(float)
    
    if fusion_method == "rrf":  # Reciprocal Rank Fusion
        for model_name, results in model_results.items():
            for result in results:
                doc_id = result["id"]
                rank = result["rank"]
                # RRF formula: 1 / (rank + k)
                doc_scores[doc_id] += 1.0 / (rank + rrf_k)
    
    elif fusion_method == "max":  # Maximum score
        for model_name, results in model_results.items():
            for result in results:
                doc_id = result["id"]
                score = result["score"]
                doc_scores[doc_id] = max(doc_scores[doc_id], score)
    
    else:  # Average score (default)
        doc_count = defaultdict(int)
        for model_name, results in model_results.items():
            for result in results:
                doc_id = result["id"]
                score = result["score"]
                doc_scores[doc_id] += score
                doc_count[doc_id] += 1
        
        # Calculate average
        for doc_id in doc_scores:
            if doc_count[doc_id] > 0:
                doc_scores[doc_id] /= doc_count[doc_id]
    
    # Step 4: Sort and format final results
    fused_results = []
    
    # Create a mapping of doc_id to original result with payload
    doc_details = {}
    for model_name, results in model_results.items():
        for result in results:
            doc_id = result["id"]
            if doc_id not in doc_details and "payload" in result:
                doc_details[doc_id] = result
    
    # Sort by fused score
    sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    
    # Create final results with limit
    for i, doc_id in enumerate(sorted_doc_ids[:limit]):
        original_result = doc_details.get(doc_id, {"id": doc_id})
        fused_result = {
            "id": doc_id,
            "score": doc_scores[doc_id],
            "rank": i + 1,
            "models": [model_name for model_name, results in model_results.items() 
                      if doc_id in [r["id"] for r in results]]
        }
        
        # Add payload if available
        if "payload" in original_result:
            fused_result["payload"] = original_result["payload"]
        
        fused_results.append(fused_result)
    
    # Step 5: Return the fused results with metadata
    return {
        "results": fused_results,
        "fusion_method": fusion_method,
        "embedding_models": embedding_models,
        "models_count": len(model_results),
        "total_unique_results": len(all_ids)
    }
