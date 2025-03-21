"""Batch Embedding tool for Qdrant MCP server."""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)

async def batch_embed(
    ctx: Context,
    texts: List[str],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate embeddings for multiple texts in batch.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    texts : List[str]
        List of texts to embed
    model : str, optional
        Model name to use for embedding (overrides default)
        
    Returns:
    --------
    Dict[str, Any]
        Embeddings for each text
    """
    if not texts:
        return {"status": "error", "message": "No texts provided", "embeddings": [], "count": 0}
        
    await ctx.debug(f"Generating embeddings for {len(texts)} texts")
    
    # Get the embedding provider from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    embedding_provider = qdrant_connector._embedding_provider
    
    try:
        total = len(texts)
        await ctx.info(f"Generating embeddings for {total} texts")
        
        # Process in batches for efficiency
        batch_size = 32  # Optimal batch size for most embedding providers
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        all_embeddings = []
        processed = 0
        
        for i, batch in enumerate(batches):
            await ctx.info(f"Processing batch {i+1}/{len(batches)}")
            await ctx.report_progress(processed, total)
            
            # Generate embeddings for the batch
            batch_embeddings = await embedding_provider.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
            processed += len(batch)
            await ctx.debug(f"Completed batch {i+1}/{len(batches)}, {processed}/{total} texts")
        
        await ctx.report_progress(total, total)
        await ctx.info(f"Completed embedding generation for {total} texts")
        
        # Get vector details for the response
        vector_name = embedding_provider.get_vector_name()
        dimensions = len(all_embeddings[0]) if all_embeddings else 0
        
        return {
            "status": "success",
            "embeddings": all_embeddings,
            "count": len(all_embeddings),
            "vector_name": vector_name,
            "dimensions": dimensions
        }
    except Exception as e:
        error_msg = f"Error generating embeddings: {str(e)}"
        logger.error(error_msg, exc_info=True)
        await ctx.debug(error_msg)
        
        return {
            "status": "error",
            "message": f"Failed to generate embeddings: {str(e)}",
            "embeddings": [],
            "count": 0
        }
