"""Chunking and Processing tool for Qdrant MCP server."""
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context
from qdrant_client.http.models import PointStruct

from mcp_server_qdrant.tools.utils import text_to_chunks

logger = logging.getLogger(__name__)

async def chunk_and_process(
    ctx: Context,
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    collection: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Split text into chunks, generate embeddings, and optionally store in Qdrant.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    text : str
        Text to process and chunk
    chunk_size : int, default=1000
        Maximum size of each chunk
    chunk_overlap : int, default=200
        Overlap between consecutive chunks
    collection : str, optional
        Collection to store chunks in (if provided)
    metadata : Dict[str, Any], optional
        Additional metadata to include with each chunk
        
    Returns:
    --------
    Dict[str, Any]
        Processed chunks with embeddings
    """
    if not text:
        return {"status": "error", "message": "No text provided", "chunks": [], "count": 0}
    
    # Validate parameters
    if chunk_size <= 0:
        return {"status": "error", "message": "Chunk size must be positive", "chunks": [], "count": 0}
    if chunk_overlap < 0:
        return {"status": "error", "message": "Chunk overlap must be non-negative", "chunks": [], "count": 0}
    if chunk_overlap >= chunk_size:
        return {"status": "error", "message": "Chunk overlap must be less than chunk size", "chunks": [], "count": 0}
    
    await ctx.debug(f"Processing text with chunk size {chunk_size}, overlap {chunk_overlap}")
    
    # Get dependencies from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    embedding_provider = qdrant_connector._embedding_provider
    client = qdrant_connector._client
    
    # Create metadata dict if not provided
    if metadata is None:
        metadata = {}
    
    # Add timestamp to metadata
    metadata["processed_at"] = time.time()
    
    try:
        # Step 1: Split text into chunks
        await ctx.info(f"Splitting text into chunks (size: {chunk_size}, overlap: {chunk_overlap})")
        chunks = text_to_chunks(text, chunk_size, chunk_overlap)
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            return {"status": "error", "message": "No chunks generated", "chunks": [], "count": 0}
        
        await ctx.info(f"Generated {total_chunks} chunks from input text")
        
        # Step 2: Generate embeddings for all chunks
        await ctx.info(f"Generating embeddings for {total_chunks} chunks")
        
        # Process chunks in batches
        processed_chunks = []
        vector_name = embedding_provider.get_vector_name()
        
        # Generate batch of embeddings
        chunk_embeddings = await embedding_provider.embed_documents(chunks)
        
        # Step 3: Process each chunk with its embedding
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            await ctx.report_progress(i, total_chunks)
            
            # Create unique ID for the chunk
            chunk_id = str(uuid.uuid4())
            
            # Create chunk metadata
            chunk_metadata = {
                "chunk_index": i,
                "total_chunks": total_chunks,
                "created_at": time.time(),
                "chunk_size": len(chunk),
                **metadata  # Add user-provided metadata
            }
            
            # Create point
            point = {
                "id": chunk_id,
                "vector": embedding,
                "payload": {
                    "text": chunk,
                    **chunk_metadata
                }
            }
            processed_chunks.append(point)
        
        # Step 4: Upload to collection if specified
        if collection and processed_chunks:
            try:
                # Check if collection exists, create if not
                collection_exists = await client.collection_exists(collection)
                if not collection_exists:
                    await ctx.info(f"Creating collection '{collection}'")
                    # Get vector size from the first embedding
                    vector_size = len(chunk_embeddings[0])
                    await client.create_collection(
                        collection_name=collection,
                        vectors_config={
                            vector_name: {
                                "size": vector_size,
                                "distance": "Cosine"
                            }
                        }
                    )
                
                # Prepare points for upload
                points_to_upload = []
                for point in processed_chunks:
                    points_to_upload.append(
                        PointStruct(
                            id=point["id"],
                            vector={vector_name: point["vector"]},
                            payload=point["payload"]
                        )
                    )
                
                # Upload in batches of 100
                upload_batch_size = 100
                for i in range(0, len(points_to_upload), upload_batch_size):
                    batch = points_to_upload[i:i + upload_batch_size]
                    await client.upsert(
                        collection_name=collection,
                        points=batch
                    )
                    await ctx.debug(f"Uploaded batch {i//upload_batch_size + 1}/{(len(points_to_upload) + upload_batch_size - 1)//upload_batch_size}")
                
                await ctx.info(f"Successfully uploaded {len(processed_chunks)} chunks to collection '{collection}'")
            except Exception as e:
                error_msg = f"Error uploading to collection: {str(e)}"
                logger.error(error_msg, exc_info=True)
                await ctx.debug(error_msg)
                
                return {
                    "status": "error",
                    "message": error_msg,
                    "chunks": processed_chunks,
                    "count": len(processed_chunks)
                }
        
        # Return processed chunks (without the full embeddings to reduce payload size)
        result_chunks = []
        for chunk in processed_chunks:
            result_chunks.append({
                "id": chunk["id"],
                "payload": chunk["payload"],
                "vector_dimensions": len(chunk["vector"])
            })
        
        return {
            "status": "success",
            "chunks": result_chunks,
            "count": len(result_chunks),
            "metadata": metadata,
            "collection": collection
        }
    except Exception as e:
        error_msg = f"Error processing chunks: {str(e)}"
        logger.error(error_msg, exc_info=True)
        await ctx.debug(error_msg)
        
        return {
            "status": "error",
            "message": error_msg,
            "chunks": [],
            "count": 0
        }
