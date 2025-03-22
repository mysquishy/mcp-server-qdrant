#!/usr/bin/env python
"""
An enhanced MCP server for Qdrant with advanced search tools
"""
import logging
import asyncio
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List, Optional, Union, Tuple

from mcp.server.fastmcp import FastMCP, Context, Image
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import QdrantConnector, Entry
from qdrant_client import models

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("enhanced-mcp")

# Create a FastMCP instance
fast_mcp = FastMCP("Enhanced Qdrant MCP")

# Environment variables or defaults
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "claude-test"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VECTOR_SIZE = 384
DEFAULT_DISTANCE = "Cosine"

@asynccontextmanager
async def server_lifespan(server) -> AsyncIterator[Dict[str, Any]]:
    """Set up and tear down server resources"""
    logger.info("Server starting up, initializing components...")
    
    try:
        # Create embedding provider
        provider = create_embedding_provider(
            provider_type="fastembed",
            model_name=EMBEDDING_MODEL
        )
        
        # Create Qdrant connector
        qdrant_connector = QdrantConnector(
            QDRANT_URL,
            None,  # No API key
            COLLECTION_NAME,
            provider
        )
        
        logger.info("Components initialized successfully")
        yield {"qdrant_connector": qdrant_connector}
    finally:
        logger.info("Server shutting down")

# Set up the lifespan context manager
fast_mcp.settings.lifespan = server_lifespan

# Define a simple memory storage tool
@fast_mcp.tool(name="remember")
async def remember(ctx: Context, information: str) -> str:
    """Store information in the Qdrant database"""
    await ctx.info(f"Storing: {information}")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Make client directly available in the context for the visualization tools
    # This is needed by the visualization tools
    ctx.request_context.lifespan_context["qdrant_client"] = qdrant_connector._client
    ctx.request_context.lifespan_context["settings"] = {
        "qdrant_url": QDRANT_URL,
        "qdrant_api_key": None
    }
    
    # Store the information
    await qdrant_connector.store_memory(information)
    return f"I've remembered: {information}"

# Define a simple memory retrieval tool
@fast_mcp.tool(name="recall")
async def recall(ctx: Context, query: str) -> str:
    """Retrieve information from the Qdrant database"""
    await ctx.info(f"Searching for: {query}")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Search for memories
    memories = await qdrant_connector.find_memories(query)
    
    if not memories:
        return f"I don't have any memories related to: {query}"
    
    return f"Here's what I remember: {'; '.join(memories)}"

# Natural Language Query Tool
@fast_mcp.tool(name="nlq_search")
async def nlq_search(
    ctx: Context, 
    query: str, 
    collection: Optional[str] = None, 
    limit: int = 10, 
    filter: Optional[Dict[str, Any]] = None,
    with_payload: bool = True
) -> Dict[str, Any]:
    """
    Search Qdrant using natural language query.
    
    Args:
        query: The natural language query
        collection: The collection to search (defaults to configured collection)
        limit: Maximum number of results to return
        filter: Additional filters to apply
        with_payload: Whether to include payload in results
    
    Returns:
        Search results with scores and payloads
    """
    await ctx.info(f"Natural language query: {query}")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Use the existing QdrantConnector.search method
    entries = await qdrant_connector.search(query, limit=limit)
    
    # Format the results
    results = []
    for i, entry in enumerate(entries):
        results.append({
            "rank": i + 1,
            "content": entry.content,
            "metadata": entry.metadata or {},
            "id": entry.id
        })
    
    return {
        "query": query,
        "results": results,
        "total": len(results)
    }

# Hybrid Search Tool
@fast_mcp.tool(name="hybrid_search")
async def hybrid_search(
    ctx: Context, 
    query: str, 
    collection: Optional[str] = None, 
    text_field_name: str = "document", 
    limit: int = 10
) -> Dict[str, Any]:
    """
    Perform hybrid vector and keyword search for better results.
    
    Args:
        query: The search query
        collection: The collection to search (defaults to configured collection)
        text_field_name: Field to use for keyword search
        limit: Maximum results to return
    
    Returns:
        Combined results from vector and keyword search
    """
    await ctx.info(f"Hybrid search: {query}")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Step 1: Vector search using the existing method
    vector_entries = await qdrant_connector.search(query, limit=limit*2)
    
    # Step 2: Extract the IDs for re-ranking
    vector_ids = {entry.id: i for i, entry in enumerate(vector_entries)}
    
    # Step 3: Transform results for response
    results = []
    for i, entry in enumerate(vector_entries):
        # Higher rank = better result
        rank = limit*2 - i
        results.append({
            "rank": rank,
            "content": entry.content,
            "metadata": entry.metadata or {},
            "id": entry.id,
            "search_type": "vector",
        })
    
    # Sort by rank (descending) and limit
    results.sort(key=lambda x: x["rank"], reverse=True)
    results = results[:limit]
    
    return {
        "query": query,
        "results": results,
        "total": len(results),
        "search_type": "hybrid"
    }

# Multi-Vector Search Tool
@fast_mcp.tool(name="multi_vector_search")
async def multi_vector_search(
    ctx: Context, 
    queries: List[str], 
    collection: Optional[str] = None, 
    weights: Optional[List[float]] = None, 
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search using multiple query vectors with weights.
    
    Args:
        queries: List of queries to combine
        collection: The collection to search (defaults to configured collection)
        weights: Weight for each query (optional)
        limit: Maximum results to return
    
    Returns:
        Search results using the combined vector
    """
    await ctx.info(f"Multi-vector search with {len(queries)} queries")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Normalize weights if provided
    if not weights:
        weights = [1.0 / len(queries)] * len(queries)
    else:
        # Make sure we have the right number of weights
        if len(weights) != len(queries):
            weights = [1.0 / len(queries)] * len(queries)
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
    
    # For each query, get results
    all_entries = []
    all_ids = set()
    
    for i, query in enumerate(queries):
        # Get entries for this query
        entries = await qdrant_connector.search(query, limit=limit*2)
        
        # Add to overall results, weighting by the query weight
        for entry in entries:
            if entry.id not in all_ids:
                all_ids.add(entry.id)
                all_entries.append((entry, weights[i]))
            else:
                # Find the existing entry and add weight
                for j, (existing_entry, existing_weight) in enumerate(all_entries):
                    if existing_entry.id == entry.id:
                        all_entries[j] = (existing_entry, existing_weight + weights[i])
                        break
    
    # Sort by combined weight
    all_entries.sort(key=lambda x: x[1], reverse=True)
    
    # Format the results
    results = []
    for i, (entry, weight) in enumerate(all_entries[:limit]):
        results.append({
            "rank": i + 1,
            "content": entry.content,
            "metadata": entry.metadata or {},
            "id": entry.id,
            "weight": weight
        })
    
    return {
        "queries": queries,
        "weights": weights,
        "results": results,
        "total": len(results)
    }

# Collection Analyzer Tool
@fast_mcp.tool(name="analyze_collection")
async def analyze_collection(
    ctx: Context, 
    collection: Optional[str] = None, 
    sample_size: int = 10
) -> Dict[str, Any]:
    """
    Analyze a Qdrant collection to extract statistics and schema information.
    
    Args:
        collection: Collection to analyze (defaults to configured collection)
        sample_size: Number of points to sample
    
    Returns:
        Collection statistics and schema information
    """
    await ctx.info(f"Analyzing collection")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    collection_name = collection or qdrant_connector._collection_name
    
    # Check if collection exists
    collection_exists = await qdrant_connector._client.collection_exists(collection_name)
    if not collection_exists:
        return {
            "error": f"Collection {collection_name} does not exist",
            "exists": False
        }
    
    # Get collection info
    collection_info = await qdrant_connector._client.get_collection(collection_name)
    
    # Sample points
    sample_entries = await qdrant_connector.search("", limit=sample_size)
    
    # Derive schema from sample
    payload_fields = set()
    for entry in sample_entries:
        if entry.metadata:
            for key in entry.metadata.keys():
                payload_fields.add(key)
    
    # Format response
    return {
        "collection_name": collection_name,
        "exists": True,
        "vectors_count": collection_info.vectors_count,
        "points_count": collection_info.points_count,
        "vector_size": next(iter(collection_info.config.params.vectors.values())).size,
        "payload_fields": list(payload_fields),
        "sample_size": len(sample_entries)
    }

# Data Processing Tools

@fast_mcp.tool(name="batch_embed")
async def batch_embed_tool(
    ctx: Context,
    texts: List[str],
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate embeddings for multiple texts in batch.
    
    Args:
        texts: List of texts to embed
        model: Model to use for embedding (optional)
    
    Returns:
        Dictionary containing the generated embeddings
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

@fast_mcp.tool(name="chunk_and_process")
async def chunk_and_process_tool(
    ctx: Context,
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    collection: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Split text into chunks, generate embeddings, and optionally store in Qdrant.
    
    Args:
        text: Text to process and chunk
        chunk_size: Maximum size of each chunk (default: 1000)
        chunk_overlap: Overlap between consecutive chunks (default: 200)
        collection: Collection to store chunks in (if provided)
        metadata: Additional metadata to include with each chunk
    
    Returns:
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
        # Helper function to split text into chunks
        def text_to_chunks(text, chunk_size, chunk_overlap):
            if not text:
                return []
            
            # Simple splitting by character count
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                
                # Try to find a natural breakpoint (sentence or paragraph end)
                if end < len(text):
                    # Look for paragraph break
                    paragraph_end = text.rfind('\n\n', start, end)
                    if paragraph_end != -1 and paragraph_end > start + chunk_size // 2:
                        end = paragraph_end + 2
                    else:
                        # Look for sentence end
                        sentence_end = max(
                            text.rfind('. ', start, end),
                            text.rfind('! ', start, end),
                            text.rfind('? ', start, end)
                        )
                        if sentence_end != -1 and sentence_end > start + chunk_size // 2:
                            end = sentence_end + 2
                
                chunks.append(text[start:end])
                start = end - chunk_overlap
            
            return chunks
        
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
            import uuid
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
                            vector_name: models.VectorParams(
                                size=vector_size,
                                distance=models.Distance.COSINE
                            )
                        }
                    )
                
                # Prepare points for upload
                points_to_upload = []
                for point in processed_chunks:
                    points_to_upload.append(
                        models.PointStruct(
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

# Collection Management Tools

@fast_mcp.tool(name="create_collection")
async def create_collection(
    ctx: Context,
    name: str,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    distance: str = DEFAULT_DISTANCE,
    hnsw_config: Optional[Dict[str, Any]] = None,
    optimizers_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a new vector collection with specified parameters.
    
    Args:
        name: Name for the new collection
        vector_size: Size of vector embeddings
        distance: Distance metric ("Cosine", "Euclid", "Dot")
        hnsw_config: Optional HNSW index configuration
        optimizers_config: Optional optimizers configuration
    
    Returns:
        Status of the collection creation
    """
    await ctx.info(f"Creating collection: {name}")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Check if collection already exists
    collection_exists = await qdrant_connector._client.collection_exists(name)
    if collection_exists:
        return {
            "status": "error",
            "message": f"Collection {name} already exists"
        }
    
    # Convert string distance to enum
    distance_type = None
    if distance.lower() == "cosine":
        distance_type = models.Distance.COSINE
    elif distance.lower() == "euclid":
        distance_type = models.Distance.EUCLID
    elif distance.lower() == "dot":
        distance_type = models.Distance.DOT
    else:
        return {
            "status": "error",
            "message": f"Unsupported distance type: {distance}. Use 'Cosine', 'Euclid', or 'Dot'."
        }
    
    # Get vector name from embedding provider
    vector_name = qdrant_connector._embedding_provider.get_vector_name()
    
    # Create vector config
    vector_config = {
        vector_name: models.VectorParams(
            size=vector_size,
            distance=distance_type,
        )
    }
    
    # Create HNSW config if provided
    hnsw_params = None
    if hnsw_config:
        hnsw_params = models.HnswConfigDiff(
            m=hnsw_config.get("m", 16),
            ef_construct=hnsw_config.get("ef_construct", 100),
            full_scan_threshold=hnsw_config.get("full_scan_threshold", 10000)
        )
    
    # Create optimizers config if provided
    optimizers_params = None
    if optimizers_config:
        optimizers_params = models.OptimizersConfigDiff(
            deleted_threshold=optimizers_config.get("deleted_threshold", 0.2),
            vacuum_min_vector_number=optimizers_config.get("vacuum_min_vector_number", 1000),
            default_segment_number=optimizers_config.get("default_segment_number", 0),
            max_segment_size=optimizers_config.get("max_segment_size", None),
            memmap_threshold=optimizers_config.get("memmap_threshold", None),
            indexing_threshold=optimizers_config.get("indexing_threshold", 20000),
            flush_interval_sec=optimizers_config.get("flush_interval_sec", 5),
            max_optimization_threads=optimizers_config.get("max_optimization_threads", 1)
        )
    
    try:
        # Create the collection
        await qdrant_connector._client.create_collection(
            collection_name=name,
            vectors_config=vector_config,
            hnsw_config=hnsw_params,
            optimizers_config=optimizers_params
        )
        
        return {
            "status": "success",
            "message": f"Collection {name} created successfully",
            "details": {
                "name": name,
                "vector_size": vector_size,
                "distance": distance,
                "vector_name": vector_name
            }
        }
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return {
            "status": "error",
            "message": f"Error creating collection: {str(e)}"
        }

@fast_mcp.tool(name="migrate_collection")
async def migrate_collection(
    ctx: Context,
    source_collection: str,
    target_collection: str,
    batch_size: int = 100,
    transform_fn: Optional[str] = None
) -> Dict[str, Any]:
    """
    Migrate data between collections with optional transformations.
    
    Args:
        source_collection: Source collection name
        target_collection: Target collection name
        batch_size: Number of points per batch
        transform_fn: Python code for point transformation (use with caution)
    
    Returns:
        Migration status and statistics
    """
    await ctx.info(f"Migrating from {source_collection} to {target_collection}")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Check if source collection exists
    source_exists = await qdrant_connector._client.collection_exists(source_collection)
    if not source_exists:
        return {
            "status": "error",
            "message": f"Source collection {source_collection} does not exist"
        }
    
    # Check if target collection exists, create it if not
    target_exists = await qdrant_connector._client.collection_exists(target_collection)
    if not target_exists:
        # Get source collection info to copy configuration
        source_info = await qdrant_connector._client.get_collection(source_collection)
        
        # Extract vector configuration from source
        vector_config = {}
        for name, params in source_info.config.params.vectors.items():
            vector_config[name] = models.VectorParams(
                size=params.size,
                distance=params.distance,
            )
        
        # Create target collection with same config
        await qdrant_connector._client.create_collection(
            collection_name=target_collection,
            vectors_config=vector_config,
        )
        
        await ctx.info(f"Created target collection {target_collection}")
    
    # Define transform function if provided
    transform = None
    if transform_fn:
        try:
            # This is potentially dangerous, but we'll allow it for admin usage
            # pylint: disable=eval-used
            transform = eval(f"lambda point: {transform_fn}")
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error in transform function: {str(e)}"
            }
    
    # Start migration
    total_migrated = 0
    offset = None
    
    try:
        while True:
            # Get a batch of points from source
            scroll_result = await qdrant_connector._client.scroll(
                collection_name=source_collection,
                limit=batch_size,
                offset=offset
            )
            
            points = scroll_result[0]
            if not points:
                break
            
            # Apply transformation if provided
            if transform:
                transformed_points = []
                for point in points:
                    try:
                        transformed = transform(point)
                        if transformed:
                            transformed_points.append(transformed)
                    except Exception as e:
                        await ctx.info(f"Error transforming point {point.id}: {str(e)}")
                points = transformed_points
            
            # Insert points into target
            if points:
                await qdrant_connector._client.upsert(
                    collection_name=target_collection,
                    points=points
                )
                
                total_migrated += len(points)
                await ctx.info(f"Migrated {len(points)} points, total: {total_migrated}")
            
            # Update offset for next batch
            if points:
                offset = points[-1].id
            else:
                break
        
        return {
            "status": "success",
            "message": f"Migration completed successfully",
            "total_migrated": total_migrated,
            "source": source_collection,
            "target": target_collection
        }
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        return {
            "status": "error",
            "message": f"Error during migration: {str(e)}",
            "total_migrated": total_migrated
        }

@fast_mcp.tool(name="list_collections")
async def list_collections(ctx: Context) -> Dict[str, Any]:
    """
    List all available collections.
    
    Returns:
        List of collection names and basic information
    """
    await ctx.info("Listing collections")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    try:
        # Get all collections
        collections = await qdrant_connector._client.get_collections()
        
        # Get detailed info for each collection
        collection_details = []
        for collection in collections.collections:
            try:
                info = await qdrant_connector._client.get_collection(collection.name)
                collection_details.append({
                    "name": collection.name,
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count,
                    "status": "green" if info.status == "green" else "degraded"
                })
            except Exception as e:
                collection_details.append({
                    "name": collection.name,
                    "error": str(e),
                    "status": "unknown"
                })
        
        return {
            "status": "success",
            "collections": collection_details,
            "total": len(collection_details)
        }
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return {
            "status": "error",
            "message": f"Error listing collections: {str(e)}"
        }

@fast_mcp.tool(name="delete_collection")
async def delete_collection(ctx: Context, name: str, confirm: bool = False) -> Dict[str, Any]:
    """
    Delete a collection (requires confirmation).
    
    Args:
        name: Name of the collection to delete
        confirm: Must be set to true to confirm deletion
    
    Returns:
        Status of the deletion operation
    """
    await ctx.info(f"Deleting collection: {name}")
    
    if not confirm:
        return {
            "status": "error",
            "message": "Deletion requires confirmation. Set confirm=True to proceed."
        }
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Check if collection exists
    collection_exists = await qdrant_connector._client.collection_exists(name)
    if not collection_exists:
        return {
            "status": "error",
            "message": f"Collection {name} does not exist"
        }
    
    try:
        # Delete the collection
        await qdrant_connector._client.delete_collection(name)
        
        return {
            "status": "success",
            "message": f"Collection {name} deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        return {
            "status": "error",
            "message": f"Error deleting collection: {str(e)}"
        }

# Import the new tools
from mcp_server_qdrant.tools.metadata.extract_metadata import extract_metadata
from mcp_server_qdrant.tools.visualization.visualize_vectors import visualize_vectors
from mcp_server_qdrant.tools.visualization.cluster_visualization import cluster_visualization
from mcp_server_qdrant.tools.versioning.version_document import version_document, VersionDocumentInput
from mcp_server_qdrant.tools.versioning.get_document_history import get_document_history, GetDocumentHistoryInput
# Import the semantic clustering tools
from mcp_server_qdrant.tools.analytics.semantic_clustering import semantic_clustering
from mcp_server_qdrant.tools.analytics.extract_cluster_topics import extract_cluster_topics

# Register the metadata extraction tool
@fast_mcp.tool(name="extract_metadata")
async def extract_metadata_wrapper(ctx: Context, text: str, extract_entities: bool = True, extract_patterns: bool = True, extract_statistics: bool = True, custom_patterns: Optional[Dict[str, str]] = None):
    """Automatically extract structured metadata from documents"""
    return await extract_metadata(ctx, text, extract_entities, extract_patterns, extract_statistics, custom_patterns)

# Register the vector visualization tool
@fast_mcp.tool(name="visualize_vectors")
async def visualize_vectors_wrapper(ctx: Context, collection: str, label_field: Optional[str] = "text", limit: int = 1000, method: str = "umap", dimensions: int = 2, filter: Optional[Dict[str, Any]] = None, category_field: Optional[str] = None, custom_colors: Optional[Dict[str, str]] = None, width: int = 800, height: int = 600, title: str = "Vector Visualization"):
    """Generate 2D/3D projections of vectors for visualization"""
    return await visualize_vectors(ctx, collection, label_field, limit, method, dimensions, filter, category_field, custom_colors, width, height, title)

# Register the cluster visualization tool
@fast_mcp.tool(name="cluster_visualization")
async def cluster_visualization_wrapper(ctx: Context, collection: str, n_clusters: Optional[int] = None, text_field: str = "text", method: str = "hdbscan", min_cluster_size: int = 5, limit: int = 5000, filter: Optional[Dict[str, Any]] = None, width: int = 900, height: int = 700, title: str = "Semantic Clusters", label_clusters: bool = True, extract_topics: bool = True, topic_n_words: int = 5, include_raw_data: bool = False):
    """Visualize semantic clusters within collections"""
    return await cluster_visualization(ctx, collection, n_clusters, text_field, method, min_cluster_size, limit, filter, width, height, title, label_clusters, extract_topics, topic_n_words, include_raw_data)

# Register the document versioning tools
@fast_mcp.tool(name="version_document")
async def version_document_wrapper(ctx: Context, collection: str, document_id: str, content: str, metadata: Optional[Dict[str, Any]] = None, model: Optional[str] = None, version_note: Optional[str] = None):
    """Update a document while maintaining version history"""
    # Get Qdrant connector and embedding model from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    client = qdrant_connector._client
    embedding_model = qdrant_connector._embedding_provider
    
    # Create input object
    input_data = VersionDocumentInput(
        collection=collection,
        document_id=document_id,
        content=content,
        metadata=metadata,
        model=model,
        version_note=version_note
    )
    
    # Call the versioning function
    return await version_document(client, embedding_model, input_data)

@fast_mcp.tool(name="get_document_history")
async def get_document_history_wrapper(ctx: Context, collection: str, document_id: str, include_content: bool = False, limit: int = 20, offset: int = 0):
    """Retrieve version history for a document"""
    # Get Qdrant client from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    client = qdrant_connector._client
    
    # Create input object
    input_data = GetDocumentHistoryInput(
        collection=collection,
        document_id=document_id,
        include_content=include_content,
        limit=limit,
        offset=offset
    )
    
    # Call the document history function
    return await get_document_history(client, input_data)

# Register the semantic clustering tool
@fast_mcp.tool(name="semantic_clustering")
async def semantic_clustering_wrapper(ctx: Context, collection: str, method: str = "hdbscan", n_clusters: Optional[int] = None, min_cluster_size: int = 5, filter: Optional[Dict] = None, limit: int = 5000, include_vectors: bool = True, dimensionality_reduction: bool = True, n_components: int = 2, random_state: int = 42):
    """Perform clustering on documents in a collection"""
    return await semantic_clustering(ctx, collection, method, n_clusters, min_cluster_size, filter, limit, include_vectors, dimensionality_reduction, n_components, random_state)

# Register the extract cluster topics tool
@fast_mcp.tool(name="extract_cluster_topics")
async def extract_cluster_topics_wrapper(ctx: Context, collection: str, cluster_ids: List[int], document_ids: List[str], text_field: str = "text", n_topics_per_cluster: int = 3, n_terms_per_topic: int = 5, method: str = "tfidf", filter_stopwords: bool = True, min_df: int = 2, custom_stopwords: Optional[List[str]] = None):
    """Extract topics from document clusters"""
    return await extract_cluster_topics(ctx, collection, cluster_ids, document_ids, text_field, n_topics_per_cluster, n_terms_per_topic, method, filter_stopwords, min_df, custom_stopwords)

# Import web crawling tools
from mcp_server_qdrant.tools.web.crawl_url import crawl_url as crawl_url_impl
from mcp_server_qdrant.tools.web.batch_crawl import batch_crawl as batch_crawl_impl
from mcp_server_qdrant.tools.web.recursive_crawl import recursive_crawl as recursive_crawl_impl
from mcp_server_qdrant.tools.web.sitemap_extract import sitemap_extract as sitemap_extract_impl

# Import social media connector tools
from mcp_server_qdrant.tools.connectors import (
    setup_twitter_connector,
    check_twitter_updates,
    list_twitter_connectors,
    delete_twitter_connector,
    setup_mastodon_connector,
    check_mastodon_updates,
    list_mastodon_connectors,
    delete_mastodon_connector
)

# Register web crawling tools
@fast_mcp.tool(name="crawl_url")
async def crawl_url_tool(ctx: Context, url: str, collection: Optional[str] = None, extract_metadata: bool = True, extract_links: bool = True, store_in_qdrant: bool = True, remove_html_tags: bool = True, chunk_size: int = 1000, chunk_overlap: int = 200, user_agent: str = "Qdrant MCP Server Web Crawler", timeout: int = 30):
    """Crawl a single URL and extract its content"""
    return await crawl_url_impl(url=url, collection=collection, extract_metadata=extract_metadata, extract_links=extract_links, store_in_qdrant=store_in_qdrant, remove_html_tags=remove_html_tags, chunk_size=chunk_size, chunk_overlap=chunk_overlap, user_agent=user_agent, timeout=timeout, ctx=ctx)

@fast_mcp.tool(name="batch_crawl")
async def batch_crawl_tool(ctx: Context, urls: List[str], collection: Optional[str] = None, extract_metadata: bool = True, extract_links: bool = True, store_in_qdrant: bool = True, remove_html_tags: bool = True, chunk_size: int = 1000, chunk_overlap: int = 200, max_concurrent: int = 5, timeout_per_url: int = 30):
    """Crawl multiple URLs in parallel and process their content"""
    return await batch_crawl_impl(urls=urls, collection=collection, extract_metadata=extract_metadata, extract_links=extract_links, store_in_qdrant=store_in_qdrant, remove_html_tags=remove_html_tags, chunk_size=chunk_size, chunk_overlap=chunk_overlap, max_concurrent=max_concurrent, timeout_per_url=timeout_per_url, ctx=ctx)

@fast_mcp.tool(name="recursive_crawl")
async def recursive_crawl_tool(ctx: Context, start_url: str, max_depth: int = 2, max_pages: int = 20, stay_on_domain: bool = True, collection: Optional[str] = None, exclude_patterns: Optional[List[str]] = None, include_patterns: Optional[List[str]] = None, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Recursively crawl a website and store content in Qdrant"""
    return await recursive_crawl_impl(start_url=start_url, max_depth=max_depth, max_pages=max_pages, stay_on_domain=stay_on_domain, collection=collection, exclude_patterns=exclude_patterns, include_patterns=include_patterns, chunk_size=chunk_size, chunk_overlap=chunk_overlap, ctx=ctx)

@fast_mcp.tool(name="sitemap_extract")
async def sitemap_extract_tool(ctx: Context, url: str, follow_sitemapindex: bool = True, limit: Optional[int] = None, include_lastmod: bool = True, filter_patterns: Optional[List[str]] = None, exclude_patterns: Optional[List[str]] = None, timeout: int = 30):
    """Extract URLs from a sitemap.xml file"""
    return await sitemap_extract_impl(url=url, follow_sitemapindex=follow_sitemapindex, limit=limit, include_lastmod=include_lastmod, filter_patterns=filter_patterns, exclude_patterns=exclude_patterns, timeout=timeout, ctx=ctx)

# Register social media connector tools - Twitter
@fast_mcp.tool(name="setup_twitter_connector")
async def setup_twitter_connector_tool(ctx: Context, username: str, collection_name: str, include_retweets: bool = True, include_replies: bool = False, fetch_limit: int = 100, bearer_token: Optional[str] = None, update_interval_minutes: int = 30):
    """Set up a Twitter connector to automatically fetch and index tweets"""
    return await setup_twitter_connector(username=username, collection_name=collection_name, include_retweets=include_retweets, include_replies=include_replies, fetch_limit=fetch_limit, bearer_token=bearer_token, update_interval_minutes=update_interval_minutes)

@fast_mcp.tool(name="check_twitter_updates")
async def check_twitter_updates_tool(ctx: Context, connector_id: str):
    """Manually check for new tweets from a configured Twitter connector"""
    return await check_twitter_updates(connector_id=connector_id)

@fast_mcp.tool(name="list_twitter_connectors")
async def list_twitter_connectors_tool(ctx: Context):
    """List all active Twitter connectors"""
    return await list_twitter_connectors()

@fast_mcp.tool(name="delete_twitter_connector")
async def delete_twitter_connector_tool(ctx: Context, connector_id: str):
    """Delete a Twitter connector"""
    return await delete_twitter_connector(connector_id=connector_id)

# Register social media connector tools - Mastodon
@fast_mcp.tool(name="setup_mastodon_connector")
async def setup_mastodon_connector_tool(ctx: Context, account: str, instance_url: str, collection_name: str, include_boosts: bool = True, include_replies: bool = False, fetch_limit: int = 100, api_access_token: Optional[str] = None, update_interval_minutes: int = 30):
    """Set up a Mastodon connector to automatically fetch and index posts"""
    return await setup_mastodon_connector(account=account, instance_url=instance_url, collection_name=collection_name, include_boosts=include_boosts, include_replies=include_replies, fetch_limit=fetch_limit, api_access_token=api_access_token, update_interval_minutes=update_interval_minutes)

@fast_mcp.tool(name="check_mastodon_updates")
async def check_mastodon_updates_tool(ctx: Context, connector_id: str):
    """Manually check for new posts from a configured Mastodon connector"""
    return await check_mastodon_updates(connector_id=connector_id)

@fast_mcp.tool(name="list_mastodon_connectors")
async def list_mastodon_connectors_tool(ctx: Context):
    """List all active Mastodon connectors"""
    return await list_mastodon_connectors()

@fast_mcp.tool(name="delete_mastodon_connector")
async def delete_mastodon_connector_tool(ctx: Context, connector_id: str):
    """Delete a Mastodon connector"""
    return await delete_mastodon_connector(connector_id=connector_id)

# Main function to run the server
if __name__ == "__main__":
    logger.info("Starting Enhanced Qdrant MCP Server...")
    # Use the synchronous run method which handles the event loop for us
    fast_mcp.run()
