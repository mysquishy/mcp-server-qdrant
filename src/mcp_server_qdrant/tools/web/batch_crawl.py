"""Batch crawler for multiple URLs for Qdrant MCP Server."""
import os
import asyncio
from typing import Any, Dict, List, Optional, Union
import logging
import json
from datetime import datetime
import time

from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)

from .crawl_url import crawl_url, WEB_DEPENDENCIES_AVAILABLE


async def batch_crawl(
    urls: List[str],
    collection: Optional[str] = None,
    extract_metadata: bool = True,
    extract_links: bool = True,
    store_in_qdrant: bool = True,
    remove_html_tags: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    max_concurrent: int = 5,
    timeout_per_url: int = 30,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Crawl multiple URLs in parallel and process their content.

    Args:
        urls: List of URLs to crawl
        collection: The collection to store the extracted content
        extract_metadata: Whether to extract metadata from pages
        extract_links: Whether to extract links from pages
        store_in_qdrant: Whether to store content in Qdrant
        remove_html_tags: Whether to remove HTML tags from content
        chunk_size: Size of each chunk when storing in Qdrant
        chunk_overlap: Overlap between chunks
        max_concurrent: Maximum number of concurrent crawling tasks
        timeout_per_url: Timeout in seconds for each URL request
        ctx: Optional MCP context

    Returns:
        Dictionary containing results from all crawled URLs
    """
    if not WEB_DEPENDENCIES_AVAILABLE:
        error_msg = "Web crawler dependencies not installed. Install with: pip install httpx beautifulsoup4 lxml"
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg}
    
    if not urls:
        error_msg = "No URLs provided for batch crawling"
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg}
    
    # Validate max_concurrent parameter
    max_concurrent = max(1, min(max_concurrent, 20))  # Between 1 and 20
    
    # Log start of batch crawl
    if ctx:
        ctx.info(f"Starting batch crawl of {len(urls)} URLs with {max_concurrent} concurrent tasks")
        if collection:
            ctx.info(f"Will store content in collection: {collection}")
    
    start_time = time.time()
    results = []
    failures = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_crawl(url):
        """Crawl a URL with a semaphore to limit concurrency."""
        async with semaphore:
            if ctx:
                ctx.info(f"Processing URL: {url}")
            
            try:
                result = await crawl_url(
                    url=url,
                    collection=collection,
                    extract_metadata=extract_metadata,
                    extract_links=extract_links,
                    store_in_qdrant=store_in_qdrant,
                    remove_html_tags=remove_html_tags,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    timeout=timeout_per_url,
                    ctx=ctx
                )
                
                return url, result, None  # success
            except Exception as e:
                error_msg = f"Error crawling {url}: {str(e)}"
                logger.error(error_msg)
                if ctx:
                    ctx.error(error_msg)
                return url, None, str(e)  # failure
    
    # Process URLs in parallel with limited concurrency
    tasks = [bounded_crawl(url) for url in urls]
    
    # Show progress 
    completed = 0
    total = len(tasks)
    for coro in asyncio.as_completed(tasks):
        url, result, error = await coro
        completed += 1
        
        if ctx:
            ctx.report_progress(completed, total)
            ctx.info(f"Completed {completed}/{total} URLs ({completed/total*100:.1f}%)")
        
        if error:
            failures.append({"url": url, "error": error})
        else:
            results.append(result)
    
    # Calculate statistics
    end_time = time.time()
    total_time = end_time - start_time
    total_content_length = sum(r.get("content_length", 0) for r in results if isinstance(r, dict))
    total_links = sum(r.get("links_count", 0) for r in results if isinstance(r, dict))
    total_chunks_stored = sum(r.get("chunks_stored", 0) for r in results if isinstance(r, dict))
    
    if ctx:
        ctx.info(f"Batch crawl completed in {total_time:.2f} seconds")
        ctx.info(f"Successfully crawled: {len(results)}/{len(urls)} URLs")
        ctx.info(f"Failed: {len(failures)}/{len(urls)} URLs")
        if store_in_qdrant and collection:
            ctx.info(f"Stored {total_chunks_stored} chunks in collection '{collection}'")
    
    # Return summary
    return {
        "urls_processed": len(urls),
        "successful": len(results),
        "failed": len(failures),
        "total_time_seconds": total_time,
        "seconds_per_url": total_time / len(urls) if urls else 0,
        "total_content_length": total_content_length,
        "total_links_found": total_links,
        "total_chunks_stored": total_chunks_stored,
        "collection": collection if store_in_qdrant else None,
        "results": [{"url": r["url"], "title": r.get("title", ""), "content_length": r.get("content_length", 0)} 
                    for r in results if isinstance(r, dict)],
        "failures": failures
    }
