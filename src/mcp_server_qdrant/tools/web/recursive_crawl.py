"""Recursive website crawler for Qdrant MCP Server."""
import os
import asyncio
from typing import Any, Dict, List, Optional, Set, Union
import logging
import json
from datetime import datetime
import time
import re
from urllib.parse import urlparse, urljoin

from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)

from .crawl_url import crawl_url, WEB_DEPENDENCIES_AVAILABLE
from ..data_processing.chunk_and_process import chunk_and_process


async def recursive_crawl(
    start_url: str,
    collection: Optional[str] = None,
    max_depth: int = 2,
    max_urls: int = 100,
    stay_within_domain: bool = True,
    stay_within_path: bool = False,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    max_concurrent: int = 5,
    timeout_per_url: int = 30,
    extract_metadata: bool = True,
    store_in_qdrant: bool = True,
    remove_html_tags: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Recursively crawl a website by following links up to a specified depth.

    Args:
        start_url: Starting URL for the crawl
        collection: The collection to store the extracted content
        max_depth: Maximum link depth to follow (0 = start_url only)
        max_urls: Maximum number of URLs to process
        stay_within_domain: Whether to only follow links within the same domain
        stay_within_path: Whether to only follow links within the same path
        include_patterns: List of regex patterns to include URLs (None for no filtering)
        exclude_patterns: List of regex patterns to exclude URLs
        max_concurrent: Maximum number of concurrent crawling tasks
        timeout_per_url: Timeout in seconds for each URL request
        extract_metadata: Whether to extract metadata from pages
        store_in_qdrant: Whether to store content in Qdrant
        remove_html_tags: Whether to remove HTML tags from content
        chunk_size: Size of each chunk when storing in Qdrant
        chunk_overlap: Overlap between chunks
        ctx: Optional MCP context

    Returns:
        Dictionary containing results from the recursive crawl
    """
    if not WEB_DEPENDENCIES_AVAILABLE:
        error_msg = "Web crawler dependencies not installed. Install with: pip install httpx beautifulsoup4 lxml"
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg}
    
    # Validate parameters
    max_depth = max(0, min(max_depth, 10))  # Between 0 and 10
    max_urls = max(1, min(max_urls, 1000))  # Between 1 and 1000
    max_concurrent = max(1, min(max_concurrent, 20))  # Between 1 and 20
    
    # Parse the start URL to get domain and path information
    try:
        parsed_start_url = urlparse(start_url)
        start_domain = parsed_start_url.netloc
        start_path = parsed_start_url.path
        
        if not parsed_start_url.scheme or not start_domain:
            error_msg = f"Invalid start URL format: {start_url}"
            if ctx:
                ctx.error(error_msg)
            return {"error": error_msg}
    except Exception as e:
        error_msg = f"Error parsing start URL: {str(e)}"
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg}
    
    # Compile regex patterns if provided
    compiled_includes = None
    compiled_excludes = None
    
    if include_patterns:
        compiled_includes = [re.compile(pattern) for pattern in include_patterns]
        if ctx:
            ctx.info(f"Using {len(compiled_includes)} URL inclusion filters")
    
    if exclude_patterns:
        compiled_excludes = [re.compile(pattern) for pattern in exclude_patterns]
        if ctx:
            ctx.info(f"Using {len(compiled_excludes)} URL exclusion filters")
    
    if ctx:
        ctx.info(f"Starting recursive crawl from {start_url}")
        ctx.info(f"Max depth: {max_depth}, Max URLs: {max_urls}")
        if stay_within_domain:
            ctx.info(f"Staying within domain: {start_domain}")
        if stay_within_path:
            ctx.info(f"Staying within path: {start_path}")
    
    # Initialize crawl state
    start_time = time.time()
    visited_urls = set()
    queued_urls = set()
    failed_urls = []
    processed_results = []
    url_queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Add the start URL to the queue with depth 0
    await url_queue.put((start_url, 0))
    queued_urls.add(start_url)
    
    async def should_crawl_url(url: str, depth: int) -> bool:
        """Determine if a URL should be crawled based on filtering rules."""
        # Skip if we've reached max URLs
        if len(visited_urls) >= max_urls:
            return False
        
        # Skip if depth is greater than max_depth
        if depth > max_depth:
            return False
        
        # Skip if already visited or queued
        if url in visited_urls or url in queued_urls:
            return False
        
        try:
            parsed_url = urlparse(url)
            
            # Skip non-HTTP(S) URLs
            if parsed_url.scheme not in ('http', 'https'):
                return False
            
            # Skip URLs without domain
            if not parsed_url.netloc:
                return False
            
            # Check domain constraint
            if stay_within_domain and parsed_url.netloc != start_domain:
                return False
            
            # Check path constraint
            if stay_within_path and not parsed_url.path.startswith(start_path):
                return False
            
            # Apply include filters
            if compiled_includes and not any(pattern.search(url) for pattern in compiled_includes):
                return False
            
            # Apply exclude filters
            if compiled_excludes and any(pattern.search(url) for pattern in compiled_excludes):
                return False
            
            return True
        except Exception:
            return False
    
    async def process_url(url: str, depth: int):
        """Process a single URL and extract links for further crawling."""
        async with semaphore:
            if ctx:
                ctx.info(f"Processing URL (depth {depth}): {url}")
            
            visited_urls.add(url)
            
            try:
                # Crawl the URL
                result = await crawl_url(
                    url=url,
                    collection=collection if store_in_qdrant else None,
                    extract_metadata=extract_metadata,
                    extract_links=True,  # Always extract links for recursive crawling
                    store_in_qdrant=store_in_qdrant,
                    remove_html_tags=remove_html_tags,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    timeout=timeout_per_url,
                    ctx=ctx
                )
                
                # Add depth information to result
                result['depth'] = depth
                processed_results.append(result)
                
                # Process links if not at max depth and not reached max URLs
                if depth < max_depth and len(visited_urls) < max_urls:
                    links = result.get('links', [])
                    
                    if ctx:
                        ctx.info(f"Found {len(links)} links at depth {depth}")
                    
                    # Queue new URLs for crawling
                    for link_info in links:
                        link_url = link_info.get('url', '')
                        
                        # Check if we should crawl this URL
                        if await should_crawl_url(link_url, depth + 1):
                            await url_queue.put((link_url, depth + 1))
                            queued_urls.add(link_url)
                            
                            if ctx and url_queue.qsize() % 10 == 0:
                                ctx.info(f"Queue size: {url_queue.qsize()}")
            
            except Exception as e:
                error_msg = f"Error processing {url}: {str(e)}"
                logger.error(error_msg)
                if ctx:
                    ctx.error(error_msg)
                failed_urls.append({"url": url, "depth": depth, "error": str(e)})
    
    # Start the crawl
    tasks = []
    processed_count = 0
    
    if ctx:
        ctx.info("Starting crawl process...")
    
    while processed_count < max_urls:
        try:
            # Get the next URL from the queue (with timeout)
            try:
                url, depth = await asyncio.wait_for(url_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # No more URLs in the queue
                if not tasks:
                    # If no tasks are running, we're done
                    break
                else:
                    # Otherwise, wait for the running tasks to finish
                    await asyncio.sleep(0.1)
                    continue
            
            # Process the URL
            task = asyncio.create_task(process_url(url, depth))
            tasks.append(task)
            processed_count += 1
            
            # Report progress
            if ctx:
                ctx.report_progress(processed_count, max_urls)
                if processed_count % 10 == 0 or processed_count == 1:
                    ctx.info(f"Processed {processed_count}/{max_urls} URLs")
            
            # Clean up completed tasks
            tasks = [t for t in tasks if not t.done()]
            
            # Wait if we have too many tasks
            if len(tasks) >= max_concurrent * 2:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        except Exception as e:
            error_msg = f"Error in crawl loop: {str(e)}"
            logger.error(error_msg)
            if ctx:
                ctx.error(error_msg)
    
    # Wait for all remaining tasks to complete
    if tasks:
        if ctx:
            ctx.info(f"Waiting for {len(tasks)} remaining tasks to complete...")
        await asyncio.gather(*tasks, return_exceptions=True)
    
    # Calculate statistics
    end_time = time.time()
    total_time = end_time - start_time
    total_content_length = sum(r.get("content_length", 0) for r in processed_results if isinstance(r, dict))
    total_links = sum(r.get("links_count", 0) for r in processed_results if isinstance(r, dict))
    total_chunks_stored = sum(r.get("chunks_stored", 0) for r in processed_results if isinstance(r, dict))
    
    # Group results by depth
    results_by_depth = {}
    for result in processed_results:
        depth = result.get('depth', 0)
        if depth not in results_by_depth:
            results_by_depth[depth] = []
        results_by_depth[depth].append(result)
    
    if ctx:
        ctx.info(f"Recursive crawl completed in {total_time:.2f} seconds")
        ctx.info(f"Processed: {len(processed_results)} URLs")
        ctx.info(f"Failed: {len(failed_urls)} URLs")
        
        for depth in sorted(results_by_depth.keys()):
            ctx.info(f"Depth {depth}: {len(results_by_depth[depth])} URLs")
        
        if store_in_qdrant and collection:
            ctx.info(f"Stored {total_chunks_stored} chunks in collection '{collection}'")
    
    # Return summary
    return {
        "start_url": start_url,
        "max_depth": max_depth,
        "max_urls": max_urls,
        "urls_processed": len(processed_results),
        "urls_failed": len(failed_urls),
        "total_time_seconds": total_time,
        "seconds_per_url": total_time / len(processed_results) if processed_results else 0,
        "total_content_length": total_content_length,
        "total_links_found": total_links,
        "total_chunks_stored": total_chunks_stored,
        "collection": collection if store_in_qdrant else None,
        "depth_statistics": {depth: len(results) for depth, results in results_by_depth.items()},
        "results": [{"url": r["url"], "title": r.get("title", ""), "depth": r.get("depth", 0), 
                    "content_length": r.get("content_length", 0), "links_count": r.get("links_count", 0)} 
                    for r in processed_results if isinstance(r, dict)],
        "failures": failed_urls
    }
