"""Single URL crawling tool for Qdrant MCP Server."""
import os
from typing import Any, Dict, List, Optional, Union
import logging
import json
from datetime import datetime
import re
from urllib.parse import urlparse, urljoin

from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)

try:
    import httpx
    from bs4 import BeautifulSoup
    WEB_DEPENDENCIES_AVAILABLE = True
except ImportError:
    logger.warning("Web crawler dependencies not installed. Install with: pip install httpx beautifulsoup4 lxml")
    WEB_DEPENDENCIES_AVAILABLE = False

from ..data_processing.chunk_and_process import chunk_and_process


async def crawl_url(
    url: str,
    collection: Optional[str] = None,
    extract_metadata: bool = True,
    extract_links: bool = True,
    store_in_qdrant: bool = True,
    remove_html_tags: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    user_agent: str = "Qdrant MCP Server Web Crawler",
    timeout: int = 30,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Crawl a single URL and extract its content.

    Args:
        url: The URL to crawl
        collection: The collection to store the extracted content (optional)
        extract_metadata: Whether to extract metadata from the page
        extract_links: Whether to extract links from the page
        store_in_qdrant: Whether to store the extracted content in Qdrant
        remove_html_tags: Whether to remove HTML tags from the content
        chunk_size: Size of each chunk when storing in Qdrant
        chunk_overlap: Overlap between chunks
        user_agent: User agent string to use for the request
        timeout: Timeout in seconds for the request
        ctx: Optional MCP context

    Returns:
        Dictionary containing the extracted content and metadata
    """
    if not WEB_DEPENDENCIES_AVAILABLE:
        error_msg = "Web crawler dependencies not installed. Install with: pip install httpx beautifulsoup4 lxml"
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg}
    
    if ctx:
        ctx.info(f"Crawling URL: {url}")
    
    # Parse URL to ensure it's valid
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            error_msg = f"Invalid URL format: {url}"
            if ctx:
                ctx.error(error_msg)
            return {"error": error_msg}
    except Exception as e:
        error_msg = f"Error parsing URL: {str(e)}"
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg}
    
    # Make the HTTP request
    try:
        headers = {
            "User-Agent": user_agent
        }
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            if ctx:
                ctx.info(f"Sending HTTP request to {url}")
            
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            if ctx:
                ctx.info(f"Received response: {response.status_code} {response.reason_phrase}")
    except httpx.RequestError as e:
        error_msg = f"Request error: {str(e)}"
        logger.error(error_msg)
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg}
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error: {e.response.status_code} {e.response.reason_phrase}"
        logger.error(error_msg)
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg}
    
    # Parse HTML with BeautifulSoup
    try:
        soup = BeautifulSoup(response.text, "lxml")
        
        if ctx:
            ctx.info("Successfully parsed HTML content")
    except Exception as e:
        error_msg = f"Error parsing HTML content: {str(e)}"
        logger.error(error_msg)
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg}
    
    # Extract title and other metadata
    metadata = {}
    if extract_metadata:
        if ctx:
            ctx.info("Extracting metadata from the page")
        
        # Extract title
        title_tag = soup.find('title')
        metadata['title'] = title_tag.text.strip() if title_tag else "No title found"
        
        # Extract meta description
        description_tag = soup.find('meta', attrs={'name': 'description'})
        if description_tag and 'content' in description_tag.attrs:
            metadata['description'] = description_tag['content']
        
        # Extract meta keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag and 'content' in keywords_tag.attrs:
            metadata['keywords'] = keywords_tag['content']
        
        # Extract published date if available
        published_tag = soup.find('meta', attrs={'property': 'article:published_time'})
        if published_tag and 'content' in published_tag.attrs:
            metadata['published_date'] = published_tag['content']
        
        # Extract author if available
        author_tag = soup.find('meta', attrs={'name': 'author'})
        if author_tag and 'content' in author_tag.attrs:
            metadata['author'] = author_tag['content']
        
        # Add URL and crawled time
        metadata['url'] = url
        metadata['crawled_at'] = datetime.now().isoformat()
    
    # Extract main content
    content = ""
    
    # Try to find main content container
    main_content = None
    for selector in [
        'article', 'main', '.content', '#content', '.post', '.article',
        '.entry-content', '.post-content', '.article-content'
    ]:
        main_content = soup.select_one(selector)
        if main_content:
            break
    
    # If main content container found, extract from it
    if main_content:
        if remove_html_tags:
            content = main_content.get_text(separator='\n', strip=True)
        else:
            content = str(main_content)
    else:
        # No specific content container found, extract from body
        body = soup.find('body')
        if body:
            if remove_html_tags:
                # Remove script, style, and other non-content elements
                for element in body(['script', 'style', 'head', 'header', 'footer', 'nav']):
                    element.extract()
                content = body.get_text(separator='\n', strip=True)
            else:
                content = str(body)
    
    # Clean up the content
    if remove_html_tags:
        # Remove multiple newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        # Remove multiple spaces
        content = re.sub(r' {2,}', ' ', content)
    
    # Extract links if needed
    links = []
    if extract_links:
        if ctx:
            ctx.info("Extracting links from the page")
        
        # Find all <a> tags with href attributes
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']
            if href and not href.startswith('#'):  # Skip anchors
                # Convert relative URLs to absolute
                if not href.startswith(('http://', 'https://', 'ftp://', 'mailto:')):
                    href = urljoin(url, href)
                
                # Extract link text
                link_text = link_tag.get_text(strip=True) or "No text"
                
                links.append({
                    "url": href,
                    "text": link_text
                })
        
        if ctx:
            ctx.info(f"Extracted {len(links)} links")
    
    # Store in Qdrant if requested
    points_stored = 0
    if store_in_qdrant and collection and content:
        if ctx:
            ctx.info(f"Storing content in Qdrant collection: {collection}")
        
        # Prepare metadata for storage
        storage_metadata = {
            **metadata,
            "content_type": "webpage",
            "links_count": len(links)
        }
        
        # Store the extracted links in metadata if there aren't too many
        if len(links) <= 20:  # Limit to avoid metadata size issues
            storage_metadata["links"] = links
        
        # Use the chunk_and_process tool to store in Qdrant
        try:
            result = await chunk_and_process(
                text=content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                collection=collection,
                metadata=storage_metadata,
                ctx=ctx
            )
            
            if isinstance(result, dict) and "chunks" in result:
                points_stored = len(result["chunks"])
                if ctx:
                    ctx.info(f"Successfully stored {points_stored} chunks in Qdrant")
        except Exception as e:
            error_msg = f"Error storing content in Qdrant: {str(e)}"
            logger.error(error_msg)
            if ctx:
                ctx.error(error_msg)
    
    # Return results
    return {
        "url": url,
        "title": metadata.get('title', ''),
        "metadata": metadata,
        "content_length": len(content),
        "links_count": len(links),
        "links": links[:10] if len(links) > 10 else links,  # Limit links in response
        "chunks_stored": points_stored,
        "collection": collection if store_in_qdrant else None
    }
