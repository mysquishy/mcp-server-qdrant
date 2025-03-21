"""Sitemap extraction tool for Qdrant MCP Server."""
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
    import xml.etree.ElementTree as ET
    WEB_DEPENDENCIES_AVAILABLE = True
except ImportError:
    logger.warning("Web crawler dependencies not installed. Install with: pip install httpx beautifulsoup4 lxml")
    WEB_DEPENDENCIES_AVAILABLE = False


async def sitemap_extract(
    url: str,
    follow_sitemapindex: bool = True,
    limit: Optional[int] = None,
    include_lastmod: bool = True,
    include_priority: bool = True,
    include_changefreq: bool = True,
    filter_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    timeout: int = 30,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Extract URLs from a sitemap.xml file.

    Args:
        url: URL of the sitemap.xml file or the base website URL
        follow_sitemapindex: Whether to follow sitemap index references
        limit: Maximum number of URLs to extract (None for no limit)
        include_lastmod: Include last modification dates
        include_priority: Include priority values
        include_changefreq: Include change frequency values
        filter_patterns: List of regex patterns to include URLs (None for no filtering)
        exclude_patterns: List of regex patterns to exclude URLs
        timeout: Timeout in seconds for HTTP requests
        ctx: Optional MCP context

    Returns:
        Dictionary containing extracted URLs and metadata
    """
    if not WEB_DEPENDENCIES_AVAILABLE:
        error_msg = "Web crawler dependencies not installed. Install with: pip install httpx beautifulsoup4 lxml"
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg}
    
    if ctx:
        ctx.info(f"Extracting URLs from sitemap: {url}")
    
    # Normalize URL - if it's a base domain, append sitemap.xml
    parsed_url = urlparse(url)
    if not url.endswith('.xml') and not url.endswith('sitemap'):
        # This might be a base URL, try to find sitemap.xml
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        url = urljoin(base_url, "sitemap.xml")
        if ctx:
            ctx.info(f"Normalized URL to: {url}")
    
    # Compile regex patterns if provided
    compiled_filters = None
    compiled_excludes = None
    
    if filter_patterns:
        compiled_filters = [re.compile(pattern) for pattern in filter_patterns]
        if ctx:
            ctx.info(f"Using {len(compiled_filters)} URL inclusion filters")
    
    if exclude_patterns:
        compiled_excludes = [re.compile(pattern) for pattern in exclude_patterns]
        if ctx:
            ctx.info(f"Using {len(compiled_excludes)} URL exclusion filters")
    
    # Fetch the sitemap
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if ctx:
                ctx.info(f"Fetching sitemap from: {url}")
            
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            
            if ctx:
                ctx.info(f"Received response: {response.status_code} {response.reason_phrase}")
                ctx.info(f"Content type: {response.headers.get('content-type', 'unknown')}")
    except httpx.RequestError as e:
        error_msg = f"Request error fetching sitemap: {str(e)}"
        logger.error(error_msg)
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg}
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error fetching sitemap: {e.response.status_code} {e.response.reason_phrase}"
        logger.error(error_msg)
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg}
    
    # Parse the sitemap XML
    try:
        # Try to parse as XML first
        root = ET.fromstring(response.text)
        
        # Determine if this is a sitemap index or a regular sitemap
        ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # Extract URLs
        urls = []
        sitemap_urls = []
        
        # Check if this is a sitemap index
        is_sitemap_index = root.findall('.//sm:sitemapindex', ns) or root.tag.endswith('sitemapindex')
        
        if is_sitemap_index:
            if ctx:
                ctx.info("Found a sitemap index file")
            
            # Extract sitemap URLs
            for sitemap in root.findall('.//sm:sitemap', ns) or root.findall('.//sitemap'):
                loc_elem = sitemap.find('.//sm:loc', ns) or sitemap.find('loc')
                if loc_elem is not None and loc_elem.text:
                    sitemap_url = loc_elem.text.strip()
                    sitemap_urls.append(sitemap_url)
            
            if ctx:
                ctx.info(f"Found {len(sitemap_urls)} sitemap URLs in the index")
            
            # If following sitemap index references, fetch each sitemap
            if follow_sitemapindex and sitemap_urls:
                for i, sitemap_url in enumerate(sitemap_urls):
                    if ctx:
                        ctx.info(f"Processing sitemap {i+1}/{len(sitemap_urls)}: {sitemap_url}")
                        ctx.report_progress(i, len(sitemap_urls))
                    
                    # Recursively call sitemap_extract for each sitemap URL
                    try:
                        sub_result = await sitemap_extract(
                            url=sitemap_url,
                            follow_sitemapindex=False,  # Avoid infinite recursion
                            limit=None,  # We'll apply the limit to the combined results
                            include_lastmod=include_lastmod,
                            include_priority=include_priority,
                            include_changefreq=include_changefreq,
                            filter_patterns=filter_patterns,
                            exclude_patterns=exclude_patterns,
                            timeout=timeout,
                            ctx=ctx
                        )
                        
                        if 'urls' in sub_result and isinstance(sub_result['urls'], list):
                            urls.extend(sub_result['urls'])
                            if ctx:
                                ctx.info(f"Added {len(sub_result['urls'])} URLs from sub-sitemap")
                    except Exception as e:
                        if ctx:
                            ctx.warning(f"Error processing sub-sitemap {sitemap_url}: {str(e)}")
        else:
            # Regular sitemap with URLs
            if ctx:
                ctx.info("Processing regular sitemap with URLs")
            
            for url_elem in root.findall('.//sm:url', ns) or root.findall('.//url'):
                # Extract URL location (required)
                loc_elem = url_elem.find('sm:loc', ns) or url_elem.find('loc')
                if loc_elem is not None and loc_elem.text:
                    page_url = loc_elem.text.strip()
                    
                    # Apply filters if specified
                    if compiled_filters and not any(pattern.search(page_url) for pattern in compiled_filters):
                        continue
                    
                    if compiled_excludes and any(pattern.search(page_url) for pattern in compiled_excludes):
                        continue
                    
                    # Create URL entry
                    url_entry = {"url": page_url}
                    
                    # Extract optional metadata
                    if include_lastmod:
                        lastmod_elem = url_elem.find('sm:lastmod', ns) or url_elem.find('lastmod')
                        if lastmod_elem is not None and lastmod_elem.text:
                            url_entry["lastmod"] = lastmod_elem.text.strip()
                    
                    if include_changefreq:
                        changefreq_elem = url_elem.find('sm:changefreq', ns) or url_elem.find('changefreq')
                        if changefreq_elem is not None and changefreq_elem.text:
                            url_entry["changefreq"] = changefreq_elem.text.strip()
                    
                    if include_priority:
                        priority_elem = url_elem.find('sm:priority', ns) or url_elem.find('priority')
                        if priority_elem is not None and priority_elem.text:
                            try:
                                url_entry["priority"] = float(priority_elem.text.strip())
                            except ValueError:
                                pass
                    
                    urls.append(url_entry)
            
            if ctx:
                ctx.info(f"Extracted {len(urls)} URLs from sitemap")
        
        # Apply limit if specified
        if limit is not None and urls:
            urls = urls[:limit]
            if ctx:
                ctx.info(f"Limited to {len(urls)} URLs as requested")
        
        # Return results
        return {
            "urls": urls,
            "total_urls": len(urls),
            "is_sitemap_index": is_sitemap_index,
            "sitemaps_in_index": sitemap_urls if is_sitemap_index else [],
            "source_url": url
        }
    
    except ET.ParseError:
        error_msg = "Failed to parse XML sitemap"
        logger.error(error_msg)
        if ctx:
            ctx.error(error_msg)
            ctx.info("Attempting to find sitemap links in HTML")
        
        # If XML parsing fails, try to find sitemap links in HTML
        try:
            soup = BeautifulSoup(response.text, "lxml")
            sitemap_links = []
            
            # Look for <link rel="sitemap" href="..."> tags
            for link in soup.find_all("link", rel="sitemap"):
                if "href" in link.attrs:
                    sitemap_links.append(link["href"])
            
            # Look for sitemap.xml mentions in robots.txt format
            for line in response.text.splitlines():
                if "sitemap:" in line.lower():
                    parts = line.split(":", 1)
                    if len(parts) > 1 and parts[1].strip():
                        sitemap_links.append(parts[1].strip())
            
            if sitemap_links:
                if ctx:
                    ctx.info(f"Found {len(sitemap_links)} potential sitemap URLs in HTML/text")
                
                # Return the found sitemap links for further processing
                return {
                    "urls": [],
                    "total_urls": 0,
                    "is_sitemap_index": False,
                    "sitemaps_found_in_html": sitemap_links,
                    "source_url": url,
                    "note": "No valid XML sitemap found, but potential sitemap URLs were found in HTML/text"
                }
            
            return {
                "error": "Failed to parse XML sitemap and no sitemap links found in HTML",
                "urls": [],
                "total_urls": 0
            }
            
        except Exception as html_e:
            error_msg = f"Failed to parse XML sitemap and HTML processing failed: {str(html_e)}"
            logger.error(error_msg)
            if ctx:
                ctx.error(error_msg)
            return {"error": error_msg, "urls": [], "total_urls": 0}
    
    except Exception as e:
        error_msg = f"Error processing sitemap: {str(e)}"
        logger.error(error_msg)
        if ctx:
            ctx.error(error_msg)
        return {"error": error_msg, "urls": [], "total_urls": 0}
