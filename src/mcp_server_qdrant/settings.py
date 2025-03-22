import os
from typing import Optional, Dict, Any

from pydantic import Field
from pydantic_settings import BaseSettings

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType

# Default tool descriptions
DEFAULT_TOOL_STORE_DESCRIPTION = (
    "Keep the memory for later use, when you are asked to remember something."
)
DEFAULT_TOOL_FIND_DESCRIPTION = (
    "Look up memories in Qdrant. Use this tool when you need to: \n"
    " - Find memories by their content \n"
    " - Access memories for further analysis \n"
    " - Get some personal information about the user"
)

# Default descriptions for custom tools
DEFAULT_TOOL_DESCRIPTIONS = {
    # Search tools
    "nlq_search": "Search Qdrant using natural language",
    "hybrid_search": "Perform hybrid vector and keyword search",
    "multi_vector_search": "Search using multiple query vectors with weights",
    
    # Collection management tools
    "create_collection": "Create a new vector collection with specified parameters",
    "migrate_collection": "Migrate data between collections with optional transformations",
    
    # Data processing tools
    "batch_embed": "Generate embeddings for multiple texts in batch",
    "chunk_and_process": "Split text into chunks, generate embeddings, and store in Qdrant",
    
    # Advanced query tools
    "semantic_router": "Route a query to the most semantically similar destination",
    "decompose_query": "Decompose a complex query into multiple simpler subqueries",
    
    # Analytics tools
    "analyze_collection": "Analyze a Qdrant collection to extract statistics and schema information",
    "benchmark_query": "Benchmark query performance by running multiple iterations",
    
    # Document processing tools
    "index_document": "Fetch, process, and index a document from a URL",
    "process_pdf": "Process a PDF file, extract content, and optionally store in Qdrant",
    
    # Metadata tools
    "extract_metadata": "Extract structured metadata from document text including entities, patterns, and statistics",
    
    # Visualization tools
    "visualize_vectors": "Generate 2D or 3D projections of vectors for visualization",
    "cluster_visualization": "Visualize semantic clusters within collections and extract topics",
    
    # Web crawling tools
    "crawl_url": "Crawl a single URL and store content in Qdrant",
    "batch_crawl": "Process multiple URLs in batch and store in Qdrant",
    "recursive_crawl": "Recursively crawl a website and store content in Qdrant",
    "sitemap_extract": "Extract and process URLs from a sitemap.xml file",
    
    # Social Media Connector tools - Twitter
    "setup_twitter_connector": "Set up a Twitter connector to automatically fetch and index tweets from a specific user",
    "check_twitter_updates": "Manually check for new tweets from a configured Twitter connector",
    "list_twitter_connectors": "List all active Twitter connectors",
    "delete_twitter_connector": "Delete a Twitter connector",
    
    # Social Media Connector tools - Mastodon
    "setup_mastodon_connector": "Set up a Mastodon connector to automatically fetch and index posts from a specific user",
    "check_mastodon_updates": "Manually check for new posts from a configured Mastodon connector",
    "list_mastodon_connectors": "List all active Mastodon connectors",
    "delete_mastodon_connector": "Delete a Mastodon connector"
}


class ToolSettings(BaseSettings):
    """
    Configuration for all the tools.
    """

    tool_store_description: str = Field(
        default=DEFAULT_TOOL_STORE_DESCRIPTION,
        validation_alias="TOOL_STORE_DESCRIPTION",
    )
    tool_find_description: str = Field(
        default=DEFAULT_TOOL_FIND_DESCRIPTION,
        validation_alias="TOOL_FIND_DESCRIPTION",
    )
    
    # Custom tool descriptions with environment variable overrides
    nlq_search_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["nlq_search"],
        validation_alias="NLQ_SEARCH_DESCRIPTION",
    )
    hybrid_search_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["hybrid_search"],
        validation_alias="HYBRID_SEARCH_DESCRIPTION",
    )
    multi_vector_search_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["multi_vector_search"],
        validation_alias="MULTI_VECTOR_SEARCH_DESCRIPTION",
    )
    create_collection_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["create_collection"],
        validation_alias="CREATE_COLLECTION_DESCRIPTION",
    )
    migrate_collection_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["migrate_collection"],
        validation_alias="MIGRATE_COLLECTION_DESCRIPTION",
    )
    batch_embed_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["batch_embed"],
        validation_alias="BATCH_EMBED_DESCRIPTION",
    )
    chunk_and_process_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["chunk_and_process"],
        validation_alias="CHUNK_AND_PROCESS_DESCRIPTION",
    )
    semantic_router_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["semantic_router"],
        validation_alias="SEMANTIC_ROUTER_DESCRIPTION",
    )
    decompose_query_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["decompose_query"],
        validation_alias="DECOMPOSE_QUERY_DESCRIPTION",
    )
    analyze_collection_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["analyze_collection"],
        validation_alias="ANALYZE_COLLECTION_DESCRIPTION",
    )
    benchmark_query_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["benchmark_query"],
        validation_alias="BENCHMARK_QUERY_DESCRIPTION",
    )
    index_document_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["index_document"],
        validation_alias="INDEX_DOCUMENT_DESCRIPTION",
    )
    process_pdf_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["process_pdf"],
        validation_alias="PROCESS_PDF_DESCRIPTION",
    )
    
    # Metadata tool descriptions
    extract_metadata_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["extract_metadata"],
        validation_alias="EXTRACT_METADATA_DESCRIPTION",
    )
    
    # Visualization tool descriptions
    visualize_vectors_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["visualize_vectors"],
        validation_alias="VISUALIZE_VECTORS_DESCRIPTION",
    )
    cluster_visualization_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["cluster_visualization"],
        validation_alias="CLUSTER_VISUALIZATION_DESCRIPTION",
    )
    
    # Web crawling tool descriptions
    crawl_url_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["crawl_url"],
        validation_alias="CRAWL_URL_DESCRIPTION",
    )
    batch_crawl_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["batch_crawl"],
        validation_alias="BATCH_CRAWL_DESCRIPTION",
    )
    recursive_crawl_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["recursive_crawl"],
        validation_alias="RECURSIVE_CRAWL_DESCRIPTION",
    )
    sitemap_extract_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["sitemap_extract"],
        validation_alias="SITEMAP_EXTRACT_DESCRIPTION",
    )
    
    # Social Media Connector tool descriptions - Twitter
    setup_twitter_connector_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["setup_twitter_connector"],
        validation_alias="SETUP_TWITTER_CONNECTOR_DESCRIPTION",
    )
    check_twitter_updates_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["check_twitter_updates"],
        validation_alias="CHECK_TWITTER_UPDATES_DESCRIPTION",
    )
    list_twitter_connectors_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["list_twitter_connectors"],
        validation_alias="LIST_TWITTER_CONNECTORS_DESCRIPTION",
    )
    delete_twitter_connector_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["delete_twitter_connector"],
        validation_alias="DELETE_TWITTER_CONNECTOR_DESCRIPTION",
    )
    
    # Social Media Connector tool descriptions - Mastodon
    setup_mastodon_connector_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["setup_mastodon_connector"],
        validation_alias="SETUP_MASTODON_CONNECTOR_DESCRIPTION",
    )
    check_mastodon_updates_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["check_mastodon_updates"],
        validation_alias="CHECK_MASTODON_UPDATES_DESCRIPTION",
    )
    list_mastodon_connectors_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["list_mastodon_connectors"],
        validation_alias="LIST_MASTODON_CONNECTORS_DESCRIPTION",
    )
    delete_mastodon_connector_description: str = Field(
        default=DEFAULT_TOOL_DESCRIPTIONS["delete_mastodon_connector"],
        validation_alias="DELETE_MASTODON_CONNECTOR_DESCRIPTION",
    )


class EmbeddingProviderSettings(BaseSettings):
    """
    Configuration for the embedding provider.
    """

    provider_type: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.FASTEMBED,
        validation_alias="EMBEDDING_PROVIDER",
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )


class QdrantSettings(BaseSettings):
    """
    Configuration for the Qdrant connector.
    """

    location: Optional[str] = Field(default=None, validation_alias="QDRANT_URL")
    api_key: Optional[str] = Field(default=None, validation_alias="QDRANT_API_KEY")
    collection_name: str = Field(validation_alias="COLLECTION_NAME")
    local_path: Optional[str] = Field(
        default=None, validation_alias="QDRANT_LOCAL_PATH"
    )
    
    # Default values for collection creation
    default_vector_size: int = Field(
        default=384, validation_alias="DEFAULT_VECTOR_SIZE"
    )
    default_distance: str = Field(
        default="Cosine", validation_alias="DEFAULT_DISTANCE"
    )
    
    # PDF processing settings
    pdf_extract_tables: bool = Field(
        default=False, validation_alias="PDF_EXTRACT_TABLES"
    )
    pdf_extract_images: bool = Field(
        default=False, validation_alias="PDF_EXTRACT_IMAGES"
    )

    def get_qdrant_location(self) -> str:
        """
        Get the Qdrant location, either the URL or the local path.
        """
        return self.location or self.local_path
