"""Tool for retrieving document version history."""

from typing import Any, Dict, List, Optional
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from pydantic import BaseModel, Field

class GetDocumentHistoryInput(BaseModel):
    """Input schema for the get_document_history tool."""
    
    collection: str = Field(
        description="Collection name where the document is stored"
    )
    document_id: str = Field(
        description="Unique identifier for the document"
    )
    include_content: bool = Field(
        default=False,
        description="Whether to include the full content of each version"
    )
    limit: int = Field(
        default=20,
        description="Maximum number of versions to return"
    )
    offset: int = Field(
        default=0,
        description="Offset for pagination"
    )

class VersionInfo(BaseModel):
    """Schema for individual version information."""
    
    version_id: str = Field(
        description="Unique identifier for this version"
    )
    version_number: int = Field(
        description="Sequential version number"
    )
    timestamp: str = Field(
        description="Timestamp when the version was created"
    )
    is_latest: bool = Field(
        description="Whether this is the latest version"
    )
    version_note: Optional[str] = Field(
        description="Note describing the changes in this version"
    )
    content: Optional[str] = Field(
        description="Full content of this version (if requested)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        description="Additional metadata included with this version"
    )

class GetDocumentHistoryOutput(BaseModel):
    """Output schema for the get_document_history tool."""
    
    document_id: str = Field(
        description="Unique identifier for the document"
    )
    versions: List[VersionInfo] = Field(
        description="List of version information, ordered by version number (descending)"
    )
    total_versions: int = Field(
        description="Total number of versions available"
    )
    success: bool = Field(
        description="Whether the retrieval was successful"
    )
    message: str = Field(
        description="Status message about the operation"
    )

async def get_document_history(
    client: QdrantClient,
    input_data: GetDocumentHistoryInput
) -> GetDocumentHistoryOutput:
    """
    Retrieve version history for a document.
    
    This tool fetches the version history of a document, including timestamps,
    version notes, and optionally the full content of each version.
    
    Args:
        client: QdrantClient instance
        input_data: Input parameters including document ID and history retrieval options
    
    Returns:
        List of versions with their metadata, ordered by version number (descending)
    """
    try:
        # Get the document versions
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=input_data.document_id)
                )
            ]
        )
        
        # Search for existing versions with pagination
        search_result = client.scroll(
            collection_name=input_data.collection,
            scroll_filter=filter_condition,
            limit=input_data.limit,
            offset=input_data.offset,
            with_payload=True,
            with_vectors=False
        )
        
        points = search_result[0]
        
        if not points:
            return GetDocumentHistoryOutput(
                document_id=input_data.document_id,
                versions=[],
                total_versions=0,
                success=False,
                message=f"No versions found for document {input_data.document_id}"
            )
        
        # Count total versions
        total_result = client.count(
            collection_name=input_data.collection,
            count_filter=filter_condition
        )
        
        total_versions = total_result.count
        
        # Parse versions
        versions = []
        for point in points:
            payload = point.payload
            
            # Extract version information
            version_info = VersionInfo(
                version_id=payload.get("version_id", ""),
                version_number=payload.get("version_number", 0),
                timestamp=payload.get("timestamp", ""),
                is_latest=payload.get("is_latest", False),
                version_note=payload.get("version_note", None),
                content=payload.get("content", None) if input_data.include_content else None,
                metadata=payload.get("metadata", None)
            )
            
            versions.append(version_info)
        
        # Sort versions by version number (descending)
        versions.sort(key=lambda x: x.version_number, reverse=True)
        
        return GetDocumentHistoryOutput(
            document_id=input_data.document_id,
            versions=versions,
            total_versions=total_versions,
            success=True,
            message=f"Successfully retrieved {len(versions)} versions of document {input_data.document_id}"
        )
    
    except Exception as e:
        return GetDocumentHistoryOutput(
            document_id=input_data.document_id,
            versions=[],
            total_versions=0,
            success=False,
            message=f"Error retrieving document history: {str(e)}"
        )
