"""Tool for updating documents while maintaining version history."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import uuid

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct
from pydantic import BaseModel, Field

class VersionDocumentInput(BaseModel):
    """Input schema for the version_document tool."""
    
    collection: str = Field(
        description="Collection name to store the document in"
    )
    document_id: str = Field(
        description="Unique identifier for the document"
    )
    content: str = Field(
        description="Updated document content"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata to include with the document"
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use for embedding (optional)"
    )
    version_note: Optional[str] = Field(
        default=None,
        description="Note describing the changes in this version"
    )

class VersionDocumentOutput(BaseModel):
    """Output schema for the version_document tool."""
    
    document_id: str = Field(
        description="Unique identifier for the document"
    )
    version_id: str = Field(
        description="Unique identifier for this version"
    )
    version_number: int = Field(
        description="Sequential version number"
    )
    timestamp: str = Field(
        description="Timestamp when the version was created"
    )
    success: bool = Field(
        description="Whether the versioning was successful"
    )
    message: str = Field(
        description="Status message about the versioning operation"
    )

async def version_document(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    input_data: VersionDocumentInput
) -> VersionDocumentOutput:
    """
    Update a document while maintaining version history.
    
    This tool allows updating a document's content while preserving its previous versions,
    making it possible to track changes over time and revert to earlier versions if needed.
    
    Args:
        client: QdrantClient instance
        embedding_model: TextEmbedding model to use for generating embeddings
        input_data: Input parameters including document content and metadata
    
    Returns:
        Information about the newly created version
    """
    try:
        # Get existing document and its version history
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=input_data.document_id)
                )
            ]
        )
        
        # Search for existing versions
        search_result = client.scroll(
            collection_name=input_data.collection,
            scroll_filter=filter_condition,
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        points = search_result[0]
        
        # Determine next version number
        if points:
            # Find the highest version number
            highest_version = max(
                (point.payload.get("version_number", 0) for point in points), 
                default=0
            )
            next_version = highest_version + 1
        else:
            # This is the first version
            next_version = 1
        
        # Create version metadata
        version_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Create payload for the new version
        payload = {
            "document_id": input_data.document_id,
            "content": input_data.content,
            "version_id": version_id,
            "version_number": next_version,
            "timestamp": timestamp,
            "is_latest": True
        }
        
        # Add optional metadata
        if input_data.metadata:
            payload["metadata"] = input_data.metadata
            
        if input_data.version_note:
            payload["version_note"] = input_data.version_note
        
        # Generate embedding for the content
        embedding = embedding_model.embed(input_data.content)[0].tolist()
        
        # Update previous version to mark it as not latest
        if points:
            for point in points:
                if point.payload.get("is_latest", False):
                    client.update_payload(
                        collection_name=input_data.collection,
                        payload={"is_latest": False},
                        points=[point.id]
                    )
        
        # Create a new point for this version
        client.upsert(
            collection_name=input_data.collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        
        return VersionDocumentOutput(
            document_id=input_data.document_id,
            version_id=version_id,
            version_number=next_version,
            timestamp=timestamp,
            success=True,
            message=f"Successfully created version {next_version} of document {input_data.document_id}"
        )
    
    except Exception as e:
        return VersionDocumentOutput(
            document_id=input_data.document_id,
            version_id="",
            version_number=0,
            timestamp=datetime.utcnow().isoformat(),
            success=False,
            message=f"Error creating document version: {str(e)}"
        )
