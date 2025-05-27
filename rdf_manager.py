def _prepare_document(self, entity: Dict[str, Any], embedding: List[float]) -> Dict[str, Any]:
    """Prepare a document for Elasticsearch indexing."""
    from datetime import datetime
    
    # Get current timestamp in ISO format
    current_timestamp = datetime.now().isoformat()
    
    doc = {
        "uri": entity.get('uri', ''),
        "type": entity.get('type', ''),
        "local_name": entity.get('local_name', ''),
        "namespace": entity.get('namespace', ''),
        "labels": entity.get('labels', []),
        "comments": entity.get('comments', []),
        "text_content": entity.get('text_content', ''),
        "embedding": embedding,
        "created_at": current_timestamp,  # Use ISO format timestamp
        "updated_at": current_timestamp   # Use ISO format timestamp
    }
    
    optional_fields = [
        'superclasses', 'subclasses', 'related_properties', 
        'property_values', 'metadata', 'types', 'domains', 'ranges'
    ]
    
    for field in optional_fields:
        if field in entity:
            doc[field] = entity[field]
    
    return doc
