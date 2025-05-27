"""
Enhanced Elasticsearch Vector Store with text-embedding-3-large support.
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from elasticsearch import Elasticsearch
from langchain_openai import AzureOpenAIEmbeddings
from app.utils.auth_helper import get_azure_token

logger = logging.getLogger(__name__)

class EnhancedElasticsearchVectorStore:
    """
    Enhanced vector store using text-embedding-3-large and Elasticsearch.
    """
    
    def __init__(self, 
                 hosts: List[str] = None,
                 index_name: str = "rdf_knowledge_graph",
                 embedding_model: str = "text-embedding-3-large",
                 embedding_dimensions: int = 3072):
        """
        Initialize Enhanced Elasticsearch Vector Store.
        
        Args:
            hosts: Elasticsearch host addresses
            index_name: Name of the Elasticsearch index
            embedding_model: Azure OpenAI embedding model name
            embedding_dimensions: Dimension of the embedding vectors
        """
        self.hosts = hosts or ["localhost:9200"]
        self.index_name = index_name
        self.embedding_model_name = embedding_model
        self.embedding_dimensions = embedding_dimensions
        
        # Initialize Azure OpenAI embeddings with Entra ID authentication
        self.embedding_model = self._setup_embedding_model(embedding_model, embedding_dimensions)
        
        # Initialize Elasticsearch client
        try:
            # Configure Elasticsearch connection
            es_config = {
                "hosts": self.hosts,
                "timeout": 30,
                "max_retries": 3,
                "retry_on_timeout": True
            }
            
            # Add authentication if provided
            if os.getenv("ELASTICSEARCH_USERNAME") and os.getenv("ELASTICSEARCH_PASSWORD"):
                es_config["basic_auth"] = (
                    os.getenv("ELASTICSEARCH_USERNAME"),
                    os.getenv("ELASTICSEARCH_PASSWORD")
                )
            
            # Add SSL configuration if needed
            if os.getenv("ELASTICSEARCH_USE_SSL", "false").lower() == "true":
                es_config["use_ssl"] = True
                es_config["verify_certs"] = True
            
            self.es = Elasticsearch(**es_config)
            
            # Test connection
            if self.es.ping():
                logger.info(f"Connected to Elasticsearch at {self.hosts}")
            else:
                raise ConnectionError("Could not connect to Elasticsearch")
                
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
        
        # Create index if it doesn't exist
        self._create_index()
    
    def _create_index(self):
        """Create the Elasticsearch index with proper mapping for text-embedding-3-large."""
        try:
            if not self.es.indices.exists(index=self.index_name):
                mapping = {
                    "mappings": {
                        "properties": {
                            # Entity identification
                            "uri": {
                                "type": "keyword",
                                "index": True
                            },
                            "type": {
                                "type": "keyword",
                                "index": True
                            },
                            "local_name": {
                                "type": "text",
                                "analyzer": "standard",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword",
                                        "ignore_above": 256
                                    }
                                }
                            },
                            "namespace": {
                                "type": "keyword",
                                "index": True
                            },
                            
                            # Text content for search
                            "labels": {
                                "type": "text",
                                "analyzer": "standard"
                            },
                            "comments": {
                                "type": "text",
                                "analyzer": "standard"
                            },
                            "text_content": {
                                "type": "text",
                                "analyzer": "standard"
                            },
                            
                            # Structured data
                            "metadata": {
                                "type": "object",
                                "enabled": True
                            },
                            
                            # Relationships
                            "superclasses": {
                                "type": "keyword"
                            },
                            "subclasses": {
                                "type": "keyword"
                            },
                            "related_properties": {
                                "type": "nested",
                                "properties": {
                                    "uri": {"type": "keyword"},
                                    "local_name": {"type": "keyword"},
                                    "type": {"type": "keyword"}
                                }
                            },
                            "property_values": {
                                "type": "object",
                                "enabled": True
                            },
                            
                            # Vector embedding for text-embedding-3-large
                            "embedding": {
                                "type": "dense_vector",
                                "dims": self.embedding_dimensions,
                                "index": True,
                                "similarity": "cosine"
                            },
                            
                            # Timestamps
                            "created_at": {
                                "type": "date"
                            },
                            "updated_at": {
                                "type": "date"
                            }
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 1,
                        "index": {
                            "knn": True,
                            "knn.algo_param.ef_search": 100
                        }
                    }
                }
                
                self.es.indices.create(index=self.index_name, body=mapping)
                logger.info(f"Created Elasticsearch index: {self.index_name}")
            else:
                logger.info(f"Elasticsearch index already exists: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error creating Elasticsearch index: {e}")
            raise
    
    def _setup_embedding_model(self, embedding_model: str, embedding_dimensions: int) -> AzureOpenAIEmbeddings:
        """
        Set up Azure OpenAI embeddings with Entra ID authentication.
        
        Args:
            embedding_model: Name of the embedding model
            embedding_dimensions: Vector dimensions
            
        Returns:
            Configured AzureOpenAIEmbeddings instance
        """
        try:
            # Get Azure credentials from environment
            tenant_id = os.getenv("AZURE_TENANT_ID")
            client_id = os.getenv("AZURE_CLIENT_ID")
            client_secret = os.getenv("AZURE_CLIENT_SECRET")
            azure_endpoint = os.getenv("AZURE_ENDPOINT")
            
            if not all([tenant_id, client_id, client_secret, azure_endpoint]):
                raise ValueError("Missing Azure credentials. Check AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, and AZURE_ENDPOINT")
            
            # Create token provider function using auth_helper
            def token_provider():
                """Token provider for Azure OpenAI embeddings."""
                token = get_azure_token(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret,
                    scope="https://cognitiveservices.azure.com/.default"
                )
                if not token:
                    raise ValueError("Failed to obtain Azure token for embeddings")
                return token
            
            # Test token provider
            test_token = token_provider()
            logger.info("✓ Azure token obtained successfully for embeddings")
            
            # Initialize embeddings with token provider
            embeddings = AzureOpenAIEmbeddings(
                model=embedding_model,
                dimensions=embedding_dimensions,
                azure_endpoint=azure_endpoint,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                azure_ad_token_provider=token_provider,  # Use token provider instead of API key
                chunk_size=100,  # Process in batches for efficiency
                max_retries=3,
                request_timeout=60,
                show_progress_bar=True,  # Show progress for large batches
                skip_empty=True  # Skip empty strings
            )
            
            logger.info(f"✓ Azure OpenAI Embeddings initialized with model: {embedding_model}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to setup embedding model: {e}")
            raise
    
    def add_entities(self, entities: List[Dict[str, Any]], batch_size: int = 50) -> bool:
        """
        Add entities to the vector store with embeddings in batches.
        
        Args:
            entities: List of entity dictionaries
            batch_size: Number of entities to process in each batch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not entities:
                logger.warning("No entities provided for indexing")
                return True
            
            total_entities = len(entities)
            successful_indexes = 0
            
            # Process entities in batches
            for i in range(0, total_entities, batch_size):
                batch = entities[i:i + batch_size]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(total_entities + batch_size - 1)//batch_size}")
                
                # Prepare texts for embedding
                texts = []
                valid_entities = []
                
                for entity in batch:
                    text_content = entity.get('text_content', '')
                    if text_content and text_content.strip():
                        texts.append(text_content)
                        valid_entities.append(entity)
                    else:
                        logger.warning(f"Skipping entity {entity.get('uri', 'unknown')} - no text content")
                
                if not texts:
                    continue
                
                # Generate embeddings for the batch
                try:
                    embeddings = self.embedding_model.embed_documents(texts)
                    logger.info(f"Generated {len(embeddings)} embeddings for batch")
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch: {e}")
                    continue
                
                # Prepare documents for indexing
                actions = []
                for entity, embedding in zip(valid_entities, embeddings):
                    # Clean and prepare the document
                    doc = self._prepare_document(entity, embedding)
                    
                    action = {
                        "_index": self.index_name,
                        "_id": entity['uri'],
                        "_source": doc
                    }
                    actions.append(action)
                
                # Bulk index documents
                if actions:
                    try:
                        from elasticsearch.helpers import bulk
                        success_count, failed_items = bulk(
                            self.es, 
                            actions, 
                            chunk_size=batch_size,
                            request_timeout=60
                        )
                        successful_indexes += success_count
                        
                        if failed_items:
                            logger.warning(f"Failed to index {len(failed_items)} entities in batch")
                        
                    except Exception as e:
                        logger.error(f"Error bulk indexing batch: {e}")
                        continue
            
            logger.info(f"Successfully indexed {successful_indexes}/{total_entities} entities")
            return successful_indexes > 0
            
        except Exception as e:
            logger.error(f"Error adding entities to vector store: {e}")
            return False
    
    def _prepare_document(self, entity: Dict[str, Any], embedding: List[float]) -> Dict[str, Any]:
        """Prepare a document for Elasticsearch indexing."""
        # Create a clean document
        doc = {
            "uri": entity.get('uri', ''),
            "type": entity.get('type', ''),
            "local_name": entity.get('local_name', ''),
            "namespace": entity.get('namespace', ''),
            "labels": entity.get('labels', []),
            "comments": entity.get('comments', []),
            "text_content": entity.get('text_content', ''),
            "embedding": embedding,
            "created_at": "now",
            "updated_at": "now"
        }
        
        # Add optional fields if they exist
        optional_fields = [
            'superclasses', 'subclasses', 'related_properties', 
            'property_values', 'metadata', 'types', 'domains', 'ranges'
        ]
        
        for field in optional_fields:
            if field in entity:
                doc[field] = entity[field]
        
        return doc
    
    def search_similar(self, 
                      query_text: str, 
                      top_k: int = 10,
                      entity_types: Optional[List[str]] = None,
                      min_score: float = 0.6,
                      include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Search for entities similar to the query text using hybrid search.
        
        Args:
            query_text: Text to search for
            top_k: Number of top results to return
            entity_types: Filter by entity types
            min_score: Minimum similarity score threshold
            include_metadata: Whether to include full metadata
            
        Returns:
            List of similar entities with scores
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedding_model.embed_query(query_text)
            
            # Build hybrid search query (vector + text)
            query = {
                "size": top_k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                        "params": {"query_vector": query_embedding}
                                    },
                                    "boost": 2.0
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": ["text_content^2", "labels^1.5", "local_name^1.5", "comments"],
                                    "type": "best_fields",
                                    "boost": 1.0
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "min_score": min_score
            }
            
            # Add entity type filter if specified
            if entity_types:
                if "filter" not in query["query"]["bool"]:
                    query["query"]["bool"]["filter"] = []
                query["query"]["bool"]["filter"].append({
                    "terms": {"type": entity_types}
                })
            
            # Execute search
            response = self.es.search(index=self.index_name, body=query)
            
            results = []
            for hit in response['hits']['hits']:
                score = hit['_score']
                result = hit['_source'].copy()
                result['similarity_score'] = score
                result['search_type'] = 'hybrid'
                
                # Remove embedding from result if not needed
                if not include_metadata and 'embedding' in result:
                    del result['embedding']
                
                results.append(result)
            
            logger.info(f"Found {len(results)} similar entities for query: '{query_text}'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching for similar entities: {e}")
            return []
    
    def search_by_keywords(self, 
                          keywords: List[str], 
                          top_k: int = 10,
                          entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search entities by keywords using text search.
        
        Args:
            keywords: List of keywords to search for
            top_k: Number of results to return
            entity_types: Filter by entity types
            
        Returns:
            List of matching entities
        """
        try:
            query = {
                "size": top_k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "terms": {
                                    "local_name.keyword": keywords,
                                    "boost": 3.0
                                }
                            },
                            {
                                "terms": {
                                    "labels": keywords,
                                    "boost": 2.0
                                }
                            },
                            {
                                "multi_match": {
                                    "query": " ".join(keywords),
                                    "fields": ["text_content", "comments"],
                                    "boost": 1.0
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                }
            }
            
            # Add entity type filter if specified
            if entity_types:
                query["query"]["bool"]["filter"] = [
                    {"terms": {"type": entity_types}}
                ]
            
            response = self.es.search(index=self.index_name, body=query)
            
            results = []
            for hit in response['hits']['hits']:
                result = hit['_source'].copy()
                result['search_score'] = hit['_score']
                result['search_type'] = 'keyword'
                
                # Remove embedding from result
                if 'embedding' in result:
                    del result['embedding']
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching by keywords: {e}")
            return []
    
    def get_entity(self, uri: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific entity by URI.
        
        Args:
            uri: URI of the entity
            
        Returns:
            Entity data or None if not found
        """
        try:
            response = self.es.get(index=self.index_name, id=uri)
            result = response['_source']
            
            # Remove embedding for cleaner output
            if 'embedding' in result:
                del result['embedding']
            
            return result
        except Exception:
            logger.debug(f"Entity not found: {uri}")
            return None
    
    def get_entities_by_type(self, entity_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all entities of a specific type.
        
        Args:
            entity_type: Type of entity to retrieve
            limit: Maximum number of entities to return
            
        Returns:
            List of entities of the specified type
        """
        try:
            query = {
                "size": limit,
                "query": {
                    "term": {"type": entity_type}
                },
                "sort": [{"local_name.keyword": {"order": "asc"}}]
            }
            
            response = self.es.search(index=self.index_name, body=query)
            
            results = []
            for hit in response['hits']['hits']:
                result = hit['_source'].copy()
                # Remove embedding for cleaner output
                if 'embedding' in result:
                    del result['embedding']
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting entities by type {entity_type}: {e}")
            return []
    
    def update_entity(self, uri: str, updates: Dict[str, Any]) -> bool:
        """
        Update an entity in the vector store.
        
        Args:
            uri: URI of the entity to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # If text_content is updated, regenerate embedding
            if 'text_content' in updates and updates['text_content']:
                embedding = self.embedding_model.embed_query(updates['text_content'])
                updates['embedding'] = embedding
            
            updates['updated_at'] = 'now'
            
            self.es.update(
                index=self.index_name,
                id=uri,
                body={"doc": updates}
            )
            
            logger.info(f"Updated entity: {uri}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating entity {uri}: {e}")
            return False
    
    def delete_entity(self, uri: str) -> bool:
        """
        Delete an entity from the vector store.
        
        Args:
            uri: URI of the entity to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.es.delete(index=self.index_name, id=uri)
            logger.info(f"Deleted entity: {uri}")
            return True
        except Exception as e:
            logger.error(f"Error deleting entity {uri}: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        try:
            # Get basic index stats
            stats = self.es.indices.stats(index=self.index_name)
            count_response = self.es.count(index=self.index_name)
            
            # Get entity type distribution
            type_agg_query = {
                "size": 0,
                "aggs": {
                    "entity_types": {
                        "terms": {
                            "field": "type",
                            "size": 20
                        }
                    }
                }
            }
            
            type_response = self.es.search(index=self.index_name, body=type_agg_query)
            type_distribution = {
                bucket['key']: bucket['doc_count'] 
                for bucket in type_response['aggregations']['entity_types']['buckets']
            }
            
            return {
                "total_entities": count_response['count'],
                "index_size_bytes": stats['indices'][self.index_name]['total']['store']['size_in_bytes'],
                "index_size_mb": round(stats['indices'][self.index_name]['total']['store']['size_in_bytes'] / (1024 * 1024), 2),
                "embedding_model": self.embedding_model_name,
                "embedding_dimensions": self.embedding_dimensions,
                "entity_type_distribution": type_distribution,
                "elasticsearch_hosts": self.hosts
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}
    
    def clear_index(self) -> bool:
        """
        Clear all documents from the index.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete all documents
            delete_query = {"query": {"match_all": {}}}
            response = self.es.delete_by_query(index=self.index_name, body=delete_query)
            
            deleted_count = response.get('deleted', 0)
            logger.info(f"Cleared {deleted_count} documents from index {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return False
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the Elasticsearch connection and index.
        
        Returns:
            Health status information
        """
        try:
            # Check cluster health
            cluster_health = self.es.cluster.health()
            
            # Check if index exists
            index_exists = self.es.indices.exists(index=self.index_name)
            
            # Get basic stats if index exists
            entity_count = 0
            if index_exists:
                count_response = self.es.count(index=self.index_name)
                entity_count = count_response['count']
            
            return {
                "elasticsearch_cluster_status": cluster_health['status'],
                "index_exists": index_exists,
                "entity_count": entity_count,
                "embedding_model": self.embedding_model_name,
                "embedding_dimensions": self.embedding_dimensions,
                "healthy": cluster_health['status'] in ['yellow', 'green'] and index_exists
            }
            
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }
