"""
Enhanced Elasticsearch Vector Store with direct Azure OpenAI embedding calls.
NO TIKTOKEN - Direct Azure OpenAI API calls only.
"""

import logging
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from elasticsearch import Elasticsearch

# Direct Azure OpenAI imports (no LangChain embedding wrapper)
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider, ClientSecretCredential
from app.utils.auth_helper import get_azure_token

# RDFLib imports for SPARQL endpoint connection
import rdflib
from rdflib import Graph
from rdflib.plugins.stores.sparqlstore import SPARQLStore
from rdflib.plugins.sparql import prepareQuery
from SPARQLWrapper import SPARQLWrapper, JSON, POST, GET
from SPARQLWrapper.SPARQLExceptions import SPARQLWrapperException

logger = logging.getLogger(__name__)

class DirectAzureEmbeddingClient:
    """
    Direct Azure OpenAI embedding client that bypasses LangChain and tiktoken.
    Calls Azure OpenAI API directly: https://{endpoint}/openai/deployments/{model}/embeddings
    """
    
    def __init__(self, 
                 azure_endpoint: str,
                 deployment_name: str,
                 api_version: str = "2023-05-15",
                 dimensions: int = 3072,
                 tenant_id: str = None,
                 client_id: str = None,
                 client_secret: str = None):
        """
        Initialize direct Azure OpenAI embedding client.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            deployment_name: Name of the embedding model deployment
            api_version: Azure OpenAI API version
            dimensions: Expected embedding dimensions
            tenant_id: Azure tenant ID for authentication
            client_id: Azure client ID for authentication
            client_secret: Azure client secret for authentication
        """
        self.azure_endpoint = azure_endpoint
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.dimensions = dimensions
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        
        # Ensure endpoint format
        if not self.azure_endpoint.startswith(('http://', 'https://')):
            self.azure_endpoint = f"https://{self.azure_endpoint}"
        if not self.azure_endpoint.endswith('/'):
            self.azure_endpoint = f"{self.azure_endpoint}/"
        
        # Create Azure OpenAI client
        self.client = self._create_azure_client()
        
        logger.info(f"Direct Azure embedding client initialized:")
        logger.info(f"  Endpoint: {self.azure_endpoint}")
        logger.info(f"  Deployment: {self.deployment_name}")
        logger.info(f"  API Version: {self.api_version}")
        logger.info(f"  Expected URL: {self.azure_endpoint}openai/deployments/{self.deployment_name}/embeddings?api-version={self.api_version}")
    
    def _create_azure_client(self) -> AzureOpenAI:
        """Create Azure OpenAI client with token authentication."""
        try:
            # Create token provider using service principal
            def token_provider():
                token = get_azure_token(
                    tenant_id=self.tenant_id,
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    scope="https://cognitiveservices.azure.com/.default"
                )
                if not token:
                    raise ValueError("Failed to obtain Azure AD token")
                return token
            
            # Test token
            test_token = token_provider()
            logger.info(f"✓ Azure AD token obtained: {test_token[:20]}...")
            
            # Create Azure OpenAI client
            client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                azure_ad_token_provider=token_provider,
                api_version=self.api_version
            )
            
            logger.info("✓ Direct Azure OpenAI client created successfully")
            return client
            
        except Exception as e:
            logger.error(f"Failed to create Azure OpenAI client: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single text query.
        Calls: https://{endpoint}/openai/deployments/{deployment}/embeddings
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return [0.0] * self.dimensions
            
            # Clean text
            cleaned_text = self._clean_text(text)
            
            logger.debug(f"Generating embedding for text: '{cleaned_text[:100]}...'")
            
            # Call Azure OpenAI embeddings API directly
            response = self.client.embeddings.create(
                model=self.deployment_name,  # This is the deployment name
                input=cleaned_text,
                dimensions=self.dimensions
            )
            
            # Extract embedding
            embedding = response.data[0].embedding
            
            # Validate dimensions
            if len(embedding) != self.dimensions:
                logger.error(f"Embedding dimension mismatch: expected {self.dimensions}, got {len(embedding)}")
                raise ValueError(f"Embedding dimension mismatch")
            
            logger.debug(f"✓ Embedding generated: {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            logger.error(f"Text length: {len(text)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding lists
        """
        try:
            if not texts:
                return []
            
            logger.info(f"Generating embeddings for {len(texts)} documents")
            
            # Clean texts
            cleaned_texts = [self._clean_text(text) for text in texts if text and text.strip()]
            
            if not cleaned_texts:
                logger.warning("No valid texts after cleaning")
                return []
            
            # Call Azure OpenAI embeddings API for batch
            response = self.client.embeddings.create(
                model=self.deployment_name,
                input=cleaned_texts,
                dimensions=self.dimensions
            )
            
            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            
            # Validate all dimensions
            for i, embedding in enumerate(embeddings):
                if len(embedding) != self.dimensions:
                    logger.error(f"Embedding {i} dimension mismatch: expected {self.dimensions}, got {len(embedding)}")
                    raise ValueError(f"Embedding dimension mismatch for item {i}")
            
            logger.info(f"✓ Generated {len(embeddings)} embeddings successfully")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            logger.error(f"Number of texts: {len(texts)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean text for embedding generation."""
        if not text:
            return ""
        
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Truncate if too long (Azure OpenAI has token limits)
        max_length = 8000  # Conservative limit
        if len(text) > max_length:
            text = text[:max_length-3] + "..."
            logger.debug(f"Truncated text to {max_length} characters")
        
        return text
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the embedding client connection."""
        try:
            test_text = "Test connection to Azure OpenAI embeddings"
            start_time = time.time()
            
            embedding = self.embed_query(test_text)
            
            end_time = time.time()
            duration = end_time - start_time
            
            return {
                "success": True,
                "endpoint_url": f"{self.azure_endpoint}openai/deployments/{self.deployment_name}/embeddings?api-version={self.api_version}",
                "deployment_name": self.deployment_name,
                "api_version": self.api_version,
                "test_text": test_text,
                "embedding_dimensions": len(embedding),
                "expected_dimensions": self.dimensions,
                "response_time_seconds": round(duration, 3),
                "sample_values": embedding[:5],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "endpoint_url": f"{self.azure_endpoint}openai/deployments/{self.deployment_name}/embeddings?api-version={self.api_version}",
                "deployment_name": self.deployment_name,
                "timestamp": datetime.now().isoformat()
            }


class EnhancedElasticsearchVectorStore:
    """
    Enhanced vector store using direct Azure OpenAI API calls and Elasticsearch 8.13.
    NO TIKTOKEN - Direct Azure OpenAI API integration only.
    """
    
    def __init__(self, 
                 hosts: List[str] = None,
                 index_name: str = "rdf_knowledge_graph",
                 embedding_model: str = "text-embedding-3-large",
                 embedding_dimensions: int = 3072,
                 sparql_endpoint_url: str = None,
                 sparql_username: str = None,
                 sparql_bearer_token: str = None):
        """
        Initialize Enhanced Elasticsearch Vector Store with direct Azure OpenAI integration.
        
        Args:
            hosts: Elasticsearch host addresses
            index_name: Name of the Elasticsearch index
            embedding_model: Azure OpenAI deployment name (text-embedding-3-large)
            embedding_dimensions: Dimension of the embedding vectors (3072)
            sparql_endpoint_url: Remote SPARQL endpoint URL
            sparql_username: Username for SPARQL endpoint authentication
            sparql_bearer_token: Bearer token for SPARQL endpoint authentication
        """
        # Parse and validate Elasticsearch hosts
        self.hosts = self._parse_elasticsearch_hosts(hosts)
        self.index_name = index_name
        self.embedding_model_name = embedding_model
        self.embedding_dimensions = embedding_dimensions
        
        # SPARQL endpoint configuration
        self.sparql_endpoint_url = sparql_endpoint_url or os.getenv("SPARQL_ENDPOINT_URL")
        self.sparql_username = sparql_username or os.getenv("SPARQL_USERNAME")
        self.sparql_bearer_token = sparql_bearer_token or os.getenv("SPARQL_BEARER_TOKEN")
        
        # SPARQL connections
        self.sparql_store = None
        self.sparql_graph = None
        self.sparql_wrapper = None
        
        # Initialize direct Azure OpenAI embedding client (NO LANGCHAIN, NO TIKTOKEN)
        self.embedding_client = self._setup_direct_embedding_client()
        
        # Initialize Elasticsearch client (optimized for 8.13)
        try:
            self.es = self._setup_elasticsearch_client()
            logger.info(f"Connected to Elasticsearch 8.13 at {self.hosts}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
        
        # Initialize SPARQL endpoint connection
        if self.sparql_endpoint_url:
            self._setup_sparql_connection()
        
        # Create index if it doesn't exist (optimized for ES 8.13)
        self._create_index()
        
        logger.info("✅ Enhanced Elasticsearch Vector Store initialized successfully!")
        logger.info(f"  🚫 NO tiktoken usage - Direct Azure OpenAI API only")
        logger.info(f"  📊 Embedding model: {self.embedding_model_name}")
        logger.info(f"  📏 Dimensions: {self.embedding_dimensions}")
    
    def _setup_direct_embedding_client(self) -> DirectAzureEmbeddingClient:
        """
        Set up direct Azure OpenAI embedding client (bypasses LangChain completely).
        Calls: https://{AZURE_ENDPOINT}/openai/deployments/text-embedding-3-large/embeddings
        """
        try:
            # Get Azure credentials from environment
            tenant_id = os.getenv("AZURE_TENANT_ID")
            client_id = os.getenv("AZURE_CLIENT_ID")
            client_secret = os.getenv("AZURE_CLIENT_SECRET")
            azure_endpoint = os.getenv("AZURE_ENDPOINT")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
            
            if not all([tenant_id, client_id, client_secret, azure_endpoint]):
                raise ValueError("Missing Azure credentials. Check AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, and AZURE_ENDPOINT")
            
            logger.info(f"🔧 Setting up direct Azure OpenAI embedding client:")
            logger.info(f"  📡 Azure endpoint: {azure_endpoint}")
            logger.info(f"  🚀 Model deployment: {self.embedding_model_name}")
            logger.info(f"  📅 API version: {api_version}")
            logger.info(f"  📏 Dimensions: {self.embedding_dimensions}")
            logger.info(f"  🎯 Target URL: {azure_endpoint}/openai/deployments/{self.embedding_model_name}/embeddings?api-version={api_version}")
            
            # Create direct embedding client
            embedding_client = DirectAzureEmbeddingClient(
                azure_endpoint=azure_endpoint,
                deployment_name=self.embedding_model_name,
                api_version=api_version,
                dimensions=self.embedding_dimensions,
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret
            )
            
            # Test the connection
            logger.info("🧪 Testing direct embedding client...")
            test_result = embedding_client.test_connection()
            
            if test_result["success"]:
                logger.info("✅ Direct Azure OpenAI embedding client test successful!")
                logger.info(f"  📡 Endpoint: {test_result['endpoint_url']}")
                logger.info(f"  ⏱️ Response time: {test_result['response_time_seconds']}s")
                logger.info(f"  📏 Dimensions: {test_result['embedding_dimensions']}")
                logger.info(f"  🔢 Sample values: {test_result['sample_values']}")
            else:
                logger.error(f"❌ Direct embedding client test failed: {test_result['error']}")
                raise ValueError(f"Embedding client test failed: {test_result['error']}")
            
            return embedding_client
            
        except Exception as e:
            logger.error(f"Failed to setup direct embedding client: {e}")
            logger.error(f"Make sure you have:")
            logger.error(f"  1. Correct Azure OpenAI deployment name: {self.embedding_model_name}")
            logger.error(f"  2. Valid Azure AD credentials")
            logger.error(f"  3. Deployment is active and accessible")
            raise
    
    def _setup_sparql_connection(self):
        """Set up SPARQL endpoint connection using rdflib with authentication."""
        try:
            logger.info(f"Setting up SPARQL connection to: {self.sparql_endpoint_url}")
            
            # Method 1: Using SPARQLWrapper for authenticated queries
            self.sparql_wrapper = SPARQLWrapper(self.sparql_endpoint_url)
            
            # Configure authentication
            if self.sparql_bearer_token:
                logger.info("Configuring SPARQL endpoint with Bearer token authentication")
                self.sparql_wrapper.addCustomHttpHeader("Authorization", f"Bearer {self.sparql_bearer_token}")
            
            if self.sparql_username:
                logger.info(f"Configuring SPARQL endpoint with username: {self.sparql_username}")
                # If you also have a password, you can set basic auth
                sparql_password = os.getenv("SPARQL_PASSWORD")
                if sparql_password:
                    self.sparql_wrapper.setCredentials(self.sparql_username, sparql_password)
            
            # Configure return format
            self.sparql_wrapper.setReturnFormat(JSON)
            self.sparql_wrapper.setMethod(POST)
            
            # Method 2: Using rdflib SPARQLStore (for graph operations)
            try:
                # Create custom headers for SPARQLStore
                custom_headers = {}
                if self.sparql_bearer_token:
                    custom_headers["Authorization"] = f"Bearer {self.sparql_bearer_token}"
                
                # Initialize SPARQLStore
                self.sparql_store = SPARQLStore(
                    query_endpoint=self.sparql_endpoint_url,
                    update_endpoint=self.sparql_endpoint_url.replace('/sparql', '/update') if '/sparql' in self.sparql_endpoint_url else None
                )
                
                # Add authentication headers if available
                if custom_headers:
                    # For rdflib 6.0+, we might need to configure headers differently
                    if hasattr(self.sparql_store, 'setCredentials'):
                        if self.sparql_username:
                            sparql_password = os.getenv("SPARQL_PASSWORD", "")
                            self.sparql_store.setCredentials(self.sparql_username, sparql_password)
                
                # Create graph with SPARQL store
                self.sparql_graph = Graph(store=self.sparql_store)
                
                logger.info("✓ SPARQL store configured successfully")
                
            except Exception as e:
                logger.warning(f"Could not set up SPARQLStore: {e}")
                self.sparql_store = None
                self.sparql_graph = None
            
            # Test the connection
            self._test_sparql_connection()
            
        except Exception as e:
            logger.error(f"Error setting up SPARQL connection: {e}")
            self.sparql_wrapper = None
            self.sparql_store = None
            self.sparql_graph = None
    
    def _test_sparql_connection(self):
        """Test SPARQL endpoint connectivity with authentication."""
        try:
            if self.sparql_wrapper:
                # Test with a simple query
                test_query = """
                SELECT ?s ?p ?o 
                WHERE { 
                    ?s ?p ?o 
                } 
                LIMIT 1
                """
                
                self.sparql_wrapper.setQuery(test_query)
                results = self.sparql_wrapper.query()
                
                if results:
                    logger.info("✓ SPARQL endpoint connection test successful")
                    return True
                else:
                    logger.warning("SPARQL endpoint returned empty results")
                    return False
                    
        except SPARQLWrapperException as e:
            logger.error(f"SPARQL endpoint test failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Error testing SPARQL endpoint: {e}")
            return False
        
        return False
    
    def execute_sparql_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SPARQL query on the remote endpoint."""
        if not self.sparql_wrapper:
            logger.error("SPARQL wrapper not configured")
            return []
        
        try:
            self.sparql_wrapper.setQuery(query)
            results = self.sparql_wrapper.query()
            
            if results.response.read():
                result_data = results.convert()
                
                # Convert to list of dictionaries
                processed_results = []
                if "results" in result_data and "bindings" in result_data["results"]:
                    for binding in result_data["results"]["bindings"]:
                        result_dict = {}
                        for var, value in binding.items():
                            result_dict[var] = value["value"]
                        processed_results.append(result_dict)
                
                logger.info(f"SPARQL query returned {len(processed_results)} results")
                return processed_results
            else:
                logger.warning("SPARQL query returned no results")
                return []
                
        except SPARQLWrapperException as e:
            logger.error(f"SPARQL query execution failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}")
            return []
    
    def _create_index(self):
        """Create the Elasticsearch index optimized for version 8.13."""
        try:
            if not self.es.indices.exists(index=self.index_name):
                # Elasticsearch 8.13 optimized configuration
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
                            
                            # Vector embedding - Elasticsearch 8.13 optimized (NO ef_search)
                            "embedding": {
                                "type": "dense_vector",
                                "dims": self.embedding_dimensions,
                                "index": True,
                                "similarity": "cosine",
                                "index_options": {
                                    "type": "hnsw",
                                    "m": 16,
                                    "ef_construction": 100
                                }
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
                            "refresh_interval": "1s",
                            "max_result_window": 10000,
                            "max_inner_result_window": 100,
                            "max_terms_count": 65536
                        },
                        "analysis": {
                            "analyzer": {
                                "standard_lowercase": {
                                    "type": "standard",
                                    "lowercase": True
                                }
                            }
                        }
                    }
                }
                
                self.es.indices.create(index=self.index_name, body=mapping)
                logger.info(f"Created Elasticsearch 8.13 index: {self.index_name}")
            else:
                logger.info(f"Elasticsearch index already exists: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error creating Elasticsearch index: {e}")
            raise
    
    def _parse_elasticsearch_hosts(self, hosts: Optional[List[str]]) -> List[str]:
        """Parse and validate Elasticsearch host URLs."""
        if not hosts:
            hosts_env = os.getenv("ELASTICSEARCH_HOSTS", "localhost:9200")
            hosts = [h.strip() for h in hosts_env.split(",") if h.strip()]
        
        formatted_hosts = []
        for host in hosts:
            host = host.strip()
            if host.startswith(('http://', 'https://')):
                formatted_hosts.append(host)
            else:
                if ':' in host:
                    formatted_hosts.append(f"http://{host}")
                else:
                    formatted_hosts.append(f"http://{host}:9200")
        
        logger.info(f"Elasticsearch hosts: {formatted_hosts}")
        return formatted_hosts
    
    def _setup_elasticsearch_client(self) -> Elasticsearch:
        """Set up Elasticsearch client optimized for 8.13 with SSL support."""
        try:
            logger.info(f"Setting up Elasticsearch client for hosts: {self.hosts}")
            
            # Base configuration
            es_config = {
                "hosts": self.hosts,
                "timeout": 30,
                "max_retries": 3,
                "retry_on_timeout": True,
                "request_timeout": 60,
                "retry_on_status": [429, 502, 503, 504]
            }
            
            # Authentication configuration
            username = os.getenv("ELASTICSEARCH_USERNAME")
            password = os.getenv("ELASTICSEARCH_PASSWORD")
            if username and password:
                es_config["basic_auth"] = (username, password)
                logger.info(f"Using basic authentication for Elasticsearch with username: {username}")
            else:
                logger.warning("No Elasticsearch credentials provided")
            
            # SSL/TLS Configuration
            ca_cert_path = os.getenv("ELASTICSEARCH_CA_CERT_PATH", "certs/ca.crt")
            use_ssl = os.getenv("ELASTICSEARCH_USE_SSL", "auto")  # auto-detect based on URL
            verify_certs = os.getenv("ELASTICSEARCH_VERIFY_CERTS", "true").lower() == "true"
            
            # Auto-detect SSL based on URL scheme or default behavior
            if use_ssl == "auto":
                # Check if any host uses https or if we're on default port 9200 with security
                if any(host.startswith("https://") for host in self.hosts):
                    use_ssl = "true"
                elif any(":9200" in host or host.endswith("9200") for host in self.hosts):
                    # Elasticsearch 8.x defaults to HTTPS on port 9200
                    use_ssl = "true"
                    # Update hosts to use HTTPS if they don't specify protocol
                    updated_hosts = []
                    for host in self.hosts:
                        if not host.startswith(('http://', 'https://')):
                            updated_hosts.append(f"https://{host}")
                        elif host.startswith('http://'):
                            updated_hosts.append(host.replace('http://', 'https://'))
                        else:
                            updated_hosts.append(host)
                    self.hosts = updated_hosts
                    es_config["hosts"] = self.hosts
                    logger.info(f"Auto-detected SSL, updated hosts to: {self.hosts}")
                else:
                    use_ssl = "false"
            
            # Configure SSL if enabled
            if use_ssl.lower() == "true":
                logger.info("Configuring SSL for Elasticsearch connection")
                
                # Check if CA certificate file exists
                if os.path.exists(ca_cert_path):
                    es_config["ca_certs"] = ca_cert_path
                    es_config["verify_certs"] = verify_certs
                    logger.info(f"Using CA certificate: {ca_cert_path}")
                    logger.info(f"Certificate verification: {'enabled' if verify_certs else 'disabled'}")
                else:
                    logger.warning(f"CA certificate not found at {ca_cert_path}")
                    
                    if verify_certs:
                        # Try common certificate paths
                        common_paths = [
                            "certs/ca.crt",
                            "certs/http_ca.crt", 
                            "config/certs/ca.crt",
                            "elasticsearch/config/certs/http_ca.crt",
                            "/usr/share/elasticsearch/config/certs/http_ca.crt"
                        ]
                        
                        ca_found = False
                        for path in common_paths:
                            if os.path.exists(path):
                                es_config["ca_certs"] = path
                                logger.info(f"Found CA certificate at: {path}")
                                ca_found = True
                                break
                        
                        if not ca_found:
                            logger.warning("No CA certificate found, disabling certificate verification")
                            es_config["verify_certs"] = False
                            es_config["ssl_show_warn"] = False
                    else:
                        es_config["verify_certs"] = False
                        es_config["ssl_show_warn"] = False
                        logger.warning("SSL certificate verification disabled")
            else:
                logger.info("SSL disabled for Elasticsearch connection")
            
            logger.info(f"Final Elasticsearch configuration:")
            config_for_log = es_config.copy()
            if "basic_auth" in config_for_log:
                config_for_log["basic_auth"] = (config_for_log["basic_auth"][0], "***")
            logger.info(f"  {config_for_log}")
            
            # Create Elasticsearch client
            es_client = Elasticsearch(**es_config)
            
            # Test connection with detailed error handling
            logger.info("Testing Elasticsearch connection...")
            try:
                if es_client.ping():
                    cluster_info = es_client.info()
                    logger.info(f"✓ Connected to Elasticsearch cluster: {cluster_info['cluster_name']} "
                               f"(version: {cluster_info['version']['number']})")
                    return es_client
                else:
                    raise ConnectionError("Elasticsearch ping returned False")
                    
            except Exception as ping_error:
                logger.error(f"Elasticsearch ping failed: {ping_error}")
                raise ConnectionError(f"All Elasticsearch connection attempts failed. Last error: {ping_error}")
            
        except Exception as e:
            logger.error(f"Failed to setup Elasticsearch client: {e}")
            raise
    
    def search_similar(self, 
                      query_text: str, 
                      top_k: int = 10,
                      entity_types: Optional[List[str]] = None,
                      min_score: float = 0.6,
                      include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Search for entities using Elasticsearch 8.13 KNN search with direct Azure OpenAI embeddings.
        """
        try:
            logger.info(f"Searching for entities similar to: '{query_text[:100]}...'")
            
            # Generate embedding using direct Azure OpenAI client (NO TIKTOKEN)
            logger.debug("Generating query embedding using direct Azure OpenAI client...")
            query_embedding = self.embedding_client.embed_query(query_text)
            
            logger.debug(f"Generated query embedding with {len(query_embedding)} dimensions")
            
            # Build KNN search query optimized for ES 8.13
            query = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": min(top_k * 2, 100),
                    "num_candidates": min(top_k * 5, 200),
                    "boost": 2.0
                },
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": ["text_content^2", "labels^1.5", "local_name^1.5", "comments"],
                                    "type": "best_fields",
                                    "boost": 1.0,
                                    "fuzziness": "AUTO"
                                }
                            }
                        ],
                        "minimum_should_match": 0
                    }
                },
                "size": top_k,
                "min_score": min_score,
                "_source": {
                    "excludes": ["embedding"] if not include_metadata else []
                }
            }
            
            # Add entity type filter if specified
            if entity_types:
                if "filter" not in query["query"]["bool"]:
                    query["query"]["bool"]["filter"] = []
                query["query"]["bool"]["filter"].append({
                    "terms": {"type": entity_types}
                })
                logger.debug(f"Added entity type filter: {entity_types}")
            
            # Execute search
            logger.debug("Executing KNN search in Elasticsearch...")
            response = self.es.search(index=self.index_name, body=query)
            
            results = []
            hits = response.get('hits', {}).get('hits', [])
            
            for hit in hits:
                score = hit['_score']
                result = hit['_source'].copy()
                result['similarity_score'] = score
                result['search_type'] = 'knn_hybrid'
                result['_id'] = hit['_id']
                
                # Remove embedding from result if not needed for metadata
                if not include_metadata and 'embedding' in result:
                    del result['embedding']
                
                results.append(result)
            
            logger.info(f"Found {len(results)} similar entities (scores: {[f'{r['similarity_score']:.3f}' for r in results[:3]]})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching for similar entities: {e}")
            logger.error(f"Query text: {query_text[:200]}...")
            return []
    
    def add_entities(self, entities: List[Dict[str, Any]], batch_size: int = 10) -> bool:
        """
        Add entities to the vector store with direct Azure OpenAI embeddings.
        NO TIKTOKEN - Direct Azure API calls only.
        
        Args:
            entities: List of entity dictionaries
            batch_size: Number of entities to process in each batch (smaller for large embeddings)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not entities:
                logger.warning("No entities provided for indexing")
                return True
            
            total_entities = len(entities)
            successful_indexes = 0
            
            logger.info(f"Processing {total_entities} entities with direct Azure OpenAI embedding calls")
            logger.info(f"Using batch size: {batch_size}")
            logger.info(f"🚫 NO tiktoken usage - Direct Azure OpenAI API only")
            
            # Process entities in smaller batches for large embeddings
            for i in range(0, total_entities, batch_size):
                batch = entities[i:i + batch_size]
                batch_num = i//batch_size + 1
                total_batches = (total_entities + batch_size - 1)//batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} entities)")
                
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
                    logger.warning(f"No valid texts in batch {batch_num}, skipping")
                    continue
                
                # Generate embeddings using direct Azure OpenAI client (NO TIKTOKEN)
                try:
                    logger.info(f"Generating embeddings for {len(texts)} texts using direct Azure OpenAI...")
                    
                    # Use direct embedding client
                    embeddings = self.embedding_client.embed_documents(texts)
                    
                    logger.info(f"✓ Generated {len(embeddings)} embeddings")
                    logger.info(f"  Embedding dimensions: {len(embeddings[0]) if embeddings else 'N/A'}")
                    
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch {batch_num}: {e}")
                    logger.error("Please check your Azure OpenAI configuration and model deployment")
                    continue
                
                # Prepare documents for Elasticsearch indexing
                actions = []
                for entity, embedding in zip(valid_entities, embeddings):
                    try:
                        # Clean and prepare the document
                        doc = self._prepare_document(entity, embedding)
                        
                        action = {
                            "_index": self.index_name,
                            "_id": entity['uri'],
                            "_source": doc
                        }
                        actions.append(action)
                    except Exception as e:
                        logger.error(f"Error preparing document for entity {entity.get('uri', 'unknown')}: {e}")
                        continue
                
                # Bulk index documents in Elasticsearch
                if actions:
                    try:
                        from elasticsearch.helpers import bulk
                        
                        logger.info(f"Indexing {len(actions)} documents in Elasticsearch...")
                        success_count, failed_items = bulk(
                            self.es, 
                            actions, 
                            chunk_size=batch_size,
                            request_timeout=120,
                            max_retries=3,
                            initial_backoff=2,
                            max_backoff=600
                        )
                        
                        successful_indexes += success_count
                        
                        if failed_items:
                            logger.warning(f"Failed to index {len(failed_items)} entities in batch {batch_num}")
                            for failed_item in failed_items[:3]:
                                logger.warning(f"  Failed item: {failed_item}")
                        
                        logger.info(f"✓ Batch {batch_num} completed: {success_count}/{len(actions)} documents indexed")
                        
                    except Exception as e:
                        logger.error(f"Error bulk indexing batch {batch_num}: {e}")
                        continue
                else:
                    logger.warning(f"No valid documents to index in batch {batch_num}")
            
            logger.info(f"✓ Embedding and indexing completed!")
            logger.info(f"  Total entities processed: {total_entities}")
            logger.info(f"  Successfully indexed: {successful_indexes}")
            logger.info(f"  Success rate: {(successful_indexes/total_entities)*100:.1f}%")
            logger.info(f"  🚫 NO tiktoken usage - Direct Azure OpenAI API calls only")
            
            return successful_indexes > 0
            
        except Exception as e:
            logger.error(f"Error adding entities to vector store: {e}")
            return False
    
    def _prepare_document(self, entity: Dict[str, Any], embedding: List[float]) -> Dict[str, Any]:
        """Prepare a document for Elasticsearch indexing."""
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
        
        optional_fields = [
            'superclasses', 'subclasses', 'related_properties', 
            'property_values', 'metadata', 'types', 'domains', 'ranges'
        ]
        
        for field in optional_fields:
            if field in entity:
                doc[field] = entity[field]
        
        return doc
    
    def test_embedding_endpoint(self) -> Dict[str, Any]:
        """Test the direct Azure OpenAI embedding endpoint."""
        try:
            logger.info("Testing direct Azure OpenAI embedding endpoint...")
            return self.embedding_client.test_connection()
        except Exception as e:
            logger.error(f"Embedding endpoint test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_entity(self, uri: str) -> Optional[Dict[str, Any]]:
        """Get a specific entity by URI."""
        try:
            response = self.es.get(index=self.index_name, id=uri)
            result = response['_source']
            
            if 'embedding' in result:
                del result['embedding']
            
            return result
        except Exception:
            logger.debug(f"Entity not found: {uri}")
            return None
    
    def get_entities_by_type(self, entity_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all entities of a specific type."""
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
                if 'embedding' in result:
                    del result['embedding']
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting entities by type {entity_type}: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the index."""
        try:
            stats = self.es.indices.stats(index=self.index_name)
            count_response = self.es.count(index=self.index_name)
            
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
                "elasticsearch_hosts": self.hosts,
                "elasticsearch_version": "8.13",
                "sparql_endpoint": self.get_sparql_endpoint_info(),
                "embedding_method": "Direct Azure OpenAI API (NO tiktoken)"
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}
    
    def get_sparql_endpoint_info(self) -> Dict[str, Any]:
        """Get information about the configured SPARQL endpoint."""
        if self.sparql_endpoint_url:
            return {
                "endpoint_url": self.sparql_endpoint_url,
                "status": "configured",
                "type": "remote",
                "authentication": {
                    "username": bool(self.sparql_username),
                    "bearer_token": bool(self.sparql_bearer_token)
                },
                "rdflib_store_available": self.sparql_store is not None,
                "sparql_wrapper_available": self.sparql_wrapper is not None
            }
        else:
            return {
                "endpoint_url": None,
                "status": "local_graph_only",
                "type": "local"
            }
    
    def test_sparql_endpoint(self) -> bool:
        """Test connectivity to the SPARQL endpoint."""
        if not self.sparql_endpoint_url:
            logger.info("No remote SPARQL endpoint configured")
            return False
        
        return self._test_sparql_connection()
    
    def check_health(self) -> Dict[str, Any]:
        """Check the health of the Elasticsearch connection and index."""
        try:
            cluster_health = self.es.cluster.health()
            index_exists = self.es.indices.exists(index=self.index_name)
            
            entity_count = 0
            if index_exists:
                count_response = self.es.count(index=self.index_name)
                entity_count = count_response['count']
            
            sparql_healthy = self.test_sparql_endpoint() if self.sparql_endpoint_url else True
            
            # Test embedding client
            embedding_test = self.test_embedding_endpoint()
            
            return {
                "elasticsearch_cluster_status": cluster_health['status'],
                "elasticsearch_version": "8.13",
                "index_exists": index_exists,
                "entity_count": entity_count,
                "embedding_model": self.embedding_model_name,
                "embedding_dimensions": self.embedding_dimensions,
                "embedding_client_healthy": embedding_test["success"],
                "embedding_method": "Direct Azure OpenAI API (NO tiktoken)",
                "sparql_endpoint_healthy": sparql_healthy,
                "sparql_endpoint_info": self.get_sparql_endpoint_info(),
                "healthy": (cluster_health['status'] in ['yellow', 'green'] and 
                           index_exists and embedding_test["success"])
            }
            
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }
