"""
Enhanced RDF Knowledge Graph Chatbot with GraphSparqlQAChain integration.
"""

import logging
import os
from typing import Dict, Any, Optional, List
from langchain_openai import AzureChatOpenAI
from app.core.rdf_manager import EnhancedRDFManager
from app.core.vector_store import EnhancedElasticsearchVectorStore
from app.core.query_processor import EnhancedQueryProcessor
from app.utils.auth_helper import get_azure_token

logger = logging.getLogger(__name__)

class EnhancedRDFChatbot:
    """
    Enhanced RDF chatbot that combines vector search with GraphSparqlQAChain.
    """
    
    def __init__(self, 
                 ontology_path: str = None,
                 elasticsearch_hosts: List[str] = None,
                 index_name: str = None,
                 embedding_model: str = None,
                 llm_model: str = None):
        """
        Initialize the Enhanced RDF chatbot.
        
        Args:
            ontology_path: Path to the ontology TTL file
            elasticsearch_hosts: Elasticsearch host addresses
            index_name: Elasticsearch index name
            embedding_model: Embedding model name
            llm_model: LLM model name
        """
        # Set defaults from environment or use provided values
        self.ontology_path = ontology_path or os.getenv("ONTOLOGY_PATH", "data/ontology.ttl")
        self.elasticsearch_hosts = elasticsearch_hosts or os.getenv("ELASTICSEARCH_HOSTS", "localhost:9200").split(",")
        self.index_name = index_name or os.getenv("ELASTICSEARCH_INDEX", "rdf_knowledge_graph")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        self.llm_model = llm_model or os.getenv("MODEL_NAME", "gpt-4o-mini")
        
        # Component instances
        self.rdf_manager = None
        self.vector_store = None
        self.query_processor = None
        self.llm = None
        
        # Initialization status
        self.initialization_status = {
            'rdf_manager': False,
            'vector_store': False,
            'llm': False,
            'sparql_chain': False,
            'query_processor': False
        }
        
        # Initialize components
        self._initialize_components()

    def _setup_llm(self) -> Optional[AzureChatOpenAI]:
    """Set up Azure OpenAI LLM with token authentication."""
    try:
        # Get Azure credentials
        tenant_id = os.getenv("AZURE_TENANT_ID")
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")
        azure_endpoint = os.getenv("AZURE_ENDPOINT")
        
        if not all([tenant_id, client_id, client_secret, azure_endpoint]):
            logger.error("Missing Azure OpenAI credentials")
            return None
        
        logger.info("Getting Azure token for LLM initialization...")
        
        # Get token using the enhanced auth helper
        token = get_azure_token(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            scope="https://cognitiveservices.azure.com/.default"
        )
        
        if not token:
            logger.error("Failed to obtain Azure token")
            return None
        
        logger.info(f"✓ Azure token obtained successfully: {token[:20]}...")
        
        # Create token provider function that refreshes tokens automatically
        def token_provider():
            fresh_token = get_azure_token(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
                scope="https://cognitiveservices.azure.com/.default"
            )
            if not fresh_token:
                logger.warning("Token provider returned None, using cached token")
                return token  # Fallback to initial token
            return fresh_token
        
        # Initialize LLM with proper token handling
        logger.info("Initializing AzureChatOpenAI...")
        
        try:
            # Try with azure_ad_token_provider first (newer LangChain versions)
            llm = AzureChatOpenAI(
                model=self.llm_model,
                azure_endpoint=azure_endpoint,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                azure_ad_token_provider=token_provider,
                temperature=float(os.getenv("TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("MAX_TOKENS", "4000")),
                streaming=False,  # Disable for better compatibility with SPARQL chain
                request_timeout=60.0,
                max_retries=3
            )
            logger.info("✓ LLM initialized with azure_ad_token_provider")
            
        except Exception as provider_error:
            logger.warning(f"azure_ad_token_provider failed: {provider_error}")
            logger.info("Trying with direct azure_ad_token...")
            
            # Fallback: Try with direct token (older LangChain versions or different parameter name)
            try:
                llm = AzureChatOpenAI(
                    model=self.llm_model,
                    azure_endpoint=azure_endpoint,
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                    azure_ad_token=token,  # Direct token instead of provider
                    temperature=float(os.getenv("TEMPERATURE", "0.1")),
                    max_tokens=int(os.getenv("MAX_TOKENS", "4000")),
                    streaming=False,
                    request_timeout=60.0,
                    max_retries=3
                )
                logger.info("✓ LLM initialized with direct azure_ad_token")
                
            except Exception as direct_error:
                logger.error(f"Direct azure_ad_token also failed: {direct_error}")
                
                # Final fallback: Try with API key approach if available
                api_key = os.getenv("AZURE_OPENAI_API_KEY")
                if api_key:
                    logger.info("Trying with Azure OpenAI API key...")
                    llm = AzureChatOpenAI(
                        model=self.llm_model,
                        azure_endpoint=azure_endpoint,
                        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                        api_key=api_key,
                        temperature=float(os.getenv("TEMPERATURE", "0.1")),
                        max_tokens=int(os.getenv("MAX_TOKENS", "4000")),
                        streaming=False,
                        request_timeout=60.0,
                        max_retries=3
                    )
                    logger.info("✓ LLM initialized with API key")
                else:
                    logger.error("No API key available as fallback")
                    raise direct_error
        
        # Test the LLM with a simple call
        logger.info("Testing LLM with a simple call...")
        try:
            test_response = llm.invoke("Hello, this is a test.")
            if test_response:
                logger.info("✓ LLM test successful")
                return llm
            else:
                logger.error("LLM test returned empty response")
                return None
        except Exception as test_error:
            logger.error(f"LLM test failed: {test_error}")
            return None
            
    except Exception as e:
        logger.error(f"Error setting up LLM: {e}")
        logger.error("Please check your Azure OpenAI configuration:")
        logger.error("1. Verify AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET")
        logger.error("2. Verify AZURE_ENDPOINT and model deployment")
        logger.error("3. Check if the service principal has proper permissions")
        return None
    
    def _initialize_components(self):
        """Initialize all chatbot components."""
        try:
            logger.info("=== Initializing Enhanced RDF Chatbot ===")
            
            # 1. Initialize RDF Manager
            logger.info("1. Initializing Enhanced RDF Manager...")
            self.rdf_manager = EnhancedRDFManager(self.ontology_path)
            self.initialization_status['rdf_manager'] = True
            logger.info("✓ RDF Manager initialized successfully")
            
            # 2. Initialize LLM
            logger.info("2. Initializing Azure OpenAI LLM...")
            self.llm = self._setup_llm()
            if self.llm:
                self.initialization_status['llm'] = True
                logger.info("✓ Azure OpenAI LLM initialized successfully")
            else:
                logger.warning("⚠ LLM initialization failed")
            
            # 3. Set up GraphSparqlQAChain
            if self.llm and self.rdf_manager:
                logger.info("3. Setting up GraphSparqlQAChain...")
                success = self.rdf_manager.setup_sparql_chain(self.llm)
                self.initialization_status['sparql_chain'] = success
                if success:
                    logger.info("✓ GraphSparqlQAChain setup successful")
                else:
                    logger.warning("⚠ GraphSparqlQAChain setup failed")
            
            # 4. Initialize Vector Store
            logger.info("4. Initializing Enhanced Vector Store...")
            try:
                self.vector_store = EnhancedElasticsearchVectorStore(
                    hosts=self.elasticsearch_hosts,
                    index_name=self.index_name,
                    embedding_model=self.embedding_model,
                    embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
                )
                self.initialization_status['vector_store'] = True
                logger.info("✓ Vector Store initialized successfully")
            except Exception as e:
                logger.error(f"⚠ Vector Store initialization failed: {e}")
                logger.error("Vector search will be disabled. Check Elasticsearch connection.")
                self.vector_store = None
                self.initialization_status['vector_store'] = False
            
            # 5. Initialize Query Processor
            logger.info("5. Initializing Enhanced Query Processor...")
            self.query_processor = EnhancedQueryProcessor(
                self.rdf_manager, 
                self.vector_store
            )
            self.initialization_status['query_processor'] = True
            logger.info("✓ Query Processor initialized successfully")
            
            logger.info("=== Enhanced RDF Chatbot Initialization Complete ===")
            self._log_initialization_summary()
            
        except Exception as e:
            logger.error(f"Error during chatbot initialization: {e}")
            raise
    
    def _setup_llm(self) -> Optional[AzureChatOpenAI]:
        """Set up Azure OpenAI LLM with token authentication."""
        try:
            # Get Azure credentials
            tenant_id = os.getenv("AZURE_TENANT_ID")
            client_id = os.getenv("AZURE_CLIENT_ID")
            client_secret = os.getenv("AZURE_CLIENT_SECRET")
            azure_endpoint = os.getenv("AZURE_ENDPOINT")
            
            if not all([tenant_id, client_id, client_secret, azure_endpoint]):
                logger.error("Missing Azure OpenAI credentials")
                return None
            
            # Get token using the enhanced auth helper
            token = get_azure_token(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
                scope="https://cognitiveservices.azure.com/.default"
            )
            
            if not token:
                logger.error("Failed to obtain Azure token")
                return None
            
            # Create token provider
            def token_provider():
                return get_azure_token(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret,
                    scope="https://cognitiveservices.azure.com/.default"
                )
            
            # Initialize LLM
            llm = AzureChatOpenAI(
                model=self.llm_model,
                azure_endpoint=azure_endpoint,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                azure_ad_token_provider=token_provider,
                temperature=float(os.getenv("TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("MAX_TOKENS", "4000")),
                streaming=False,  # Disable for better compatibility with SPARQL chain
                request_timeout=60.0,
                max_retries=3
            )
            
            # Test the LLM
            test_response = llm.invoke("Hello, this is a test.")
            logger.info("LLM test successful")
            
            return llm
            
        except Exception as e:
            logger.error(f"Error setting up LLM: {e}")
            return None
    
    def _log_initialization_summary(self):
        """Log a summary of the initialization status."""
        logger.info("=== Initialization Summary ===")
        for component, status in self.initialization_status.items():
            status_str = "✓ Ready" if status else "✗ Failed"
            logger.info(f"{component}: {status_str}")
        
        ready_components = sum(self.initialization_status.values())
        total_components = len(self.initialization_status)
        logger.info(f"Components ready: {ready_components}/{total_components}")
    
    def initialize_knowledge_base(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Initialize the knowledge base by indexing RDF entities.
        
        Args:
            force_rebuild: Whether to force rebuild the index
            
        Returns:
            Dictionary with initialization results
        """
        try:
            logger.info("=== Initializing Knowledge Base ===")
            
            # Check if index already has content and force_rebuild is False
            if not force_rebuild:
                stats = self.vector_store.get_index_stats()
                if stats.get('total_entities', 0) > 0:
                    logger.info(f"Index already contains {stats['total_entities']} entities")
                    return {
                        'success': True,
                        'message': f"Knowledge base already initialized with {stats['total_entities']} entities",
                        'entities_indexed': stats['total_entities'],
                        'force_rebuild': False
                    }
            
            # Clear index if force rebuild
            if force_rebuild:
                logger.info("Force rebuild requested - clearing existing index...")
                self.vector_store.clear_index()
            
            # Extract entities from RDF graph
            logger.info("Extracting entities from RDF graph...")
            entities = self.rdf_manager.get_all_entities()
            
            if not entities:
                logger.warning("No entities found in RDF graph")
                return {
                    'success': False,
                    'message': "No entities found in RDF graph",
                    'entities_indexed': 0
                }
            
            logger.info(f"Extracted {len(entities)} entities from RDF graph")
            
            # Index entities in vector store
            logger.info("Indexing entities in vector store...")
            success = self.vector_store.add_entities(entities, batch_size=25)
            
            if success:
                # Get final stats
                final_stats = self.vector_store.get_index_stats()
                entities_count = final_stats.get('total_entities', 0)
                
                logger.info(f"✓ Knowledge base initialized successfully with {entities_count} entities")
                
                return {
                    'success': True,
                    'message': f"Knowledge base initialized successfully",
                    'entities_indexed': entities_count,
                    'entity_types': final_stats.get('entity_type_distribution', {}),
                    'index_size_mb': final_stats.get('index_size_mb', 0),
                    'force_rebuild': force_rebuild
                }
            else:
                logger.error("Failed to index entities in vector store")
                return {
                    'success': False,
                    'message': "Failed to index entities in vector store",
                    'entities_indexed': 0
                }
                
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            return {
                'success': False,
                'message': f"Error initializing knowledge base: {e}",
                'entities_indexed': 0
            }
    
    def chat(self, 
             user_message: str, 
             include_context: bool = False,
             use_sparql_chain: bool = True,
             max_entities: int = 10) -> Dict[str, Any]:
        """
        Process a user message and generate a response.
        
        Args:
            user_message: The user's question or message
            include_context: Whether to include detailed context in response
            use_sparql_chain: Whether to use GraphSparqlQAChain
            max_entities: Maximum number of entities to retrieve
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            logger.info(f"Processing chat message: '{user_message}'")
            
            # Validate components
            if not self.initialization_status['query_processor']:
                return {
                    'user_message': user_message,
                    'response': "Sorry, the chatbot is not properly initialized. Please check the system status.",
                    'success': False,
                    'error': 'Query processor not initialized'
                }
            
            # Process the query using enhanced query processor
            query_result = self.query_processor.process_query(
                user_message,
                top_k=max_entities,
                use_sparql_chain=use_sparql_chain
            )
            
            if not query_result['success']:
                return {
                    'user_message': user_message,
                    'response': "I encountered an error processing your question. Please try rephrasing it.",
                    'success': False,
                    'error': query_result.get('error', 'Unknown error')
                }
            
            # Generate response using LLM
            if self.llm:
                response = self._generate_llm_response(
                    query_result['context'], 
                    user_message,
                    query_result
                )
            else:
                # Fallback response without LLM
                response = self._generate_fallback_response(query_result)
            
            # Prepare result
            result = {
                'user_message': user_message,
                'response': response,
                'query_classification': query_result['query_classification'],
                'key_concepts': query_result['key_concepts'],
                'processing_methods': query_result['processing_method'],
                'num_relevant_entities': len(query_result.get('vector_search_results', [])),
                'success': True
            }
            
            # Include additional context if requested
            if include_context:
                result.update({
                    'vector_search_results': query_result.get('vector_search_results', [])[:5],
                    'sparql_chain_result': query_result.get('sparql_chain_result'),
                    'direct_sparql_results': query_result.get('direct_sparql_results', []),
                    'context_used': query_result['context']
                })
            
            logger.info(f"Chat response generated successfully using methods: {query_result['processing_method']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return {
                'user_message': user_message,
                'response': "I'm sorry, I encountered an unexpected error. Please try again.",
                'success': False,
                'error': str(e)
            }
    
    def _generate_llm_response(self, 
                              context: str, 
                              user_message: str,
                              query_result: Dict[str, Any]) -> str:
        """Generate a response using the LLM with comprehensive context."""
        try:
            # Create an enhanced prompt
            intent = query_result['query_classification']['primary_intent']
            
            # Customize prompt based on intent
            intent_instructions = {
                'definition': "Provide a clear and comprehensive definition. Include all relevant details about the concept.",
                'relationship': "Explain the relationships and connections clearly. Show how different concepts are related.",
                'property': "List and explain the properties and attributes. Provide examples where possible.",
                'listing': "Provide a well-organized list. Group similar items together and explain their significance.",
                'comparison': "Compare and contrast the concepts clearly. Highlight similarities and differences.",
                'hierarchical': "Explain the hierarchical structure clearly. Show parent-child relationships.",
                'existence': "Confirm whether the concept exists and provide relevant details if it does.",
                'count': "Provide accurate counts and explain what the numbers represent."
            }
            
            instruction = intent_instructions.get(intent, "Provide a comprehensive and helpful answer.")
            
            prompt = f"""You are a knowledgeable assistant that answers questions about RDF knowledge graphs and ontologies.
            
Query Intent: {intent}
Key Concepts: {', '.join(query_result['key_concepts'][:3])}

Context from Knowledge Graph:
{context}

User Question: {user_message}

Instructions:
- {instruction}
- Base your answer on the provided knowledge graph context
- Be accurate and cite specific information from the context when possible
- If multiple processing methods were used, synthesize the information effectively
- If the GraphSparqlQAChain provided an answer, prioritize and build upon it
- Be conversational but informative
- If information is insufficient, acknowledge this and suggest related topics

Answer:"""
            
            # Generate response using the LLM
            response = self.llm.invoke(prompt)
            
            # Extract the content from the response
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return self._generate_fallback_response(query_result)
    
    def _generate_fallback_response(self, query_result: Dict[str, Any]) -> str:
        """Generate a fallback response when LLM is not available."""
        try:
            # Use GraphSparqlQAChain result if available
            if query_result.get('sparql_chain_result') and not query_result['sparql_chain_result'].get('error'):
                return query_result['sparql_chain_result']['answer']
            
            # Use vector search results
            if query_result.get('vector_search_results'):
                entities = query_result['vector_search_results'][:3]
                response_parts = [
                    f"I found {len(entities)} relevant entities in the knowledge graph:"
                ]
                
                for i, entity in enumerate(entities, 1):
                    response_parts.append(f"{i}. {entity['local_name']} ({entity['type']})")
                    if entity.get('comments'):
                        response_parts.append(f"   {entity['comments'][0]}")
                
                return "\n".join(response_parts)
            
            # Use direct SPARQL results
            if query_result.get('direct_sparql_results'):
                return f"I found information using SPARQL queries. Please check the detailed results."
            
            return "I couldn't find specific information about your query in the knowledge graph."
            
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return "I encountered an error processing your query."
    
    def get_entity_details(self, entity_uri: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific entity.
        
        Args:
            entity_uri: URI of the entity
            
        Returns:
            Entity details or None if not found
        """
        try:
            # Get entity from vector store
            entity = self.vector_store.get_entity(entity_uri)
            if entity:
                # Enhance with related entities from RDF graph
                related = self.rdf_manager.find_related_entities(entity_uri, max_depth=1)
                entity['related_entities'] = related[:10]
                return entity
            return None
        except Exception as e:
            logger.error(f"Error getting entity details: {e}")
            return None
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the knowledge base.
        
        Returns:
            Dictionary with knowledge base statistics
        """
        try:
            # Vector store statistics
            vector_stats = self.vector_store.get_index_stats()
            
            # RDF graph statistics
            graph_stats = {
                'total_triples': len(self.rdf_manager.graph),
                'namespaces': list(self.rdf_manager.namespaces.keys()),
                'classes': len(self.rdf_manager.schema_info.get('classes', [])),
                'properties': len(self.rdf_manager.schema_info.get('properties', [])),
                'individuals': len(self.rdf_manager.schema_info.get('individuals', []))
            }
            
            # System status
            system_status = {
                'initialization_status': self.initialization_status,
                'components_ready': sum(self.initialization_status.values()),
                'total_components': len(self.initialization_status),
                'ontology_path': self.ontology_path,
                'models': {
                    'llm_model': self.llm_model,
                    'embedding_model': self.embedding_model
                }
            }
            
            return {
                'vector_store': vector_stats,
                'rdf_graph': graph_stats,
                'system_status': system_status
            }
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {'error': str(e)}
    
    def refresh_knowledge_base(self) -> Dict[str, Any]:
        """
        Refresh the knowledge base by reloading and reindexing.
        
        Returns:
            Dictionary with refresh results
        """
        try:
            logger.info("Refreshing knowledge base...")
            
            # Reload the RDF graph
            logger.info("Reloading RDF graph...")
            self.rdf_manager.load_ontology()
            
            # Reinitialize the vector store
            logger.info("Rebuilding vector store...")
            result = self.initialize_knowledge_base(force_rebuild=True)
            
            if result['success']:
                logger.info("Knowledge base refresh completed successfully")
                return {
                    'success': True,
                    'message': 'Knowledge base refreshed successfully',
                    'entities_reindexed': result['entities_indexed']
                }
            else:
                logger.error("Knowledge base refresh failed")
                return {
                    'success': False,
                    'message': 'Knowledge base refresh failed',
                    'error': result.get('message', 'Unknown error')
                }
            
        except Exception as e:
            logger.error(f"Error refreshing knowledge base: {e}")
            return {
                'success': False,
                'message': f'Error refreshing knowledge base: {e}'
            }
    
    def get_query_suggestions(self, partial_query: str = "") -> List[str]:
        """
        Get query suggestions for the user.
        
        Args:
            partial_query: Partial query input
            
        Returns:
            List of suggested queries
        """
        try:
            if self.query_processor:
                return self.query_processor.get_query_suggestions(partial_query)
            return []
        except Exception as e:
            logger.error(f"Error getting query suggestions: {e}")
            return []
    
    def check_health(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of all components.
        
        Returns:
            Health status information
        """
        try:
            health_status = {
                'overall_healthy': True,
                'components': {},
                'timestamp': 'now'
            }
            
            # Check RDF Manager
            try:
                if self.rdf_manager and len(self.rdf_manager.graph) > 0:
                    health_status['components']['rdf_manager'] = {
                        'healthy': True,
                        'triples_count': len(self.rdf_manager.graph)
                    }
                else:
                    health_status['components']['rdf_manager'] = {
                        'healthy': False,
                        'error': 'No triples loaded'
                    }
                    health_status['overall_healthy'] = False
            except Exception as e:
                health_status['components']['rdf_manager'] = {
                    'healthy': False,
                    'error': str(e)
                }
                health_status['overall_healthy'] = False
            
            # Check Vector Store
            try:
                vector_health = self.vector_store.check_health()
                health_status['components']['vector_store'] = vector_health
                if not vector_health.get('healthy', False):
                    health_status['overall_healthy'] = False
            except Exception as e:
                health_status['components']['vector_store'] = {
                    'healthy': False,
                    'error': str(e)
                }
                health_status['overall_healthy'] = False
            
            # Check LLM
            try:
                if self.llm:
                    test_response = self.llm.invoke("Test")
                    health_status['components']['llm'] = {
                        'healthy': True,
                        'model': self.llm_model
                    }
                else:
                    health_status['components']['llm'] = {
                        'healthy': False,
                        'error': 'LLM not initialized'
                    }
                    health_status['overall_healthy'] = False
            except Exception as e:
                health_status['components']['llm'] = {
                    'healthy': False,
                    'error': str(e)
                }
                health_status['overall_healthy'] = False
            
            # Check SPARQL Chain
            health_status['components']['sparql_chain'] = {
                'healthy': self.initialization_status.get('sparql_chain', False),
                'available': self.rdf_manager.sparql_chain is not None
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            return {
                'overall_healthy': False,
                'error': str(e)
            }
