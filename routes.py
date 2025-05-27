"""
FastAPI routes for the Enhanced RDF Knowledge Graph Chatbot.
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from app.core.chatbot import EnhancedRDFChatbot

logger = logging.getLogger(__name__)

# Pydantic models for API requests and responses
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's question or message")
    include_context: bool = Field(False, description="Include detailed context in response")
    use_sparql_chain: bool = Field(True, description="Use GraphSparqlQAChain for processing")
    max_entities: int = Field(10, ge=1, le=50, description="Maximum number of entities to retrieve")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated response")
    user_message: str = Field(..., description="Original user message")
    query_classification: Optional[Dict[str, Any]] = Field(None, description="Query classification details")
    key_concepts: Optional[List[str]] = Field(None, description="Extracted key concepts")
    processing_methods: Optional[List[str]] = Field(None, description="Methods used for processing")
    num_relevant_entities: Optional[int] = Field(None, description="Number of relevant entities found")
    success: bool = Field(..., description="Whether the request was successful")
    error: Optional[str] = Field(None, description="Error message if any")

class EntityResponse(BaseModel):
    uri: str
    type: str
    local_name: str
    namespace: Optional[str] = None
    labels: Optional[List[str]] = None
    comments: Optional[List[str]] = None
    text_content: Optional[str] = None
    related_entities: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

class StatsResponse(BaseModel):
    vector_store: Dict[str, Any]
    rdf_graph: Dict[str, Any]
    system_status: Dict[str, Any]

class HealthResponse(BaseModel):
    overall_healthy: bool
    components: Dict[str, Any]
    timestamp: str

class InitializeRequest(BaseModel):
    force_rebuild: bool = Field(False, description="Force rebuild of the knowledge base")

class InitializeResponse(BaseModel):
    success: bool
    message: str
    entities_indexed: int
    entity_types: Optional[Dict[str, int]] = None
    index_size_mb: Optional[float] = None
    force_rebuild: bool

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    entity_types: Optional[List[str]] = Field(None, description="Filter by entity types")
    min_score: float = Field(0.6, ge=0.0, le=1.0, description="Minimum similarity score")

# Global chatbot instance
chatbot_instance: Optional[EnhancedRDFChatbot] = None

def get_chatbot() -> EnhancedRDFChatbot:
    """Dependency to get the chatbot instance."""
    global chatbot_instance
    if chatbot_instance is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    return chatbot_instance

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Enhanced RDF Knowledge Graph Chatbot",
        description="An advanced chatbot that answers questions based on RDF knowledge graphs using Azure OpenAI and GraphSparqlQAChain",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize the chatbot on startup."""
        global chatbot_instance
        try:
            logger.info("Starting Enhanced RDF Chatbot API...")
            chatbot_instance = EnhancedRDFChatbot()
            
            # Check if knowledge base needs initialization
            stats = chatbot_instance.get_knowledge_base_stats()
            entity_count = stats.get('vector_store', {}).get('total_entities', 0)
            
            if entity_count == 0:
                logger.info("Knowledge base is empty, initializing...")
                init_result = chatbot_instance.initialize_knowledge_base()
                if init_result['success']:
                    logger.info(f"Knowledge base initialized with {init_result['entities_indexed']} entities")
                else:
                    logger.warning(f"Knowledge base initialization failed: {init_result['message']}")
            else:
                logger.info(f"Knowledge base already contains {entity_count} entities")
                
            logger.info("Enhanced RDF Chatbot API started successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {e}")
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logger.info("Shutting down Enhanced RDF Chatbot API...")
    
    # Main chat endpoint
    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest, chatbot: EnhancedRDFChatbot = Depends(get_chatbot)):
        """
        Chat with the RDF knowledge graph chatbot.
        
        Process a natural language question and get an AI-generated response
        based on the RDF knowledge graph using multiple processing methods.
        """
        try:
            result = chatbot.chat(
                user_message=request.message,
                include_context=request.include_context,
                use_sparql_chain=request.use_sparql_chain,
                max_entities=request.max_entities
            )
            return ChatResponse(**result)
        except Exception as e:
            logger.error(f"Error in chat endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Entity details endpoint
    @app.get("/entity/{entity_uri:path}", response_model=EntityResponse)
    async def get_entity(entity_uri: str, chatbot: EnhancedRDFChatbot = Depends(get_chatbot)):
        """
        Get detailed information about a specific entity by URI.
        
        Returns comprehensive information about an entity including
        its properties, relationships, and related entities.
        """
        try:
            entity = chatbot.get_entity_details(entity_uri)
            if entity is None:
                raise HTTPException(status_code=404, detail="Entity not found")
            return EntityResponse(**entity)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting entity: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Search endpoint
    @app.post("/search")
    async def search_entities(request: SearchRequest, chatbot: EnhancedRDFChatbot = Depends(get_chatbot)):
        """
        Search for entities using vector similarity and text search.
        
        Performs hybrid search combining vector embeddings and traditional
        text search to find the most relevant entities.
        """
        try:
            results = chatbot.vector_store.search_similar(
                query_text=request.query,
                top_k=request.top_k,
                entity_types=request.entity_types,
                min_score=request.min_score
            )
            return {"results": results, "total": len(results)}
        except Exception as e:
            logger.error(f"Error in search endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Get entities by type
    @app.get("/entities/type/{entity_type}")
    async def get_entities_by_type(
        entity_type: str, 
        limit: int = Query(100, ge=1, le=1000),
        chatbot: EnhancedRDFChatbot = Depends(get_chatbot)
    ):
        """
        Get all entities of a specific type.
        
        Returns a list of entities filtered by their type (e.g., 'Class', 'ObjectProperty').
        """
        try:
            entities = chatbot.vector_store.get_entities_by_type(entity_type, limit)
            return {"entities": entities, "total": len(entities), "type": entity_type}
        except Exception as e:
            logger.error(f"Error getting entities by type: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Knowledge base statistics
    @app.get("/stats", response_model=StatsResponse)
    async def get_stats(chatbot: EnhancedRDFChatbot = Depends(get_chatbot)):
        """
        Get comprehensive statistics about the knowledge base.
        
        Returns information about the vector store, RDF graph,
        and system status.
        """
        try:
            stats = chatbot.get_knowledge_base_stats()
            return StatsResponse(**stats)
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Initialize knowledge base
    @app.post("/initialize", response_model=InitializeResponse)
    async def initialize_knowledge_base(
        request: InitializeRequest,
        background_tasks: BackgroundTasks,
        chatbot: EnhancedRDFChatbot = Depends(get_chatbot)
    ):
        """
        Initialize or rebuild the knowledge base.
        
        Extracts entities from the RDF graph and creates vector embeddings.
        Can be run in the background for large ontologies.
        """
        try:
            if request.force_rebuild:
                # Run rebuild in background for large datasets
                background_tasks.add_task(
                    chatbot.initialize_knowledge_base, 
                    force_rebuild=True
                )
                return InitializeResponse(
                    success=True,
                    message="Knowledge base rebuild started in background",
                    entities_indexed=0,
                    force_rebuild=True
                )
            else:
                # Run immediately for regular initialization
                result = chatbot.initialize_knowledge_base(force_rebuild=False)
                return InitializeResponse(**result)
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Refresh knowledge base
    @app.post("/refresh")
    async def refresh_knowledge_base(
        background_tasks: BackgroundTasks,
        chatbot: EnhancedRDFChatbot = Depends(get_chatbot)
    ):
        """
        Refresh the knowledge base by reloading the ontology and rebuilding the index.
        
        This operation runs in the background to avoid blocking the API.
        """
        try:
            background_tasks.add_task(chatbot.refresh_knowledge_base)
            return {"message": "Knowledge base refresh started in background"}
        except Exception as e:
            logger.error(f"Error starting refresh: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Query suggestions
    @app.get("/suggestions")
    async def get_query_suggestions(
        partial_query: str = Query("", description="Partial query for suggestions"),
        chatbot: EnhancedRDFChatbot = Depends(get_chatbot)
    ):
        """
        Get query suggestions based on partial input.
        
        Helps users formulate better questions by suggesting
        common query patterns and available concepts.
        """
        try:
            suggestions = chatbot.get_query_suggestions(partial_query)
            return {"suggestions": suggestions}
        except Exception as e:
            logger.error(f"Error getting suggestions: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Health check
    @app.get("/health", response_model=HealthResponse)
    async def health_check(chatbot: EnhancedRDFChatbot = Depends(get_chatbot)):
        """
        Comprehensive health check of all system components.
        
        Checks the status of RDF manager, vector store, LLM,
        and GraphSparqlQAChain components.
        """
        try:
            health_status = chatbot.check_health()
            return HealthResponse(**health_status)
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            # Return partial health info even if chatbot check fails
            return HealthResponse(
                overall_healthy=False,
                components={"error": str(e)},
                timestamp="now"
            )
    
    # Simple health endpoint for load balancers
    @app.get("/ping")
    async def ping():
        """Simple ping endpoint for load balancer health checks."""
        return {"status": "ok", "service": "rdf-chatbot-api"}
    
    # Schema information
    @app.get("/schema")
    async def get_schema_info(chatbot: EnhancedRDFChatbot = Depends(get_chatbot)):
        """
        Get information about the RDF schema/ontology.
        
        Returns classes, properties, and individuals defined in the ontology.
        """
        try:
            if chatbot.rdf_manager:
                schema_summary = chatbot.rdf_manager.get_schema_summary()
                schema_info = chatbot.rdf_manager.schema_info
                
                return {
                    "summary": schema_summary,
                    "classes": schema_info.get('classes', []),
                    "properties": schema_info.get('properties', []),
                    "individuals": schema_info.get('individuals', []),
                    "namespaces": list(chatbot.rdf_manager.namespaces.keys())
                }
            else:
                raise HTTPException(status_code=500, detail="RDF manager not available")
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # SPARQL query endpoint
    @app.post("/sparql")
    async def execute_sparql(
        query: str = Query(..., description="SPARQL query to execute"),
        chatbot: EnhancedRDFChatbot = Depends(get_chatbot)
    ):
        """
        Execute a SPARQL query directly on the RDF graph.
        
        Advanced users can run custom SPARQL queries against the knowledge graph.
        Use with caution as this provides direct access to the underlying data.
        """
        try:
            if not chatbot.rdf_manager:
                raise HTTPException(status_code=500, detail="RDF manager not available")
            
            # Basic query validation
            if not query.strip():
                raise HTTPException(status_code=400, detail="Empty query provided")
            
            # Execute the query
            results = chatbot.rdf_manager.query_sparql(query)
            
            return {
                "query": query,
                "results": results,
                "result_count": len(results)
            }
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}")
            raise HTTPException(status_code=400, detail=f"SPARQL query error: {e}")
    
    # Natural language to SPARQL endpoint
    @app.post("/nl2sparql")
    async def natural_language_to_sparql(
        question: str = Query(..., description="Natural language question"),
        chatbot: EnhancedRDFChatbot = Depends(get_chatbot)
    ):
        """
        Convert natural language question to SPARQL using GraphSparqlQAChain.
        
        Uses the LLM to generate SPARQL queries from natural language questions.
        Returns both the generated query and its results.
        """
        try:
            if not chatbot.rdf_manager or not chatbot.rdf_manager.sparql_chain:
                raise HTTPException(status_code=500, detail="SPARQL chain not available")
            
            result = chatbot.rdf_manager.query_with_langchain(question)
            
            if result.get('error'):
                raise HTTPException(status_code=400, detail=result['error'])
            
            return {
                "question": question,
                "sparql_query": result.get('sparql_query', ''),
                "answer": result.get('answer', ''),
                "success": True
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in NL2SPARQL: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Ontology file upload endpoint (if needed for dynamic updates)
    @app.post("/upload-ontology")
    async def upload_ontology():
        """
        Upload a new ontology file (placeholder for future implementation).
        
        This endpoint would allow dynamic updating of the knowledge base
        with new ontology files.
        """
        raise HTTPException(
            status_code=501, 
            detail="Ontology upload not implemented. Please update the file directly and use /refresh."
        )
    
    # Exception handlers
    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        return JSONResponse(
            status_code=404,
            content={"detail": "Endpoint not found", "path": str(request.url.path)}
        )
    
    @app.exception_handler(500)
    async def internal_error_handler(request, exc):
        logger.error(f"Internal server error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__}
        )
    
    return app

# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
