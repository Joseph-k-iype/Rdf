"""
Enhanced RDF Knowledge Graph Chatbot - Main Entry Point

This application provides a sophisticated chatbot that can answer questions
about RDF knowledge graphs using:
- Vector embeddings with text-embedding-3-large
- GraphSparqlQAChain for natural language to SPARQL conversion
- Azure OpenAI GPT-4o-mini for response generation
- Elasticsearch for vector storage and search

Usage:
    python main.py                    # Interactive CLI mode
    python main.py api               # Start FastAPI server
    python main.py api --host 0.0.0.0 --port 8080  # Custom host/port
    python main.py init              # Initialize knowledge base only
    python main.py health            # Check system health
"""

import os
import sys
import asyncio
import logging
import argparse
from typing import Optional
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('chatbot.log') if os.getenv("LOG_TO_FILE", "false").lower() == "true" else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup and validate environment variables."""
    required_vars = [
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID", 
        "AZURE_CLIENT_SECRET",
        "AZURE_ENDPOINT"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please check your .env file or environment settings")
        return False
    
    # Validate paths
    ontology_path = os.getenv("ONTOLOGY_PATH", "data/ontology.ttl")
    if not os.path.exists(ontology_path):
        logger.error(f"Ontology file not found: {ontology_path}")
        logger.error("Please ensure your ontology.ttl file exists in the data/ directory")
        return False
    
    return True

def run_interactive_mode():
    """Run the chatbot in interactive command-line mode."""
    try:
        print("=" * 60)
        print("🤖 Enhanced RDF Knowledge Graph Chatbot")
        print("=" * 60)
        print("Initializing chatbot components...")
        
        from app.core.chatbot import EnhancedRDFChatbot
        
        chatbot = EnhancedRDFChatbot()
        
        # Check knowledge base status
        print("\n📊 Checking knowledge base status...")
        stats = chatbot.get_knowledge_base_stats()
        entity_count = stats.get('vector_store', {}).get('total_entities', 0)
        
        if entity_count == 0:
            print("📚 Knowledge base is empty. Initializing...")
            init_result = chatbot.initialize_knowledge_base()
            if init_result['success']:
                print(f"✅ Knowledge base initialized with {init_result['entities_indexed']} entities")
            else:
                print(f"❌ Knowledge base initialization failed: {init_result['message']}")
                print("You can still use the chatbot, but responses may be limited.")
        else:
            print(f"✅ Knowledge base loaded with {entity_count} entities")
        
        # Show system status
        print("\n🔧 System Status:")
        for component, status in chatbot.initialization_status.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {component}")
        
        print(f"\n💬 Chatbot ready! Available processing methods:")
        if chatbot.initialization_status.get('sparql_chain'):
            print("  🔗 GraphSparqlQAChain (Natural language to SPARQL)")
        if chatbot.initialization_status.get('vector_store'):
            print("  🔍 Vector similarity search")
        print("  📝 Direct SPARQL query generation")
        
        print(f"\n🎯 Query suggestions:")
        suggestions = chatbot.get_query_suggestions()
        for i, suggestion in enumerate(suggestions[:5], 1):
            print(f"  {i}. {suggestion}")
        
        print(f"\n{'='*60}")
        print("Type your questions below. Commands:")
        print("  'help' - Show help")
        print("  'suggestions' - Get query suggestions")
        print("  'stats' - Show knowledge base statistics") 
        print("  'health' - Check system health")
        print("  'quit' or 'exit' - Exit the chatbot")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n🤔 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == 'help':
                    print("\n📋 Available Commands:")
                    print("  help - Show this help message")
                    print("  suggestions - Get query suggestions")
                    print("  stats - Show knowledge base statistics")
                    print("  health - Check system health")
                    print("  quit/exit - Exit the chatbot")
                    print("\n💡 Example Questions:")
                    print("  • What is a Person?")
                    print("  • List all classes in the ontology")
                    print("  • How are different entities related?")
                    print("  • What properties does X have?")
                    continue
                
                elif user_input.lower() == 'suggestions':
                    suggestions = chatbot.get_query_suggestions()
                    print("\n💡 Query Suggestions:")
                    for i, suggestion in enumerate(suggestions[:8], 1):
                        print(f"  {i}. {suggestion}")
                    continue
                
                elif user_input.lower() == 'stats':
                    stats = chatbot.get_knowledge_base_stats()
                    print("\n📊 Knowledge Base Statistics:")
                    print(f"  Total entities: {stats.get('vector_store', {}).get('total_entities', 0)}")
                    print(f"  RDF triples: {stats.get('rdf_graph', {}).get('total_triples', 0)}")
                    print(f"  Classes: {stats.get('rdf_graph', {}).get('classes', 0)}")
                    print(f"  Properties: {stats.get('rdf_graph', {}).get('properties', 0)}")
                    print(f"  Index size: {stats.get('vector_store', {}).get('index_size_mb', 0):.2f} MB")
                    continue
                
                elif user_input.lower() == 'health':
                    health = chatbot.check_health()
                    print(f"\n🏥 System Health: {'✅ Healthy' if health['overall_healthy'] else '❌ Issues detected'}")
                    for component, status in health['components'].items():
                        health_icon = "✅" if status.get('healthy', False) else "❌"
                        print(f"  {health_icon} {component}")
                        if not status.get('healthy', False) and 'error' in status:
                            print(f"      Error: {status['error']}")
                    continue
                
                # Process regular questions
                print("🤖 Bot: Processing your question...")
                
                result = chatbot.chat(
                    user_message=user_input, 
                    include_context=False,
                    use_sparql_chain=True,
                    max_entities=8
                )
                
                if result['success']:
                    print(f"🤖 Bot: {result['response']}")
                    
                    # Show additional info if available
                    if result.get('processing_methods'):
                        methods = ", ".join(result['processing_methods'])
                        print(f"📋 (Processing: {methods})")
                    
                    if result.get('key_concepts'):
                        concepts = ", ".join(result['key_concepts'][:3])
                        print(f"🔑 (Key concepts: {concepts})")
                    
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"❌ Error: {e}")
                print("Please try rephrasing your question.")
                
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        print(f"❌ Failed to initialize chatbot: {e}")
        print("Please check your configuration and try again.")

def run_api_mode(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the chatbot as a FastAPI web service."""
    try:
        print("🚀 Starting Enhanced RDF Chatbot API...")
        print(f"🌐 API will be available at: http://{host}:{port}")
        print(f"📚 Documentation: http://{host}:{port}/docs")
        print(f"📖 ReDoc: http://{host}:{port}/redoc")
        
        # Import here to avoid issues if dependencies are missing
        from app.api.routes import app
        
        uvicorn.run(
            "app.api.routes:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level.lower(),
            access_log=True
        )
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        print(f"❌ Failed to start API server: {e}")

def run_initialization_only():
    """Initialize the knowledge base and exit."""
    try:
        print("📚 Initializing knowledge base...")
        
        from app.core.chatbot import EnhancedRDFChatbot
        
        chatbot = EnhancedRDFChatbot()
        
        result = chatbot.initialize_knowledge_base(force_rebuild=True)
        
        if result['success']:
            print(f"✅ Knowledge base initialized successfully!")
            print(f"📊 Entities indexed: {result['entities_indexed']}")
            if result.get('entity_types'):
                print("📋 Entity type distribution:")
                for entity_type, count in result['entity_types'].items():
                    print(f"  • {entity_type}: {count}")
            print(f"💾 Index size: {result.get('index_size_mb', 0):.2f} MB")
        else:
            print(f"❌ Knowledge base initialization failed: {result['message']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to initialize knowledge base: {e}")
        print(f"❌ Failed to initialize knowledge base: {e}")
        sys.exit(1)

def run_health_check():
    """Run a health check and exit."""
    try:
        print("🏥 Running system health check...")
        
        from app.core.chatbot import EnhancedRDFChatbot
        
        chatbot = EnhancedRDFChatbot()
        health = chatbot.check_health()
        
        print(f"\n🏥 Overall Health: {'✅ Healthy' if health['overall_healthy'] else '❌ Issues detected'}")
        print("\n📋 Component Status:")
        
        for component, status in health['components'].items():
            health_icon = "✅" if status.get('healthy', False) else "❌"
            print(f"  {health_icon} {component}")
            
            if status.get('healthy', False):
                # Show additional info for healthy components
                if component == 'vector_store' and 'entity_count' in status:
                    print(f"      Entities: {status['entity_count']}")
                elif component == 'rdf_manager' and 'triples_count' in status:
                    print(f"      Triples: {status['triples_count']}")
                elif component == 'llm' and 'model' in status:
                    print(f"      Model: {status['model']}")
            else:
                # Show error details for unhealthy components
                if 'error' in status:
                    print(f"      Error: {status['error']}")
        
        # Exit with appropriate code
        sys.exit(0 if health['overall_healthy'] else 1)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        print(f"❌ Health check failed: {e}")
        sys.exit(1)

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced RDF Knowledge Graph Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Interactive mode
  python main.py api                       # Start API server
  python main.py api --host 0.0.0.0 --port 8080  # Custom host/port
  python main.py init                      # Initialize knowledge base
  python main.py health                    # Health check
        """
    )
    
    parser.add_argument(
        'mode',
        nargs='?',
        choices=['interactive', 'api', 'init', 'health'],
        default='interactive',
        help='Operation mode (default: interactive)'
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='API server host (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='API server port (default: 8000)'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Route to appropriate mode
    try:
        if args.mode == 'interactive':
            run_interactive_mode()
        elif args.mode == 'api':
            run_api_mode(args.host, args.port, args.reload)
        elif args.mode == 'init':
            run_initialization_only()
        elif args.mode == 'health':
            run_health_check()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
