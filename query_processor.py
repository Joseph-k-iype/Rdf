"""
Enhanced Query Processor with GraphSparqlQAChain integration.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from app.core.rdf_manager import EnhancedRDFManager
from app.core.vector_store import EnhancedElasticsearchVectorStore

logger = logging.getLogger(__name__)

class EnhancedQueryProcessor:
    """
    Enhanced query processor that combines vector search with GraphSparqlQAChain.
    """
    
    def __init__(self, 
                 rdf_manager: EnhancedRDFManager, 
                 vector_store: EnhancedElasticsearchVectorStore):
        """
        Initialize enhanced query processor.
        
        Args:
            rdf_manager: Enhanced RDF graph manager instance
            vector_store: Enhanced vector store for entity embeddings
        """
        self.rdf_manager = rdf_manager
        self.vector_store = vector_store
        
        # Enhanced query classification patterns
        self.query_patterns = {
            'definition': [
                r'what is (a |an )?(.+)',
                r'define (.+)',
                r'explain (.+)',
                r'describe (.+)',
                r'tell me about (.+)',
                r'give me information about (.+)',
                r'(.+) definition'
            ],
            'relationship': [
                r'how (is |are )?(.+) related to (.+)',
                r'what (.+) connect(.+) to (.+)',
                r'relationship between (.+) and (.+)',
                r'(.+) related to (.+)',
                r'how do (.+) and (.+) interact',
                r'connection between (.+) and (.+)'
            ],
            'property': [
                r'what (.+) properties (.+) (have|has)',
                r'what attributes (.+) (have|has)',
                r'properties of (.+)',
                r'attributes of (.+)',
                r'characteristics of (.+)',
                r'what can (.+) do',
                r'capabilities of (.+)'
            ],
            'listing': [
                r'list (.+)',
                r'show me (.+)',
                r'find all (.+)',
                r'what are (.+)',
                r'give me all (.+)',
                r'enumerate (.+)',
                r'all (.+) in (.+)'
            ],
            'comparison': [
                r'difference between (.+) and (.+)',
                r'compare (.+) (and|with) (.+)',
                r'(.+) vs (.+)',
                r'(.+) versus (.+)',
                r'how (.+) different from (.+)',
                r'similarities between (.+) and (.+)'
            ],
            'hierarchical': [
                r'what are the subclasses of (.+)',
                r'what are the superclasses of (.+)',
                r'(.+) hierarchy',
                r'children of (.+)',
                r'parents of (.+)',
                r'what inherits from (.+)',
                r'what does (.+) inherit from'
            ],
            'existence': [
                r'does (.+) exist',
                r'is there (.+)',
                r'are there any (.+)',
                r'do we have (.+)',
                r'can you find (.+)'
            ],
            'count': [
                r'how many (.+)',
                r'count (.+)',
                r'number of (.+)',
                r'total (.+)'
            ]
        }
        
        # Intent keywords for better classification
        self.intent_keywords = {
            'definition': ['what', 'define', 'explain', 'describe', 'meaning', 'definition'],
            'relationship': ['related', 'relationship', 'connection', 'interact', 'connect'],
            'property': ['property', 'properties', 'attribute', 'attributes', 'characteristic'],
            'listing': ['list', 'show', 'all', 'find', 'enumerate', 'give me'],
            'comparison': ['difference', 'compare', 'vs', 'versus', 'different', 'similar'],
            'hierarchical': ['subclass', 'superclass', 'hierarchy', 'inherit', 'parent', 'child'],
            'existence': ['exist', 'there', 'any', 'have', 'find'],
            'count': ['many', 'count', 'number', 'total', 'how much']
        }

    
    def _generate_targeted_sparql_queries(self, 
                                     query: str, 
                                     classification: Dict[str, Any], 
                                     concepts: List[str]) -> List[Dict[str, str]]:
    """Generate targeted SPARQL queries based on query classification with improved syntax."""
    queries = []
    primary_intent = classification['primary_intent']
    
    try:
        if primary_intent == 'listing':
            # Generate queries to list entities
            for concept in concepts[:3]:  # Limit to first 3 concepts
                # List classes containing the concept
                queries.append({
                    'type': 'list_classes',
                    'query': self._build_list_classes_query(concept),
                    'description': f'Classes related to "{concept}"'
                })
                
                # List properties containing the concept
                queries.append({
                    'type': 'list_properties',
                    'query': self._build_list_properties_query(concept),
                    'description': f'Properties related to "{concept}"'
                })
        
        elif primary_intent == 'hierarchical':
            for concept in concepts[:2]:
                # Find subclasses
                queries.append({
                    'type': 'subclasses',
                    'query': self._build_subclasses_query(concept),
                    'description': f'Subclasses of classes related to "{concept}"'
                })
                
                # Find superclasses
                queries.append({
                    'type': 'superclasses',
                    'query': self._build_superclasses_query(concept),
                    'description': f'Superclasses of classes related to "{concept}"'
                })
        
        elif primary_intent == 'relationship':
            if len(concepts) >= 2:
                concept1, concept2 = concepts[0], concepts[1]
                # Find relationships between concepts
                queries.append({
                    'type': 'relationships',
                    'query': self._build_relationships_query(concept1, concept2),
                    'description': f'Relationships between "{concept1}" and "{concept2}"'
                })
        
        elif primary_intent == 'count':
            for concept in concepts[:2]:
                # Count entities related to concept
                queries.append({
                    'type': 'count_classes',
                    'query': self._build_count_classes_query(concept),
                    'description': f'Count of classes related to "{concept}"'
                })
        
        return queries
        
    except Exception as e:
        logger.error(f"Error generating SPARQL queries: {e}")
        return []

def _build_list_classes_query(self, concept: str) -> str:
    """Build a well-formed SPARQL query to list classes."""
    return f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT ?class ?label WHERE {{
    ?class rdf:type owl:Class .
    OPTIONAL {{ ?class rdfs:label ?label }}
    FILTER(
        CONTAINS(LCASE(STR(?class)), LCASE("{self._escape_sparql_string(concept)}")) || 
        CONTAINS(LCASE(STR(?label)), LCASE("{self._escape_sparql_string(concept)}"))
    )
}} 
ORDER BY ?class
LIMIT 20
"""

def _build_list_properties_query(self, concept: str) -> str:
    """Build a well-formed SPARQL query to list properties."""
    return f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT ?property ?label ?type WHERE {{
    {{
        ?property rdf:type owl:ObjectProperty .
        BIND("ObjectProperty" AS ?type)
    }} UNION {{
        ?property rdf:type owl:DatatypeProperty .
        BIND("DatatypeProperty" AS ?type)
    }}
    OPTIONAL {{ ?property rdfs:label ?label }}
    FILTER(
        CONTAINS(LCASE(STR(?property)), LCASE("{self._escape_sparql_string(concept)}")) || 
        CONTAINS(LCASE(STR(?label)), LCASE("{self._escape_sparql_string(concept)}"))
    )
}} 
ORDER BY ?property
LIMIT 15
"""

def _build_subclasses_query(self, concept: str) -> str:
    """Build a well-formed SPARQL query to find subclasses."""
    return f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT ?subclass ?label WHERE {{
    ?class rdf:type owl:Class .
    ?subclass rdfs:subClassOf ?class .
    ?subclass rdf:type owl:Class .
    OPTIONAL {{ ?subclass rdfs:label ?label }}
    OPTIONAL {{ ?class rdfs:label ?classLabel }}
    FILTER(
        CONTAINS(LCASE(STR(?class)), LCASE("{self._escape_sparql_string(concept)}")) || 
        CONTAINS(LCASE(STR(?classLabel)), LCASE("{self._escape_sparql_string(concept)}"))
    )
}} 
ORDER BY ?subclass
LIMIT 15
"""

def _build_superclasses_query(self, concept: str) -> str:
    """Build a well-formed SPARQL query to find superclasses."""
    return f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT ?superclass ?label WHERE {{
    ?class rdf:type owl:Class .
    ?class rdfs:subClassOf ?superclass .
    ?superclass rdf:type owl:Class .
    OPTIONAL {{ ?superclass rdfs:label ?label }}
    OPTIONAL {{ ?class rdfs:label ?classLabel }}
    FILTER(
        CONTAINS(LCASE(STR(?class)), LCASE("{self._escape_sparql_string(concept)}")) || 
        CONTAINS(LCASE(STR(?classLabel)), LCASE("{self._escape_sparql_string(concept)}"))
    )
}} 
ORDER BY ?superclass
LIMIT 15
"""

def _build_relationships_query(self, concept1: str, concept2: str) -> str:
    """Build a well-formed SPARQL query to find relationships between concepts."""
    return f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT ?subject ?predicate ?object ?subjectLabel ?objectLabel WHERE {{
    ?subject ?predicate ?object .
    OPTIONAL {{ ?subject rdfs:label ?subjectLabel }}
    OPTIONAL {{ ?object rdfs:label ?objectLabel }}
    FILTER(
        (
            CONTAINS(LCASE(STR(?subject)), LCASE("{self._escape_sparql_string(concept1)}")) || 
            CONTAINS(LCASE(STR(?subjectLabel)), LCASE("{self._escape_sparql_string(concept1)}"))
        ) && (
            CONTAINS(LCASE(STR(?object)), LCASE("{self._escape_sparql_string(concept2)}")) || 
            CONTAINS(LCASE(STR(?objectLabel)), LCASE("{self._escape_sparql_string(concept2)}"))
        )
    ) || (
        (
            CONTAINS(LCASE(STR(?subject)), LCASE("{self._escape_sparql_string(concept2)}")) || 
            CONTAINS(LCASE(STR(?subjectLabel)), LCASE("{self._escape_sparql_string(concept2)}"))
        ) && (
            CONTAINS(LCASE(STR(?object)), LCASE("{self._escape_sparql_string(concept1)}")) || 
            CONTAINS(LCASE(STR(?objectLabel)), LCASE("{self._escape_sparql_string(concept1)}"))
        )
    )
    FILTER(?predicate != rdf:type)
}} 
ORDER BY ?subject ?predicate
LIMIT 20
"""

def _build_count_classes_query(self, concept: str) -> str:
    """Build a well-formed SPARQL query to count classes."""
    return f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT (COUNT(DISTINCT ?class) AS ?count) WHERE {{
    ?class rdf:type owl:Class .
    OPTIONAL {{ ?class rdfs:label ?label }}
    FILTER(
        CONTAINS(LCASE(STR(?class)), LCASE("{self._escape_sparql_string(concept)}")) || 
        CONTAINS(LCASE(STR(?label)), LCASE("{self._escape_sparql_string(concept)}"))
    )
}}
"""

def _escape_sparql_string(self, text: str) -> str:
    """Escape special characters in SPARQL string literals."""
    if not text:
        return ""
    
    # Replace quotes and backslashes that could break SPARQL
    text = text.replace('\\', '\\\\')  # Escape backslashes first
    text = text.replace('"', '\\"')    # Escape double quotes
    text = text.replace("'", "\\'")    # Escape single quotes
    text = text.replace('\n', '\\n')   # Escape newlines
    text = text.replace('\r', '\\r')   # Escape carriage returns
    text = text.replace('\t', '\\t')   # Escape tabs
    
    # Remove any other potentially problematic characters
    import re
    text = re.sub(r'[^\w\s\-_.]', '', text)
    
    return text

def test_sparql_query_generation(self) -> Dict[str, Any]:
    """Test SPARQL query generation with various inputs."""
    try:
        test_results = {}
        
        # Test basic query generation
        test_cases = [
            ("listing", ["Person", "Class"]),
            ("hierarchical", ["Animal", "Vehicle"]),
            ("relationship", ["Person", "hasName"]),
            ("count", ["Property"])
        ]
        
        for intent, concepts in test_cases:
            try:
                classification = {'primary_intent': intent}
                queries = self._generate_targeted_sparql_queries("test query", classification, concepts)
                
                test_results[intent] = {
                    'success': True,
                    'query_count': len(queries),
                    'queries': [q['query'] for q in queries]
                }
                
                # Test syntax validation for each query
                for query_info in queries:
                    validation = self.rdf_manager.validate_sparql_query_syntax(query_info['query'])
                    if not validation.get('valid', False):
                        test_results[intent]['syntax_errors'] = test_results[intent].get('syntax_errors', [])
                        test_results[intent]['syntax_errors'].append(validation.get('error', 'Unknown error'))
                        
            except Exception as e:
                test_results[intent] = {
                    'success': False,
                    'error': str(e)
                }
        
        return {
            'overall_success': all(result.get('success', False) for result in test_results.values()),
            'test_results': test_results
        }
        
    except Exception as e:
        return {
            'overall_success': False,
            'error': str(e)
        }
    def process_query(self, 
                     user_query: str, 
                     top_k: int = 15,
                     use_sparql_chain: bool = True,
                     include_vector_search: bool = True) -> Dict[str, Any]:
        """
        Process a user query using hybrid approach.
        
        Args:
            user_query: The user's question
            top_k: Number of top relevant entities to retrieve
            use_sparql_chain: Whether to use GraphSparqlQAChain
            include_vector_search: Whether to include vector similarity search
            
        Returns:
            Dictionary with comprehensive query results
        """
        try:
            logger.info(f"Processing query: {user_query}")
            
            # Clean and preprocess query
            cleaned_query = self._preprocess_query(user_query)
            
            # Classify the query with confidence
            query_classification = self._classify_query_enhanced(cleaned_query)
            
            # Extract key entities/concepts from the query
            key_concepts = self._extract_key_concepts_enhanced(cleaned_query)
            
            # Initialize result structure
            result = {
                'original_query': user_query,
                'cleaned_query': cleaned_query,
                'query_classification': query_classification,
                'key_concepts': key_concepts,
                'vector_search_results': [],
                'sparql_chain_result': None,
                'direct_sparql_results': [],
                'context': '',
                'success': True,
                'processing_method': []
            }
            
            # 1. Vector similarity search (if enabled)
            if include_vector_search and self.vector_store:
                logger.info("Performing vector similarity search...")
                try:
                    vector_results = self.vector_store.search_similar(
                        query_text=cleaned_query,
                        top_k=top_k,
                        min_score=0.4
                    )
                    result['vector_search_results'] = vector_results
                    result['processing_method'].append('vector_search')
                    logger.info(f"Vector search found {len(vector_results)} relevant entities")
                except Exception as e:
                    logger.warning(f"Vector search failed: {e}")
                    result['vector_search_results'] = []
            elif include_vector_search and not self.vector_store:
                logger.warning("Vector search requested but vector store not available")
                result['vector_search_results'] = []
            
            # 2. GraphSparqlQAChain query (if enabled and available)
            if use_sparql_chain and self.rdf_manager.sparql_chain:
                logger.info("Using GraphSparqlQAChain for natural language query...")
                try:
                    sparql_result = self.rdf_manager.query_with_langchain(cleaned_query)
                    result['sparql_chain_result'] = sparql_result
                    result['processing_method'].append('sparql_chain')
                    
                    if sparql_result and not sparql_result.get('error'):
                        logger.info("GraphSparqlQAChain generated successful response")
                    else:
                        logger.warning(f"GraphSparqlQAChain error: {sparql_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.warning(f"GraphSparqlQAChain failed: {e}")
                    result['sparql_chain_result'] = {'error': str(e)}
            
            # 3. Direct SPARQL queries based on query type
            if query_classification['primary_intent'] in ['listing', 'hierarchical', 'count', 'relationship']:
                logger.info("Generating direct SPARQL queries...")
                sparql_queries = self._generate_targeted_sparql_queries(
                    cleaned_query, 
                    query_classification, 
                    key_concepts
                )
                
                direct_sparql_results = []
                for query_info in sparql_queries:
                    try:
                        sparql_results = self.rdf_manager.query_sparql(query_info['query'])
                        if sparql_results:
                            direct_sparql_results.append({
                                'query_type': query_info['type'],
                                'sparql_query': query_info['query'],
                                'results': sparql_results,
                                'description': query_info['description']
                            })
                    except Exception as e:
                        logger.warning(f"Direct SPARQL query failed: {e}")
                
                result['direct_sparql_results'] = direct_sparql_results
                result['processing_method'].append('direct_sparql')
                logger.info(f"Direct SPARQL generated {len(direct_sparql_results)} result sets")
            
            # 4. Enhance vector results with related entities
            if result['vector_search_results']:
                logger.info("Enhancing top vector results with related entities...")
                enhanced_results = []
                for i, entity in enumerate(result['vector_search_results'][:5]):  # Enhance top 5
                    enhanced_entity = self._enhance_entity_with_relations(entity)
                    enhanced_results.append(enhanced_entity)
                result['enhanced_entities'] = enhanced_results
            
            # 5. Generate comprehensive context
            result['context'] = self._generate_comprehensive_context(result)
            
            logger.info(f"Query processing completed using methods: {result['processing_method']}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'original_query': user_query,
                'error': str(e),
                'success': False
            }
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess and clean the user query."""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Handle common abbreviations
        abbreviations = {
            r'\bwhats\b': 'what is',
            r'\bwhos\b': 'who is',
            r'\bwheres\b': 'where is',
            r'\bhows\b': 'how is',
            r'\bwhens\b': 'when is'
        }
        
        for abbr, expansion in abbreviations.items():
            query = re.sub(abbr, expansion, query, flags=re.IGNORECASE)
        
        return query
    
    def _classify_query_enhanced(self, query: str) -> Dict[str, Any]:
        """Enhanced query classification with confidence scoring."""
        query_lower = query.lower().strip()
        
        # Score each intent based on patterns and keywords
        intent_scores = {}
        
        # Pattern matching
        for intent, patterns in self.query_patterns.items():
            pattern_score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    pattern_score += 1
            intent_scores[intent] = pattern_score
        
        # Keyword matching
        for intent, keywords in self.intent_keywords.items():
            keyword_score = sum(1 for keyword in keywords if keyword in query_lower)
            intent_scores[intent] = intent_scores.get(intent, 0) + keyword_score * 0.5
        
        # Determine primary and secondary intents
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_intent = sorted_intents[0][0] if sorted_intents[0][1] > 0 else 'general'
        secondary_intent = sorted_intents[1][0] if len(sorted_intents) > 1 and sorted_intents[1][1] > 0 else None
        
        return {
            'primary_intent': primary_intent,
            'secondary_intent': secondary_intent,
            'confidence': sorted_intents[0][1] if sorted_intents else 0,
            'all_scores': dict(sorted_intents)
        }
    
    def _extract_key_concepts_enhanced(self, query: str) -> List[str]:
        """Enhanced key concept extraction."""
        # Remove stop words and extract meaningful terms
        stop_words = {
            'what', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'how', 'why', 'when', 'where', 'who', 'which', 'that', 'this',
            'tell', 'me', 'about', 'show', 'find', 'list', 'describe', 'explain', 'define',
            'do', 'does', 'did', 'can', 'could', 'would', 'should', 'will', 'have', 'has', 'had'
        }
        
        # Extract words and phrases
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        concepts = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Look for multi-word concepts (bigrams and trigrams)
        multi_words = []
        for i in range(len(concepts) - 1):
            if len(concepts[i]) > 3 and len(concepts[i + 1]) > 3:
                bigram = f"{concepts[i]} {concepts[i + 1]}"
                multi_words.append(bigram)
                
                # Trigrams
                if i < len(concepts) - 2 and len(concepts[i + 2]) > 3:
                    trigram = f"{concepts[i]} {concepts[i + 1]} {concepts[i + 2]}"
                    multi_words.append(trigram)
        
        # Look for quoted phrases
        quoted_phrases = re.findall(r'\"([^\"]+)\"', query)
        quoted_phrases.extend(re.findall(r"'([^']+)'", query))
        
        all_concepts = list(set(concepts + multi_words + quoted_phrases))
        
        # Score concepts based on position and frequency
        scored_concepts = []
        for concept in all_concepts:
            score = query.lower().count(concept.lower())
            # Boost score if concept appears early in query
            if query.lower().find(concept.lower()) < len(query) * 0.3:
                score += 1
            scored_concepts.append((concept, score))
        
        # Return concepts sorted by score
        sorted_concepts = sorted(scored_concepts, key=lambda x: x[1], reverse=True)
        return [concept[0] for concept in sorted_concepts]
    
    def _enhance_entity_with_relations(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance entity data with additional relationship information."""
        enhanced = entity.copy()
        
        try:
            # Find related entities
            related_entities = self.rdf_manager.find_related_entities(
                entity['uri'], 
                max_depth=1
            )
            enhanced['related_entities'] = related_entities[:8]  # Limit to top 8
            
            # Add specific enhancements based on entity type
            if entity['type'] == 'Class':
                # Find instances of this class
                instances = self._find_class_instances(entity['uri'])
                enhanced['instances'] = instances[:5]  # Limit to 5 examples
                
                # Find sibling classes (classes with same superclass)
                siblings = self._find_sibling_classes(entity['uri'])
                enhanced['sibling_classes'] = siblings[:5]
                
            elif entity['type'] in ['ObjectProperty', 'DatatypeProperty']:
                # Find example usage of this property
                examples = self._find_property_usage_examples(entity['uri'])
                enhanced['usage_examples'] = examples[:3]
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing entity {entity['uri']}: {e}")
            return entity
    
    def _find_class_instances(self, class_uri: str) -> List[Dict[str, str]]:
        """Find instances of a given class."""
        try:
            query = f"""
            SELECT ?instance ?label WHERE {{
                ?instance rdf:type <{class_uri}> .
                OPTIONAL {{ ?instance rdfs:label ?label }}
            }} LIMIT 10
            """
            results = self.rdf_manager.query_sparql(query)
            return [
                {
                    'uri': result['instance'],
                    'label': result.get('label', self.rdf_manager._get_local_name(result['instance']))
                }
                for result in results
            ]
        except Exception:
            return []
    
    def _find_sibling_classes(self, class_uri: str) -> List[Dict[str, str]]:
        """Find sibling classes (classes with same superclass)."""
        try:
            query = f"""
            SELECT ?sibling ?label WHERE {{
                <{class_uri}> rdfs:subClassOf ?parent .
                ?sibling rdfs:subClassOf ?parent .
                FILTER(?sibling != <{class_uri}>)
                OPTIONAL {{ ?sibling rdfs:label ?label }}
            }} LIMIT 10
            """
            results = self.rdf_manager.query_sparql(query)
            return [
                {
                    'uri': result['sibling'],
                    'label': result.get('label', self.rdf_manager._get_local_name(result['sibling']))
                }
                for result in results
            ]
        except Exception:
            return []
    
    def _find_property_usage_examples(self, property_uri: str) -> List[Dict[str, str]]:
        """Find example usage of a property."""
        try:
            query = f"""
            SELECT ?subject ?object WHERE {{
                ?subject <{property_uri}> ?object .
            }} LIMIT 5
            """
            results = self.rdf_manager.query_sparql(query)
            return [
                {
                    'subject': self.rdf_manager._get_local_name(result['subject']),
                    'object': str(result['object'])
                }
                for result in results
            ]
        except Exception:
            return []
    
    def _generate_targeted_sparql_queries(self, 
                                         query: str, 
                                         classification: Dict[str, Any], 
                                         concepts: List[str]) -> List[Dict[str, str]]:
        """Generate targeted SPARQL queries based on query classification."""
        queries = []
        primary_intent = classification['primary_intent']
        
        try:
            if primary_intent == 'listing':
                # Generate queries to list entities
                for concept in concepts[:3]:  # Limit to first 3 concepts
                    # List classes containing the concept
                    queries.append({
                        'type': 'list_classes',
                        'query': f"""
                        SELECT ?class ?label WHERE {{
                            ?class rdf:type owl:Class .
                            OPTIONAL {{ ?class rdfs:label ?label }}
                            FILTER(CONTAINS(LCASE(STR(?class)), LCASE("{concept}")) || 
                                   CONTAINS(LCASE(STR(?label)), LCASE("{concept}")))
                        }} LIMIT 20
                        """,
                        'description': f'Classes related to "{concept}"'
                    })
                    
                    # List properties containing the concept
                    queries.append({
                        'type': 'list_properties',
                        'query': f"""
                        SELECT ?property ?label ?type WHERE {{
                            {{
                                ?property rdf:type owl:ObjectProperty .
                                BIND("ObjectProperty" AS ?type)
                            }} UNION {{
                                ?property rdf:type owl:DatatypeProperty .
                                BIND("DatatypeProperty" AS ?type)
                            }}
                            OPTIONAL {{ ?property rdfs:label ?label }}
                            FILTER(CONTAINS(LCASE(STR(?property)), LCASE("{concept}")) || 
                                   CONTAINS(LCASE(STR(?label)), LCASE("{concept}")))
                        }} LIMIT 15
                        """,
                        'description': f'Properties related to "{concept}"'
                    })
            
            elif primary_intent == 'hierarchical':
                for concept in concepts[:2]:
                    # Find subclasses
                    queries.append({
                        'type': 'subclasses',
                        'query': f"""
                        SELECT ?subclass ?label WHERE {{
                            ?class rdf:type owl:Class .
                            ?subclass rdfs:subClassOf ?class .
                            OPTIONAL {{ ?subclass rdfs:label ?label }}
                            FILTER(CONTAINS(LCASE(STR(?class)), LCASE("{concept}")) || 
                                   CONTAINS(LCASE(STR(?label)), LCASE("{concept}")))
                        }} LIMIT 15
                        """,
                        'description': f'Subclasses of classes related to "{concept}"'
                    })
                    
                    # Find superclasses
                    queries.append({
                        'type': 'superclasses',
                        'query': f"""
                        SELECT ?superclass ?label WHERE {{
                            ?class rdf:type owl:Class .
                            ?class rdfs:subClassOf ?superclass .
                            OPTIONAL {{ ?superclass rdfs:label ?label }}
                            FILTER(CONTAINS(LCASE(STR(?class)), LCASE("{concept}")) || 
                                   CONTAINS(LCASE(STR(?label)), LCASE("{concept}")))
                        }} LIMIT 15
                        """,
                        'description': f'Superclasses of classes related to "{concept}"'
                    })
            
            elif primary_intent == 'relationship':
                if len(concepts) >= 2:
                    concept1, concept2 = concepts[0], concepts[1]
                    # Find relationships between concepts
                    queries.append({
                        'type': 'relationships',
                        'query': f"""
                        SELECT ?subject ?predicate ?object WHERE {{
                            ?subject ?predicate ?object .
                            FILTER(
                                (CONTAINS(LCASE(STR(?subject)), LCASE("{concept1}")) && 
                                 CONTAINS(LCASE(STR(?object)), LCASE("{concept2}"))) ||
                                (CONTAINS(LCASE(STR(?subject)), LCASE("{concept2}")) && 
                                 CONTAINS(LCASE(STR(?object)), LCASE("{concept1}")))
                            )
                        }} LIMIT 20
                        """,
                        'description': f'Relationships between "{concept1}" and "{concept2}"'
                    })
            
            elif primary_intent == 'count':
                for concept in concepts[:2]:
                    # Count entities related to concept
                    queries.append({
                        'type': 'count_classes',
                        'query': f"""
                        SELECT (COUNT(?class) AS ?count) WHERE {{
                            ?class rdf:type owl:Class .
                            OPTIONAL {{ ?class rdfs:label ?label }}
                            FILTER(CONTAINS(LCASE(STR(?class)), LCASE("{concept}")) || 
                                   CONTAINS(LCASE(STR(?label)), LCASE("{concept}")))
                        }}
                        """,
                        'description': f'Count of classes related to "{concept}"'
                    })
            
            return queries
            
        except Exception as e:
            logger.error(f"Error generating SPARQL queries: {e}")
            return []
    
    def _generate_comprehensive_context(self, result: Dict[str, Any]) -> str:
        """Generate comprehensive context for LLM from all processing results."""
        context_parts = []
        
        try:
            # Query information
            context_parts.append(f"User Query: {result['original_query']}")
            context_parts.append(f"Query Intent: {result['query_classification']['primary_intent']}")
            if result['query_classification']['secondary_intent']:
                context_parts.append(f"Secondary Intent: {result['query_classification']['secondary_intent']}")
            context_parts.append(f"Key Concepts: {', '.join(result['key_concepts'][:5])}")
            context_parts.append("")
            
            # GraphSparqlQAChain result (prioritize this as it's specifically designed for Q&A)
            if result.get('sparql_chain_result') and not result['sparql_chain_result'].get('error'):
                context_parts.append("=== LANGCHAIN SPARQL ANALYSIS ===")
                sparql_result = result['sparql_chain_result']
                if sparql_result.get('answer'):
                    context_parts.append(f"Generated Answer: {sparql_result['answer']}")
                if sparql_result.get('sparql_query'):
                    context_parts.append(f"SPARQL Query Used: {sparql_result['sparql_query']}")
                context_parts.append("")
            
            # Vector search results
            if result.get('vector_search_results'):
                context_parts.append("=== RELEVANT ENTITIES FROM KNOWLEDGE GRAPH ===")
                for i, entity in enumerate(result['vector_search_results'][:5], 1):
                    context_parts.append(f"{i}. {entity['local_name']} ({entity['type']})")
                    context_parts.append(f"   URI: {entity['uri']}")
                    
                    if entity.get('labels'):
                        context_parts.append(f"   Labels: {', '.join(entity['labels'])}")
                    
                    if entity.get('comments'):
                        context_parts.append(f"   Description: {' '.join(entity['comments'][:2])}")
                    
                    # Add enhanced information if available
                    if entity.get('related_entities'):
                        related_names = [rel['local_name'] for rel in entity['related_entities'][:3]]
                        context_parts.append(f"   Related to: {', '.join(related_names)}")
                    
                    if entity.get('instances'):
                        instance_names = [inst['label'] for inst in entity['instances'][:3]]
                        context_parts.append(f"   Examples: {', '.join(instance_names)}")
                    
                    context_parts.append(f"   Similarity Score: {entity.get('similarity_score', 0):.3f}")
                    context_parts.append("")
            
            # Direct SPARQL results
            if result.get('direct_sparql_results'):
                context_parts.append("=== ADDITIONAL SPARQL QUERY RESULTS ===")
                for sparql_result in result['direct_sparql_results']:
                    context_parts.append(f"{sparql_result['description']}:")
                    for i, res in enumerate(sparql_result['results'][:5], 1):
                        result_str = ", ".join([f"{k}: {v}" for k, v in res.items() if v])
                        context_parts.append(f"  {i}. {result_str}")
                    context_parts.append("")
            
            # Schema context if no specific results
            if not result.get('vector_search_results') and not result.get('sparql_chain_result'):
                context_parts.append("=== ONTOLOGY SCHEMA OVERVIEW ===")
                schema_summary = self.rdf_manager.get_schema_summary()
                context_parts.append(schema_summary)
                context_parts.append("")
            
            # Instructions for the LLM
            context_parts.append("=== INSTRUCTIONS ===")
            context_parts.append("Based on the above information from the RDF knowledge graph:")
            context_parts.append("1. Provide a comprehensive and accurate answer to the user's query")
            context_parts.append("2. Prioritize information from the LangChain SPARQL analysis if available")
            context_parts.append("3. Use the relevant entities and SPARQL results to provide detailed explanations")
            context_parts.append("4. Explain relationships and concepts clearly with examples where available")
            context_parts.append("5. If information is insufficient, acknowledge this and suggest related topics")
            context_parts.append("6. Focus on the most relevant information based on the query intent")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error generating context: {e}")
            return f"Error generating context: {e}"
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """Generate query suggestions based on partial input."""
        try:
            # Extract potential concepts from partial query
            concepts = self._extract_key_concepts_enhanced(partial_query)
            
            suggestions = []
            
            if concepts:
                main_concept = concepts[0]
                
                # Generate different types of questions
                templates = [
                    f"What is {main_concept}?",
                    f"Tell me about {main_concept}",
                    f"What properties does {main_concept} have?",
                    f"What are the subclasses of {main_concept}?",
                    f"How is {main_concept} related to other concepts?",
                    f"List all {main_concept} in the ontology",
                    f"Give me examples of {main_concept}"
                ]
                
                suggestions.extend(templates)
            
            # Add general suggestions
            general_suggestions = [
                "What classes are in this ontology?",
                "What properties are available?",
                "Show me the ontology structure",
                "How many entities are in the knowledge graph?"
            ]
            
            suggestions.extend(general_suggestions)
            
            return suggestions[:10]  # Return top 10 suggestions
            
        except Exception as e:
            logger.error(f"Error generating query suggestions: {e}")
            return []
