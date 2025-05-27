"""
Enhanced RDF Graph Manager with TTL support and SPARQL endpoint integration.
Updated to work with vector store SPARQL endpoint connections.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import rdflib
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, SKOS, FOAF
from langchain_community.graphs import RdfGraph
from langchain_community.chains.graph_qa.sparql import GraphSparqlQAChain
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)

class EnhancedRDFManager:
    """
    Enhanced RDF manager that handles TTL files and integrates with GraphSparqlQAChain.
    Can work with both local graphs and remote SPARQL endpoints.
    """
    
    def __init__(self, 
                 ontology_path: str = "data/ontology.ttl",
                 vector_store=None):
        """
        Initialize Enhanced RDF Manager.
        
        Args:
            ontology_path: Path to the ontology TTL file
            vector_store: Optional vector store instance with SPARQL endpoint
        """
        self.ontology_path = ontology_path
        self.graph = Graph()
        self.langchain_graph = None
        self.sparql_chain = None
        self.namespaces = {}
        self.schema_info = {}
        self.vector_store = vector_store
        
        # Common namespaces
        self.setup_namespaces()
        
        # Load the ontology
        self.load_ontology()
        
        # Initialize LangChain components
        self.setup_langchain_integration()
    
    def setup_namespaces(self):
        """Set up common namespaces for RDF processing."""
        self.namespaces = {
            'rdf': RDF,
            'rdfs': RDFS,
            'owl': OWL,
            'skos': SKOS,
            'foaf': FOAF
        }
        
        # Bind namespaces to graph
        for prefix, namespace in self.namespaces.items():
            self.graph.bind(prefix, namespace)
    
    def load_ontology(self):
        """Load the ontology from TTL file using rdflib."""
        try:
            if not os.path.exists(self.ontology_path):
                raise FileNotFoundError(f"Ontology file not found: {self.ontology_path}")
            
            logger.info(f"Loading ontology from {self.ontology_path}")
            
            # Use rdflib directly to parse TTL files
            self.graph.parse(self.ontology_path, format='turtle')
            
            # Extract custom namespaces from the loaded graph
            for prefix, namespace in self.graph.namespaces():
                if prefix and prefix not in self.namespaces:
                    self.namespaces[prefix] = namespace
            
            logger.info(f"Loaded {len(self.graph)} triples from ontology")
            logger.info(f"Found namespaces: {list(self.namespaces.keys())}")
            
            # Extract schema information
            self.extract_schema_info()
            
        except Exception as e:
            logger.error(f"Error loading ontology: {e}")
            raise
    
    def setup_langchain_integration(self):
        """Set up LangChain RdfGraph and GraphSparqlQAChain integration."""
        try:
            # Check if we have a vector store with SPARQL endpoint
            if (self.vector_store and 
                hasattr(self.vector_store, 'sparql_endpoint_url') and 
                self.vector_store.sparql_endpoint_url):
                
                logger.info("Using SPARQL endpoint from vector store for LangChain integration")
                
                # Use the SPARQL endpoint URL for LangChain
                try:
                    self.langchain_graph = RdfGraph(
                        source_file=None,
                        standard="sparql",
                        graph_kwargs={
                            "endpoint": self.vector_store.sparql_endpoint_url
                        }
                    )
                    logger.info("LangChain RdfGraph initialized with SPARQL endpoint")
                    return
                except Exception as e:
                    logger.warning(f"Could not use SPARQL endpoint for LangChain: {e}")
                    logger.info("Falling back to local RDF file method")
            
            # Fallback: Create a temporary RDF file for LangChain
            temp_rdf_path = self.ontology_path.replace('.ttl', '_temp.rdf')
            
            # Serialize graph to RDF/XML format (supported by LangChain)
            with open(temp_rdf_path, 'w', encoding='utf-8') as f:
                self.graph.serialize(f, format='xml')
            
            # Initialize LangChain RdfGraph with local file
            self.langchain_graph = RdfGraph(
                source_file=temp_rdf_path,
                standard="rdf",
                local_copy=temp_rdf_path
            )
            
            logger.info("LangChain RdfGraph integration initialized with local file")
            
        except Exception as e:
            logger.warning(f"Could not set up LangChain integration: {e}")
            self.langchain_graph = None
    
    def setup_sparql_chain(self, llm: AzureChatOpenAI) -> bool:
        """
        Set up the GraphSparqlQAChain for natural language querying.
        
        Args:
            llm: Azure OpenAI LLM instance
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.langchain_graph is None:
                logger.warning("LangChain graph not available, cannot create SPARQL chain")
                return False
            
            # Create GraphSparqlQAChain
            self.sparql_chain = GraphSparqlQAChain.from_llm(
                llm=llm,
                graph=self.langchain_graph,
                verbose=True,
                return_sparql_query=True
            )
            
            logger.info("GraphSparqlQAChain initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up SPARQL chain: {e}")
            return False
    
    def extract_schema_info(self):
        """Extract comprehensive schema information from the graph."""
        try:
            self.schema_info = {
                'classes': [],
                'properties': [],
                'individuals': [],
                'class_hierarchy': {},
                'property_domains': {},
                'property_ranges': {}
            }
            
            # Extract classes
            for cls in self.graph.subjects(RDF.type, OWL.Class):
                if not isinstance(cls, BNode):
                    class_info = {
                        'uri': str(cls),
                        'local_name': self._get_local_name(cls),
                        'labels': [str(label) for label in self.graph.objects(cls, RDFS.label)],
                        'comments': [str(comment) for comment in self.graph.objects(cls, RDFS.comment)],
                        'subclasses': [str(sub) for sub in self.graph.subjects(RDFS.subClassOf, cls)],
                        'superclasses': [str(sup) for sup in self.graph.objects(cls, RDFS.subClassOf)]
                    }
                    self.schema_info['classes'].append(class_info)
            
            # Extract object properties
            for prop in self.graph.subjects(RDF.type, OWL.ObjectProperty):
                if not isinstance(prop, BNode):
                    prop_info = {
                        'uri': str(prop),
                        'local_name': self._get_local_name(prop),
                        'labels': [str(label) for label in self.graph.objects(prop, RDFS.label)],
                        'comments': [str(comment) for comment in self.graph.objects(prop, RDFS.comment)],
                        'domains': [str(domain) for domain in self.graph.objects(prop, RDFS.domain)],
                        'ranges': [str(range_) for range_ in self.graph.objects(prop, RDFS.range)],
                        'type': 'ObjectProperty'
                    }
                    self.schema_info['properties'].append(prop_info)
            
            # Extract data properties
            for prop in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
                if not isinstance(prop, BNode):
                    prop_info = {
                        'uri': str(prop),
                        'local_name': self._get_local_name(prop),
                        'labels': [str(label) for label in self.graph.objects(prop, RDFS.label)],
                        'comments': [str(comment) for comment in self.graph.objects(prop, RDFS.comment)],
                        'domains': [str(domain) for domain in self.graph.objects(prop, RDFS.domain)],
                        'ranges': [str(range_) for range_ in self.graph.objects(prop, RDFS.range)],
                        'type': 'DatatypeProperty'
                    }
                    self.schema_info['properties'].append(prop_info)
            
            # Extract individuals
            for individual in self.graph.subjects(RDF.type, OWL.NamedIndividual):
                if not isinstance(individual, BNode):
                    individual_info = {
                        'uri': str(individual),
                        'local_name': self._get_local_name(individual),
                        'labels': [str(label) for label in self.graph.objects(individual, RDFS.label)],
                        'types': [str(type_) for type_ in self.graph.objects(individual, RDF.type) 
                                if type_ != OWL.NamedIndividual]
                    }
                    self.schema_info['individuals'].append(individual_info)
            
            logger.info(f"Extracted schema: {len(self.schema_info['classes'])} classes, "
                       f"{len(self.schema_info['properties'])} properties, "
                       f"{len(self.schema_info['individuals'])} individuals")
            
        except Exception as e:
            logger.error(f"Error extracting schema info: {e}")
    
    def query_sparql(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a SPARQL query on the graph or remote endpoint.
        
        Args:
            query: SPARQL query string
            
        Returns:
            List of result dictionaries
        """
        # Try to use remote SPARQL endpoint first if available
        if (self.vector_store and 
            hasattr(self.vector_store, 'execute_sparql_query') and
            self.vector_store.sparql_endpoint_url):
            
            try:
                logger.info("Executing SPARQL query on remote endpoint")
                return self.vector_store.execute_sparql_query(query)
            except Exception as e:
                logger.warning(f"Remote SPARQL query failed, trying local graph: {e}")
        
        # Fallback to local graph
        try:
            logger.info("Executing SPARQL query on local graph")
            results = []
            query_result = self.graph.query(query)
            
            for row in query_result:
                result_dict = {}
                for i, var in enumerate(query_result.vars):
                    value = row[i]
                    if value is not None:
                        if isinstance(value, URIRef):
                            result_dict[str(var)] = str(value)
                        elif isinstance(value, Literal):
                            result_dict[str(var)] = str(value)
                        else:
                            result_dict[str(var)] = str(value)
                    else:
                        result_dict[str(var)] = None
                results.append(result_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing SPARQL query on local graph: {e}")
            return []
    
    def query_with_langchain(self, question: str) -> Dict[str, Any]:
        """
        Query using LangChain's GraphSparqlQAChain.
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary with answer and SPARQL query
        """
        try:
            if self.sparql_chain is None:
                return {
                    'error': 'SPARQL chain not initialized',
                    'answer': None,
                    'sparql_query': None
                }
            
            result = self.sparql_chain.invoke({"query": question})
            
            return {
                'answer': result.get('result', ''),
                'sparql_query': result.get('sparql_query', ''),
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error querying with LangChain: {e}")
            return {
                'error': str(e),
                'answer': None,
                'sparql_query': None
            }
    
    def get_schema_summary(self) -> str:
        """Get a textual summary of the ontology schema."""
        try:
            summary_parts = []
            
            summary_parts.append("=== ONTOLOGY SCHEMA SUMMARY ===")
            summary_parts.append(f"Total Triples: {len(self.graph)}")
            summary_parts.append(f"Classes: {len(self.schema_info['classes'])}")
            summary_parts.append(f"Properties: {len(self.schema_info['properties'])}")
            summary_parts.append(f"Individuals: {len(self.schema_info['individuals'])}")
            
            # Add SPARQL endpoint info if available
            if (self.vector_store and 
                hasattr(self.vector_store, 'get_sparql_endpoint_info')):
                endpoint_info = self.vector_store.get_sparql_endpoint_info()
                if endpoint_info['status'] == 'configured':
                    summary_parts.append(f"SPARQL Endpoint: {endpoint_info['endpoint_url']}")
                    summary_parts.append(f"Authentication: {endpoint_info.get('authentication', {})}")
            
            summary_parts.append("")
            
            # List classes
            if self.schema_info['classes']:
                summary_parts.append("CLASSES:")
                for cls in self.schema_info['classes'][:10]:  # Show first 10
                    labels = f" ({', '.join(cls['labels'])})" if cls['labels'] else ""
                    summary_parts.append(f"  - {cls['local_name']}{labels}")
                if len(self.schema_info['classes']) > 10:
                    summary_parts.append(f"  ... and {len(self.schema_info['classes']) - 10} more")
                summary_parts.append("")
            
            # List properties
            if self.schema_info['properties']:
                summary_parts.append("PROPERTIES:")
                for prop in self.schema_info['properties'][:10]:  # Show first 10
                    labels = f" ({', '.join(prop['labels'])})" if prop['labels'] else ""
                    summary_parts.append(f"  - {prop['local_name']} ({prop['type']}){labels}")
                if len(self.schema_info['properties']) > 10:
                    summary_parts.append(f"  ... and {len(self.schema_info['properties']) - 10} more")
                summary_parts.append("")
            
            # List namespaces
            summary_parts.append("NAMESPACES:")
            for prefix, namespace in list(self.namespaces.items())[:10]:
                summary_parts.append(f"  {prefix}: {namespace}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating schema summary: {e}")
            return "Error generating schema summary"
    
    def get_all_entities(self) -> List[Dict[str, Any]]:
        """
        Extract all entities with comprehensive information for vector embedding.
        
        Returns:
            List of entity dictionaries with rich textual content
        """
        entities = []
        
        try:
            # Process classes
            for class_info in self.schema_info['classes']:
                entity = self._create_entity_from_class(class_info)
                if entity:
                    entities.append(entity)
            
            # Process properties
            for prop_info in self.schema_info['properties']:
                entity = self._create_entity_from_property(prop_info)
                if entity:
                    entities.append(entity)
            
            # Process individuals
            for individual_info in self.schema_info['individuals']:
                entity = self._create_entity_from_individual(individual_info)
                if entity:
                    entities.append(entity)
            
            # Add additional instances
            entities.extend(self._extract_class_instances())
            
            logger.info(f"Created {len(entities)} entities for vector embedding")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def _create_entity_from_class(self, class_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a rich entity representation from class information."""
        try:
            uri = URIRef(class_info['uri'])
            
            # Create comprehensive text description
            text_parts = []
            
            # Basic information
            text_parts.append(f"{class_info['local_name']} is a class in the ontology")
            
            # Labels and descriptions
            if class_info['labels']:
                text_parts.extend([f"Label: {label}" for label in class_info['labels']])
            
            if class_info['comments']:
                text_parts.extend(class_info['comments'])
            
            # Hierarchy information
            if class_info['superclasses']:
                superclass_names = [self._get_local_name(URIRef(sc)) for sc in class_info['superclasses']]
                text_parts.append(f"Subclass of: {', '.join(superclass_names)}")
            
            if class_info['subclasses']:
                subclass_names = [self._get_local_name(URIRef(sc)) for sc in class_info['subclasses']]
                text_parts.append(f"Has subclasses: {', '.join(subclass_names)}")
            
            # Find related properties
            related_properties = self._find_properties_for_class(class_info['uri'])
            if related_properties:
                prop_names = [prop['local_name'] for prop in related_properties]
                text_parts.append(f"Has properties: {', '.join(prop_names)}")
            
            # Create entity
            entity = {
                'uri': class_info['uri'],
                'type': 'Class',
                'local_name': class_info['local_name'],
                'namespace': self._get_namespace(uri),
                'labels': class_info['labels'],
                'comments': class_info['comments'],
                'superclasses': class_info['superclasses'],
                'subclasses': class_info['subclasses'],
                'related_properties': related_properties,
                'text_content': ". ".join(text_parts),
                'metadata': {
                    'entity_type': 'owl:Class',
                    'hierarchy_level': len(class_info['superclasses'])
                }
            }
            
            return entity
            
        except Exception as e:
            logger.error(f"Error creating entity from class {class_info.get('uri', 'unknown')}: {e}")
            return None
    
    def _create_entity_from_property(self, prop_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a rich entity representation from property information."""
        try:
            uri = URIRef(prop_info['uri'])
            
            # Create comprehensive text description
            text_parts = []
            
            # Basic information
            text_parts.append(f"{prop_info['local_name']} is a {prop_info['type']} in the ontology")
            
            # Labels and descriptions
            if prop_info['labels']:
                text_parts.extend([f"Label: {label}" for label in prop_info['labels']])
            
            if prop_info['comments']:
                text_parts.extend(prop_info['comments'])
            
            # Domain and range information
            if prop_info['domains']:
                domain_names = [self._get_local_name(URIRef(d)) for d in prop_info['domains']]
                text_parts.append(f"Domain: {', '.join(domain_names)}")
            
            if prop_info['ranges']:
                range_names = [self._get_local_name(URIRef(r)) for r in prop_info['ranges']]
                text_parts.append(f"Range: {', '.join(range_names)}")
            
            # Create entity
            entity = {
                'uri': prop_info['uri'],
                'type': prop_info['type'],
                'local_name': prop_info['local_name'],
                'namespace': self._get_namespace(uri),
                'labels': prop_info['labels'],
                'comments': prop_info['comments'],
                'domains': prop_info['domains'],
                'ranges': prop_info['ranges'],
                'text_content': ". ".join(text_parts),
                'metadata': {
                    'entity_type': f"owl:{prop_info['type']}",
                    'property_type': prop_info['type']
                }
            }
            
            return entity
            
        except Exception as e:
            logger.error(f"Error creating entity from property {prop_info.get('uri', 'unknown')}: {e}")
            return None
    
    def _create_entity_from_individual(self, individual_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a rich entity representation from individual information."""
        try:
            uri = URIRef(individual_info['uri'])
            
            # Create comprehensive text description
            text_parts = []
            
            # Basic information
            text_parts.append(f"{individual_info['local_name']} is an individual in the ontology")
            
            # Labels
            if individual_info['labels']:
                text_parts.extend([f"Label: {label}" for label in individual_info['labels']])
            
            # Types
            if individual_info['types']:
                type_names = [self._get_local_name(URIRef(t)) for t in individual_info['types']]
                text_parts.append(f"Instance of: {', '.join(type_names)}")
            
            # Find property values
            property_values = self._find_property_values_for_individual(individual_info['uri'])
            for prop, values in property_values.items():
                prop_name = self._get_local_name(URIRef(prop))
                text_parts.append(f"{prop_name}: {', '.join(values)}")
            
            # Create entity
            entity = {
                'uri': individual_info['uri'],
                'type': 'Individual',
                'local_name': individual_info['local_name'],
                'namespace': self._get_namespace(uri),
                'labels': individual_info['labels'],
                'types': individual_info['types'],
                'property_values': property_values,
                'text_content': ". ".join(text_parts),
                'metadata': {
                    'entity_type': 'owl:NamedIndividual',
                    'class_types': individual_info['types']
                }
            }
            
            return entity
            
        except Exception as e:
            logger.error(f"Error creating entity from individual {individual_info.get('uri', 'unknown')}: {e}")
            return None
    
    def _extract_class_instances(self) -> List[Dict[str, Any]]:
        """Extract instances of custom classes."""
        instances = []
        
        try:
            # Find instances of custom classes
            for class_info in self.schema_info['classes']:
                class_uri = URIRef(class_info['uri'])
                
                for instance in self.graph.subjects(RDF.type, class_uri):
                    if not isinstance(instance, BNode) and str(instance) not in [ind['uri'] for ind in self.schema_info['individuals']]:
                        instance_info = {
                            'uri': str(instance),
                            'local_name': self._get_local_name(instance),
                            'labels': [str(label) for label in self.graph.objects(instance, RDFS.label)],
                            'types': [class_info['uri']]
                        }
                        
                        entity = self._create_entity_from_individual(instance_info)
                        if entity:
                            instances.append(entity)
            
            return instances
            
        except Exception as e:
            logger.error(f"Error extracting class instances: {e}")
            return []
    
    def _find_properties_for_class(self, class_uri: str) -> List[Dict[str, Any]]:
        """Find properties that have the given class in their domain."""
        properties = []
        
        for prop_info in self.schema_info['properties']:
            if class_uri in prop_info['domains']:
                properties.append({
                    'uri': prop_info['uri'],
                    'local_name': prop_info['local_name'],
                    'type': prop_info['type'],
                    'ranges': prop_info['ranges']
                })
        
        return properties
    
    def _find_property_values_for_individual(self, individual_uri: str) -> Dict[str, List[str]]:
        """Find all property values for a given individual."""
        property_values = {}
        individual = URIRef(individual_uri)
        
        for predicate, obj in self.graph.predicate_objects(individual):
            # Skip RDF type assertions
            if predicate == RDF.type:
                continue
            
            pred_str = str(predicate)
            if pred_str not in property_values:
                property_values[pred_str] = []
            
            if isinstance(obj, Literal):
                property_values[pred_str].append(str(obj))
            elif isinstance(obj, URIRef):
                property_values[pred_str].append(self._get_local_name(obj))
        
        return property_values
    
    def _get_local_name(self, uri: URIRef) -> str:
        """Get the local name of a URI."""
        try:
            uri_str = str(uri)
            if '#' in uri_str:
                return uri_str.split('#')[-1]
            elif '/' in uri_str:
                return uri_str.split('/')[-1]
            else:
                return uri_str
        except:
            return str(uri)
    
    def _get_namespace(self, uri: URIRef) -> str:
        """Get the namespace of a URI."""
        try:
            uri_str = str(uri)
            if '#' in uri_str:
                return uri_str.split('#')[0] + '#'
            elif '/' in uri_str:
                parts = uri_str.split('/')
                return '/'.join(parts[:-1]) + '/'
            else:
                return ""
        except:
            return ""
    
    def find_related_entities(self, entity_uri: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Find entities related to a given entity up to a certain depth.
        Uses both local graph and remote SPARQL endpoint if available.
        """
        try:
            # If we have a SPARQL endpoint, try to use it for richer results
            if (self.vector_store and 
                hasattr(self.vector_store, 'execute_sparql_query') and
                self.vector_store.sparql_endpoint_url):
                
                try:
                    return self._find_related_entities_sparql(entity_uri, max_depth)
                except Exception as e:
                    logger.warning(f"SPARQL endpoint query for related entities failed: {e}")
            
            # Fallback to local graph
            return self._find_related_entities_local(entity_uri, max_depth)
            
        except Exception as e:
            logger.error(f"Error finding related entities: {e}")
            return []
    
    def _find_related_entities_sparql(self, entity_uri: str, max_depth: int) -> List[Dict[str, Any]]:
        """Find related entities using SPARQL endpoint."""
        query = f"""
        SELECT DISTINCT ?related ?predicate ?direction ?label
        WHERE {{
            {{
                <{entity_uri}> ?predicate ?related .
                BIND("outgoing" AS ?direction)
            }} UNION {{
                ?related ?predicate <{entity_uri}> .
                BIND("incoming" AS ?direction)
            }}
            FILTER(isURI(?related))
            OPTIONAL {{ ?related rdfs:label ?label }}
        }}
        LIMIT 20
        """
        
        results = self.vector_store.execute_sparql_query(query)
        
        related_entities = []
        for result in results:
            related_entities.append({
                'uri': result.get('related', ''),
                'local_name': result.get('label', self._get_local_name(URIRef(result.get('related', '')))),
                'relationship': self._get_local_name(URIRef(result.get('predicate', ''))),
                'relationship_uri': result.get('predicate', ''),
                'depth': 1,
                'direction': result.get('direction', 'outgoing')
            })
        
        return related_entities
    
    def _find_related_entities_local(self, entity_uri: str, max_depth: int) -> List[Dict[str, Any]]:
        """Find related entities using local graph."""
        related_entities = []
        visited = set()
        
        def explore_relationships(uri: str, depth: int, path: List[str]):
            if depth > max_depth or uri in visited:
                return
            
            visited.add(uri)
            uri_ref = URIRef(uri)
            
            # Find outgoing relationships
            for predicate, obj in self.graph.predicate_objects(uri_ref):
                if isinstance(obj, URIRef) and str(obj) not in visited:
                    related_entities.append({
                        'uri': str(obj),
                        'local_name': self._get_local_name(obj),
                        'relationship': self._get_local_name(predicate),
                        'relationship_uri': str(predicate),
                        'depth': depth + 1,
                        'path': path + [self._get_local_name(predicate)],
                        'direction': 'outgoing'
                    })
                    
                    if depth < max_depth:
                        explore_relationships(str(obj), depth + 1, path + [self._get_local_name(predicate)])
            
            # Find incoming relationships
            for subject, predicate in self.graph.subject_predicates(uri_ref):
                if isinstance(subject, URIRef) and str(subject) not in visited:
                    related_entities.append({
                        'uri': str(subject),
                        'local_name': self._get_local_name(subject),
                        'relationship': f"inverse_{self._get_local_name(predicate)}",
                        'relationship_uri': str(predicate),
                        'depth': depth + 1,
                        'path': path + [f"inverse_{self._get_local_name(predicate)}"],
                        'direction': 'incoming'
                    })
                    
                    if depth < max_depth:
                        explore_relationships(str(subject), depth + 1, path + [f"inverse_{self._get_local_name(predicate)}"])
        
        explore_relationships(entity_uri, 0, [])
        return related_entities


def query_sparql(self, query: str) -> List[Dict[str, Any]]:
    """
    Execute a SPARQL query on the graph or remote endpoint with improved error handling.
    
    Args:
        query: SPARQL query string
        
    Returns:
        List of result dictionaries
    """
    # Clean and validate the query first
    cleaned_query = self._clean_and_validate_sparql_query(query)
    if not cleaned_query:
        logger.error(f"Invalid SPARQL query: {query}")
        return []
    
    # Try to use remote SPARQL endpoint first if available
    if (self.vector_store and 
        hasattr(self.vector_store, 'execute_sparql_query') and
        self.vector_store.sparql_endpoint_url):
        
        try:
            logger.info("Executing SPARQL query on remote endpoint")
            logger.debug(f"Query: {cleaned_query}")
            return self.vector_store.execute_sparql_query(cleaned_query)
        except Exception as e:
            logger.warning(f"Remote SPARQL query failed: {e}")
            logger.info("Falling back to local graph")
    
    # Fallback to local graph
    try:
        logger.info("Executing SPARQL query on local graph")
        logger.debug(f"Query: {cleaned_query}")
        
        results = []
        
        # Add common prefixes to the query if not present
        query_with_prefixes = self._add_common_prefixes(cleaned_query)
        
        query_result = self.graph.query(query_with_prefixes)
        
        for row in query_result:
            result_dict = {}
            for i, var in enumerate(query_result.vars):
                value = row[i]
                if value is not None:
                    if isinstance(value, URIRef):
                        result_dict[str(var)] = str(value)
                    elif isinstance(value, Literal):
                        result_dict[str(var)] = str(value)
                    else:
                        result_dict[str(var)] = str(value)
                else:
                    result_dict[str(var)] = None
            results.append(result_dict)
        
        logger.debug(f"Local SPARQL query returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error executing SPARQL query on local graph: {e}")
        logger.error(f"Query was: {cleaned_query}")
        return []

def _clean_and_validate_sparql_query(self, query: str) -> str:
    """Clean and validate a SPARQL query."""
    if not query or not query.strip():
        return ""
    
    # Remove excessive whitespace and normalize
    import re
    query = re.sub(r'\s+', ' ', query.strip())
    
    # Basic validation - must contain SELECT, ASK, CONSTRUCT, or DESCRIBE
    query_upper = query.upper()
    if not any(keyword in query_upper for keyword in ['SELECT', 'ASK', 'CONSTRUCT', 'DESCRIBE']):
        logger.error("Query must contain SELECT, ASK, CONSTRUCT, or DESCRIBE")
        return ""
    
    # Check for balanced braces
    if query.count('{') != query.count('}'):
        logger.error("Unbalanced braces in SPARQL query")
        return ""
    
    # Fix common issues
    query = self._fix_common_sparql_issues(query)
    
    return query

def _fix_common_sparql_issues(self, query: str) -> str:
    """Fix common SPARQL syntax issues."""
    import re
    
    # Fix missing dots before closing braces
    query = re.sub(r'([^\s\.])\s*}', r'\1 . }', query)
    
    # Fix missing spaces around operators
    query = re.sub(r'([<>=!])([^=\s])', r'\1 \2', query)
    query = re.sub(r'([^=\s])([<>=!])', r'\1 \2', query)
    
    # Fix FILTER syntax
    query = re.sub(r'FILTER\s*\(([^)]+)\)', lambda m: f"FILTER({m.group(1).strip()})", query)
    
    # Ensure proper URI brackets
    query = re.sub(r'([^<\s])([a-zA-Z][a-zA-Z0-9]*:[a-zA-Z][a-zA-Z0-9_-]*)', r'\1 \2', query)
    
    return query

def _add_common_prefixes(self, query: str) -> str:
    """Add common prefixes to a SPARQL query if they're not already present."""
    prefixes = []
    
    # Check if query already has PREFIX declarations
    if 'PREFIX' not in query.upper():
        # Add common prefixes
        common_prefixes = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'owl': 'http://www.w3.org/2002/07/owl#',
            'xsd': 'http://www.w3.org/2001/XMLSchema#'
        }
        
        # Add prefixes that are used in the query
        for prefix, namespace in common_prefixes.items():
            if f'{prefix}:' in query:
                prefixes.append(f'PREFIX {prefix}: <{namespace}>')
        
        # Add custom namespaces from the graph
        for prefix, namespace in self.namespaces.items():
            if prefix and f'{prefix}:' in query and prefix not in ['rdf', 'rdfs', 'owl', 'xsd']:
                prefixes.append(f'PREFIX {prefix}: <{namespace}>')
    
    if prefixes:
        return '\n'.join(prefixes) + '\n\n' + query
    return query

def query_with_langchain(self, question: str) -> Dict[str, Any]:
    """
    Query using LangChain's GraphSparqlQAChain with improved error handling.
    
    Args:
        question: Natural language question
        
    Returns:
        Dictionary with answer and SPARQL query
    """
    try:
        if self.sparql_chain is None:
            return {
                'error': 'SPARQL chain not initialized',
                'answer': None,
                'sparql_query': None
            }
        
        logger.info(f"Processing question with LangChain: {question}")
        
        # Try the query with error handling
        result = self.sparql_chain.invoke({"query": question})
        
        # Extract and validate the generated SPARQL query
        sparql_query = result.get('sparql_query', '')
        if sparql_query:
            # Clean and validate the generated query
            cleaned_query = self._clean_and_validate_sparql_query(sparql_query)
            if not cleaned_query:
                logger.warning("LangChain generated invalid SPARQL query")
                return {
                    'error': 'Generated SPARQL query is invalid',
                    'answer': result.get('result', ''),
                    'sparql_query': sparql_query,
                    'cleaned_query': cleaned_query
                }
            
            # Test the query if it's different from the original
            if cleaned_query != sparql_query:
                logger.info("Testing cleaned SPARQL query...")
                try:
                    test_results = self.query_sparql(cleaned_query)
                    logger.info(f"Cleaned query executed successfully with {len(test_results)} results")
                except Exception as test_error:
                    logger.warning(f"Cleaned query still failed: {test_error}")
        
        return {
            'answer': result.get('result', ''),
            'sparql_query': sparql_query,
            'cleaned_query': cleaned_query if sparql_query else None,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Error querying with LangChain: {e}")
        
        # Try to extract more specific error information
        error_msg = str(e)
        if "QueryBadFormed" in error_msg:
            return {
                'error': 'SPARQL query syntax error - query is malformed',
                'answer': None,
                'sparql_query': None,
                'suggestion': 'Try rephrasing your question or using simpler terms'
            }
        elif "timeout" in error_msg.lower():
            return {
                'error': 'Query timeout - question may be too complex',
                'answer': None,
                'sparql_query': None,
                'suggestion': 'Try asking a more specific question'
            }
        else:
            return {
                'error': f'Query execution failed: {error_msg}',
                'answer': None,
                'sparql_query': None
            }

def test_sparql_endpoint(self) -> Dict[str, Any]:
    """Test SPARQL endpoint with a simple, safe query."""
    try:
        # Simple test query that should work on any RDF endpoint
        test_query = """
        SELECT ?s ?p ?o 
        WHERE { 
            ?s ?p ?o 
        } 
        LIMIT 1
        """
        
        logger.info("Testing SPARQL endpoint with simple query...")
        results = self.query_sparql(test_query)
        
        if results:
            return {
                'success': True,
                'message': f'SPARQL endpoint working - returned {len(results)} results',
                'sample_result': results[0] if results else None
            }
        else:
            return {
                'success': False,
                'message': 'SPARQL endpoint returned no results',
                'query_used': test_query
            }
            
    except Exception as e:
        logger.error(f"SPARQL endpoint test failed: {e}")
        return {
            'success': False,
            'message': f'SPARQL endpoint test failed: {e}',
            'error_type': type(e).__name__
        }

def validate_sparql_query_syntax(self, query: str) -> Dict[str, Any]:
    """Validate SPARQL query syntax without executing it."""
    try:
        from rdflib.plugins.sparql import prepareQuery
        
        # Clean the query first
        cleaned_query = self._clean_and_validate_sparql_query(query)
        if not cleaned_query:
            return {
                'valid': False,
                'error': 'Query failed basic validation',
                'original_query': query
            }
        
        # Add prefixes
        query_with_prefixes = self._add_common_prefixes(cleaned_query)
        
        # Try to parse the query
        try:
            prepared_query = prepareQuery(query_with_prefixes)
            return {
                'valid': True,
                'message': 'Query syntax is valid',
                'cleaned_query': cleaned_query,
                'query_with_prefixes': query_with_prefixes
            }
        except Exception as parse_error:
            return {
                'valid': False,
                'error': f'SPARQL parsing error: {parse_error}',
                'cleaned_query': cleaned_query,
                'query_with_prefixes': query_with_prefixes
            }
            
    except Exception as e:
        return {
            'valid': False,
            'error': f'Query validation failed: {e}',
            'original_query': query
        }
