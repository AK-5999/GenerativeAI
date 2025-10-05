#!python -m spacy download en_core_web_sm
#!python -m spacy download en_core_web_md
import re
from typing import Optional, List, Dict
import spacy
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class TravelQueryHandler:
    def __init__(self, vectorstore, nlp=None):
        self.vectorstore = vectorstore
        
        # Load spaCy model if not provided
        if nlp is None:
            try:
                self.nlp = spacy.load("en_core_web_md")
            except:
                print("⚠️ Downloading spaCy model...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp
        
        # Interest keywords mapping
        self.interest_keywords = {
            'history': ['history', 'historical', 'heritage', 'ancient', 'monuments'],
            'architecture': ['architecture', 'buildings', 'monuments', 'structures'],
            'nature': ['nature', 'natural', 'wildlife', 'scenic', 'mountains', 'beaches'],
            'adventure': ['adventure', 'trekking', 'hiking', 'sports', 'thrilling'],
            'spiritual': ['spiritual', 'religious', 'temples', 'pilgrimage', 'sacred'],
            'beaches': ['beach', 'beaches', 'coastal', 'sea', 'ocean'],
            'food': ['food', 'cuisine', 'culinary', 'dining', 'restaurants']
        }
        
        # Get all destinations from vectorstore for matching
        self.known_destinations = self._load_known_destinations()
    
    def _load_known_destinations(self) -> List[str]:
        """Extract all destination names from vectorstore"""
        try:
            # Get all documents to extract destination names
            all_docs = self.vectorstore.get()
            destinations = set()
            
            if all_docs and 'metadatas' in all_docs:
                for metadata in all_docs['metadatas']:
                    if 'destination' in metadata:
                        destinations.add(metadata['destination'].lower())
            
            return list(destinations)
        except Exception as e:
            print(f"⚠️ Could not load destinations: {e}")
            return []
    
    def extract_locations(self, query: str) -> List[str]:
        """Extract locations using spaCy NER"""
        # Process the text with spaCy (FIXED: was 'text', now 'query')
        doc = self.nlp(query)
        
        # Extract locations (GPE = Geopolitical Entity, LOC = Location)
        locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        
        return locations
    
    def match_destinations(self, extracted_locations: List[str]) -> List[str]:
        """Match extracted locations with known destinations"""
        matched = []
        
        for location in extracted_locations:
            location_lower = location.lower()
            
            # Exact match
            if location_lower in self.known_destinations:
                matched.append(location)
            else:
                # Fuzzy match (partial)
                for known_dest in self.known_destinations:
                    if location_lower in known_dest or known_dest in location_lower:
                        matched.append(known_dest.title())
                        break
        
        return matched
    
    def extract_query_components(self, query: str) -> Dict:
        """
        Extract destination, duration, and interests from natural language query
        """
        query_lower = query.lower()
        
        components = {
            'destination': None,
            'duration': None,
            'interests': []
        }
        
        # Extract locations using NER
        extracted_locations = self.extract_locations(query)
        
        # Match with known destinations
        if extracted_locations:
            matched_destinations = self.match_destinations(extracted_locations)
            if matched_destinations:
                components['destination'] = matched_destinations  # Keep as list
        
        # Extract duration (e.g., "3 days", "2-3 days", "one week")
        duration_patterns = [
            r'(\d+)\s*(?:days?|nights?)',
            r'(\d+)-(\d+)\s*(?:days?|nights?)',
            r'(one|two|three|four|five|six|seven)\s*(?:days?|weeks?)'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if 'week' in match.group(0):
                    duration_map = {'one': 7, 'two': 14, 'three': 21}
                    components['duration'] = duration_map.get(match.group(1), 7)
                else:
                    try:
                        components['duration'] = int(match.group(1))
                    except:
                        # Handle range (e.g., "2-3 days"), take average
                        components['duration'] = (int(match.group(1)) + int(match.group(2))) / 2
                break
        
        # Extract interests
        for interest, keywords in self.interest_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                components['interests'].append(interest)
        
        return components
    
    def build_retrieval_query(self, user_query: str, components: Dict) -> str:
        """
        Build enhanced query for semantic search
        """
        query_parts = []
        
        # Add destination explicitly
        if components.get('destination'):
            query_parts.append(f"Destination: {', '.join(components['destination'])}")
        
        # Add original query
        query_parts.append(user_query)
        
        # Add duration
        if components.get('duration'):
            query_parts.append(f"Duration: {components['duration']} days")
        
        # Add interests
        if components.get('interests'):
            query_parts.append(f"Interests: {', '.join(components['interests'])}")
        
        return ' '.join(query_parts)
    
    def retrieve_destinations(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Main retrieval function with hybrid approach
        """
        # Extract query components
        components = self.extract_query_components(query)
        #print(f"📋 Extracted Components: {components}")
        
        # Build enhanced query
        enhanced_query = self.build_retrieval_query(query, components)
        #print(f"🔍 Enhanced Query: {enhanced_query}")
        
        # Perform semantic search (retrieve more for filtering)
        results = self.vectorstore.similarity_search_with_score(
            enhanced_query,
            k=top_k * 3  # Retrieve extra for filtering
        )
        
        # Post-process and filter results
        filtered_results = []
        for doc, score in results:
            metadata = doc.metadata
            
            # Apply custom filtering logic
            matches = True
            boost_score = 1.0
            
            # Destination filter (STRONG filter if specified)
            if components.get('destination'):
                doc_destination = metadata.get('destination', '').lower()
                destination_match = any(
                    dest.lower() in doc_destination or doc_destination in dest.lower()
                    for dest in components['destination']
                )
                
                if destination_match:
                    boost_score *= 0.5  # Boost exact destination matches
                else:
                    # If destination specified but doesn't match, skip OR penalize heavily
                    matches = False  # Strict filtering
                    # OR: boost_score *= 2.0  # Soft filtering (penalize)
            
            # Duration filter (with flexibility)
            if components.get('duration'):
                doc_duration = metadata.get('duration', 0)
                duration_diff = abs(doc_duration - components['duration'])
                
                if duration_diff > 2:  # More than 2 days difference
                    matches = False
                elif duration_diff <= 1:  # Within 1 day
                    boost_score *= 0.8  # Slight boost
            
            # Interest filter
            if components.get('interests'):
                doc_interests = metadata.get('interests', '').lower()
                interest_match_count = sum(
                    1 for interest in components['interests']
                    if interest in doc_interests
                )
                
                if interest_match_count == 0:
                    boost_score *= 1.5  # Penalize no interest match
                else:
                    # Boost based on number of matching interests
                    boost_score *= (1.0 / (interest_match_count + 1))
            
            if matches:
                final_score = score * boost_score
                filtered_results.append({
                    'destination': metadata.get('destination'),
                    'duration': metadata.get('duration'),
                    'interests': metadata.get('interests'),
                    'state': metadata.get('state'),
                    'content': doc.page_content,
                    'score': final_score,
                    'original_score': score,
                    'itinerary': metadata.get('itinerary_summary')
                })
        
        # Sort by score and return top K
        filtered_results.sort(key=lambda x: x['score'])
        return filtered_results[:top_k]


def load_vectorstore(persist_directory="./chroma_travel_db"):
    """Load existing vectorstore from disk"""
    
    # Create same embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load vectorstore
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore

vectorstore = load_vectorstore()
nlp = spacy.load("en_core_web_md")
# Usage
query_handler = TravelQueryHandler(vectorstore,nlp=nlp)
# Test different query types
queries = [
    "I want to visit Agra for 2 days",
    "Suggest historical places for 3 days",
    "Beach destinations",
    "5 day trip for architecture lovers",
    "Where can I go for a week?"
]
for query in queries:
    print(f"\n🔍 Query: {query}")
    results = query_handler.retrieve_destinations(query, top_k=5)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['destination']} ({result['duration']} days)")
        print(f"   Interests: {result['interests']}")
        print(f"   Score: {result['score']:.4f}")