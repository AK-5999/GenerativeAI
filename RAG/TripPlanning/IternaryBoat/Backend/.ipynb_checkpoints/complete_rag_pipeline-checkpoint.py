# complete_rag_pipeline.py

"""
Complete RAG Pipeline for Travel Planning Chatbot
Integrates: Query Handler → Retrieval → LLM Generation → Response
"""

import re
from typing import Optional, List, Dict
import spacy
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
import torch

# ============================================================================
# 1. QUERY HANDLER (Your existing code)
# ============================================================================

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
        doc = self.nlp(query)
        locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        return locations
    
    def match_destinations(self, extracted_locations: List[str]) -> List[str]:
        """Match extracted locations with known destinations"""
        matched = []
        
        for location in extracted_locations:
            location_lower = location.lower()
            
            if location_lower in self.known_destinations:
                matched.append(location)
            else:
                for known_dest in self.known_destinations:
                    if location_lower in known_dest or known_dest in location_lower:
                        matched.append(known_dest.title())
                        break
        
        return matched
    
    def extract_query_components(self, query: str) -> Dict:
        """Extract destination, duration, and interests from natural language query"""
        query_lower = query.lower()
        
        components = {
            'destination': None,
            'duration': None,
            'interests': []
        }
        
        # Extract locations using NER
        extracted_locations = self.extract_locations(query)
        
        if extracted_locations:
            matched_destinations = self.match_destinations(extracted_locations)
            if matched_destinations:
                components['destination'] = matched_destinations
        
        # Extract duration
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
                        components['duration'] = (int(match.group(1)) + int(match.group(2))) / 2
                break
        
        # Extract interests
        for interest, keywords in self.interest_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                components['interests'].append(interest)
        
        return components
    
    def build_retrieval_query(self, user_query: str, components: Dict) -> str:
        """Build enhanced query for semantic search"""
        query_parts = []
        
        if components.get('destination'):
            query_parts.append(f"Destination: {', '.join(components['destination'])}")
        
        query_parts.append(user_query)
        
        if components.get('duration'):
            query_parts.append(f"Duration: {components['duration']} days")
        
        if components.get('interests'):
            query_parts.append(f"Interests: {', '.join(components['interests'])}")
        
        return ' '.join(query_parts)
    
    def retrieve_destinations(self, query: str, top_k: int = 5) -> List[Dict]:
        """Main retrieval function with hybrid approach"""
        components = self.extract_query_components(query)
        enhanced_query = self.build_retrieval_query(query, components)
        
        results = self.vectorstore.similarity_search_with_score(
            enhanced_query,
            k=top_k * 3
        )
        
        filtered_results = []
        for doc, score in results:
            metadata = doc.metadata
            matches = True
            boost_score = 1.0
            
            # Destination filter
            if components.get('destination'):
                doc_destination = metadata.get('destination', '').lower()
                destination_match = any(
                    dest.lower() in doc_destination or doc_destination in dest.lower()
                    for dest in components['destination']
                )
                
                if destination_match:
                    boost_score *= 0.5
                else:
                    matches = False
            
            # Duration filter
            if components.get('duration'):
                doc_duration = metadata.get('duration', 0)
                duration_diff = abs(doc_duration - components['duration'])
                
                if duration_diff > 2:
                    matches = False
                elif duration_diff <= 1:
                    boost_score *= 0.8
            
            # Interest filter
            if components.get('interests'):
                doc_interests = metadata.get('interests', '').lower()
                interest_match_count = sum(
                    1 for interest in components['interests']
                    if interest in doc_interests
                )
                
                if interest_match_count == 0:
                    boost_score *= 1.5
                else:
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
        
        filtered_results.sort(key=lambda x: x['score'])
        return filtered_results[:top_k], components


# ============================================================================
# 2. LLM SETUP (Optimized for 6GB CUDA)
# ============================================================================

def setup_llm(model_name="meta-llama/Llama-3.2-3B-Instruct"):
    """Setup quantized LLM for 6GB CUDA constraint"""
    print("🔧 Loading LLM with 4-bit quantization...")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    print("✅ LLM loaded successfully!")
    
    return llm


# ============================================================================
# 3. PROMPT TEMPLATES
# ============================================================================

ITINERARY_PROMPT = PromptTemplate(
    input_variables=["query", "primary_destination", "similar_destinations", "user_preferences"],
    template="""You are an expert travel planner assistant. Based on the retrieved destination information, create a comprehensive travel itinerary.

User Query: {query}

User Preferences:
{user_preferences}

PRIMARY DESTINATION (Most Relevant):
{primary_destination}

SIMILAR/ALTERNATIVE DESTINATIONS:
{similar_destinations}

Please provide:

1. **Recommended Primary Destination**: 
   - Why this destination matches the user's requirements
   - Key highlights and attractions

2. **Detailed Day-by-Day Itinerary** for the primary destination:
   - Morning, Afternoon, and Evening activities
   - Estimated time for each activity
   - Travel tips

3. **Alternative Destinations** (2-3 similar options):
   - Brief description of each
   - Why they might also be suitable
   - Key differences from primary destination

4. **Travel Tips**:
   - Best time to visit
   - Budget considerations
   - Local recommendations

Format your response in a clear, organized manner with proper headings and bullet points.

Response:"""
)


# ============================================================================
# 4. RAG PIPELINE CLASS
# ============================================================================

class TravelRAGPipeline:
    def __init__(self, vectorstore, llm, nlp):
        """Initialize RAG pipeline with all components"""
        self.query_handler = TravelQueryHandler(vectorstore, nlp)
        self.llm = llm
        self.vectorstore = vectorstore
        
        # Memory for conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create LLM chain
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=ITINERARY_PROMPT,
            verbose=False
        )
    
    def format_destination_context(self, destination: Dict) -> str:
        """Format a single destination into readable context"""
        context = f"""
**Destination**: {destination['destination']}
**Duration**: {destination['duration']} days
**Best For**: {destination['interests']}
**State**: {destination['state']}

**Detailed Itinerary**:
{destination['itinerary']}

**Full Description**:
{destination['content']}
---
        """
        return context
    
    def format_user_preferences(self, components: Dict) -> str:
        """Format extracted user preferences"""
        prefs = []
        
        if components.get('destination'):
            prefs.append(f"- Preferred Destination: {', '.join(components['destination'])}")
        
        if components.get('duration'):
            prefs.append(f"- Trip Duration: {components['duration']} days")
        
        if components.get('interests'):
            prefs.append(f"- Interests: {', '.join(components['interests'])}")
        
        return '\n'.join(prefs) if prefs else "No specific preferences mentioned"
    
    def generate_itinerary(self, query: str, top_k: int = 5) -> Dict:
        """
        Main RAG pipeline function
        
        Args:
            query: User's natural language query
            top_k: Number of destinations to retrieve
            
        Returns:
            Dictionary with generated itinerary and retrieved destinations
        """
        print(f"\n{'='*60}")
        print(f"🔍 Processing Query: {query}")
        print('='*60)
        
        # Step 1: Retrieve destinations
        print("\n📍 Step 1: Retrieving relevant destinations...")
        retrieved_destinations, components = self.query_handler.retrieve_destinations(
            query, 
            top_k=top_k
        )
        
        if not retrieved_destinations:
            return {
                'query': query,
                'response': "I couldn't find any destinations matching your criteria. Could you provide more details or try different preferences?",
                'destinations': [],
                'components': components
            }
        
        print(f"✅ Found {len(retrieved_destinations)} destinations")
        for i, dest in enumerate(retrieved_destinations, 1):
            print(f"   {i}. {dest['destination']} (score: {dest['score']:.4f})")
        
        # Step 2: Format context
        print("\n📝 Step 2: Formatting context for LLM...")
        
        # Primary destination (best match)
        primary_dest = self.format_destination_context(retrieved_destinations[0])
        
        # Similar/alternative destinations
        similar_dests = []
        for dest in retrieved_destinations[1:]:
            similar_dests.append(self.format_destination_context(dest))
        
        similar_dests_text = '\n'.join(similar_dests) if similar_dests else "No alternative destinations found."
        
        # User preferences
        user_prefs = self.format_user_preferences(components)
        
        # Step 3: Generate response
        print("\n🤖 Step 3: Generating itinerary with LLM...")
        
        try:
            response = self.llm_chain.run(
                query=query,
                primary_destination=primary_dest,
                similar_destinations=similar_dests_text,
                user_preferences=user_prefs
            )
            
            print("✅ Itinerary generated successfully!")
            
            return {
                'query': query,
                'response': response,
                'destinations': retrieved_destinations,
                'components': components,
                'primary_destination': retrieved_destinations[0]['destination']
            }
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return {
                'query': query,
                'response': f"Error generating itinerary: {str(e)}",
                'destinations': retrieved_destinations,
                'components': components
            }
    
    def chat(self, query: str) -> str:
        """
        Simplified chat interface
        
        Args:
            query: User's question
            
        Returns:
            Generated itinerary as string
        """
        result = self.generate_itinerary(query)
        return result['response']


# ============================================================================
# 5. UTILITY FUNCTIONS
# ============================================================================

def load_vectorstore(persist_directory="./chroma_travel_db"):
    """Load existing vectorstore from disk"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print(f"✅ Loaded vectorstore from {persist_directory}")
    return vectorstore


def print_result(result: Dict):
    """Pretty print the RAG pipeline result"""
    print(f"\n{'='*80}")
    print(f"📋 QUERY: {result['query']}")
    print('='*80)
    
    print(f"\n🎯 Extracted Components:")
    print(f"   • Destination: {result['components'].get('destination', 'Not specified')}")
    print(f"   • Duration: {result['components'].get('duration', 'Not specified')} days")
    print(f"   • Interests: {', '.join(result['components'].get('interests', [])) or 'Not specified'}")
    
    print(f"\n🏆 Top Retrieved Destinations:")
    for i, dest in enumerate(result['destinations'], 1):
        print(f"   {i}. {dest['destination']} ({dest['duration']} days) - Score: {dest['score']:.4f}")
    
    print(f"\n📝 GENERATED ITINERARY:")
    print('-'*80)
    print(result['response'])
    print('='*80)


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Step 1: Load vectorstore
    print("🚀 Initializing Travel RAG Pipeline...")
    vectorstore = load_vectorstore()
    
    # Step 2: Load spaCy
    print("📚 Loading spaCy model...")
    nlp = spacy.load("en_core_web_md")
    
    # Step 3: Setup LLM
    llm = setup_llm()
    
    # Step 4: Create RAG pipeline
    print("🔗 Creating RAG pipeline...")
    rag_pipeline = TravelRAGPipeline(vectorstore, llm, nlp)
    
    print("\n✅ RAG Pipeline Ready!")
    print("="*80)
    
    # Test queries
    test_queries = [
        "I want to visit Agra for 2 days",
        "Suggest historical places for 3 days",
        "Beach destinations for a week",
        "5 day trip for architecture lovers",
        "Plan a spiritual journey for 4 days"
    ]
    
    # Process each query
    for query in test_queries:
        result = rag_pipeline.generate_itinerary(query, top_k=5)
        print_result(result)
        print("\n" + "="*80 + "\n")


# ============================================================================
# 7. INTERACTIVE MODE
# ============================================================================

def interactive_mode():
    """Interactive chat mode"""
    
    # Initialize
    print("🚀 Initializing Travel RAG Pipeline...")
    vectorstore = load_vectorstore()
    nlp = spacy.load("en_core_web_md")
    llm = setup_llm()
    rag_pipeline = TravelRAGPipeline(vectorstore, llm, nlp)
    
    print("\n✅ RAG Pipeline Ready!")
    print("="*80)
    print("💬 Interactive Travel Planning Assistant")
    print("   Type your travel query or 'quit' to exit")
    print("="*80)
    
    while True:
        user_query = input("\n🗣️  You: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye! Happy travels!")
            break
        
        if not user_query:
            continue
        
        # Generate response
        result = rag_pipeline.generate_itinerary(user_query, top_k=5)
        
        print(f"\n🤖 Assistant:\n")
        print(result['response'])
        print("\n" + "-"*80)


# ============================================================================
# 8. API-READY FUNCTION
# ============================================================================

class TravelRAGAPI:
    """API-ready wrapper for RAG pipeline"""
    
    def __init__(self, vectorstore_path="./chroma_travel_db"):
        """Initialize RAG pipeline once"""
        self.vectorstore = load_vectorstore(vectorstore_path)
        self.nlp = spacy.load("en_core_web_md")
        self.llm = setup_llm()
        self.rag_pipeline = TravelRAGPipeline(self.vectorstore, self.llm, self.nlp)
    
    def get_itinerary(self, query: str, top_k: int = 5) -> Dict:
        """
        API endpoint function
        
        Args:
            query: User's travel query
            top_k: Number of destinations to consider
            
        Returns:
            JSON-serializable dictionary
        """
        result = self.rag_pipeline.generate_itinerary(query, top_k)
        
        # Format for API response
        return {
            'status': 'success',
            'query': result['query'],
            'itinerary': result['response'],
            'primary_destination': result.get('primary_destination'),
            'retrieved_destinations': [
                {
                    'name': dest['destination'],
                    'duration': dest['duration'],
                    'interests': dest['interests'],
                    'state': dest['state'],
                    'relevance_score': dest['score']
                }
                for dest in result['destinations']
            ],
            'user_preferences': {
                'destination': result['components'].get('destination'),
                'duration': result['components'].get('duration'),
                'interests': result['components'].get('interests')
            }
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # Option 1: Run

    