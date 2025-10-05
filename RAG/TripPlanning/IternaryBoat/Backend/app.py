# app.py

import streamlit as st
import spacy
from QuerryHandlerRetrieval import TravelQueryHandler, load_vectorstore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
import torch

# ----------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------
# Custom CSS for better styling
# ----------------------------------------------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #1E88E5;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# Load Models (cached to avoid reloading)
# ----------------------------------------------------------------
@st.cache_resource
def load_llm(model_name="meta-llama/Llama-3.2-3B-Instruct"):
    """Load LLM with 4-bit quantization"""
    with st.spinner("🤖 Loading AI model... (this may take a minute)"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
    return llm

@st.cache_resource
def load_resources():
    """Load vectorstore, spacy, and query handler"""
    with st.spinner("📦 Loading travel database..."):
        vectorstore = load_vectorstore()
        nlp = spacy.load("en_core_web_md")
        query_handler = TravelQueryHandler(vectorstore, nlp=nlp)
    return vectorstore, nlp, query_handler

# ----------------------------------------------------------------
# Initialize Session State
# ----------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.conversation_stage = "welcome"
    st.session_state.user_prefs = {
        "trip_type": None,
        "destination": None,
        "duration": None
    }
    st.session_state.current_docs = []
    st.session_state.initialized = False

# ----------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/around-the-globe.png", width=80)
    st.title("🌍 Travel Planner")
    st.markdown("---")
    
    st.subheader("📋 Your Preferences")
    if st.session_state.user_prefs["trip_type"]:
        st.info(f"**Type:** {st.session_state.user_prefs['trip_type']}")
    if st.session_state.user_prefs["destination"]:
        st.info(f"**Place:** {st.session_state.user_prefs['destination']}")
    if st.session_state.user_prefs["duration"]:
        st.info(f"**Duration:** {st.session_state.user_prefs['duration']} days")
    
    st.markdown("---")
    
    if st.button("🔄 Start New Trip", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_stage = "welcome"
        st.session_state.user_prefs = {
            "trip_type": None,
            "destination": None,
            "duration": None
        }
        st.session_state.current_docs = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 How it works")
    st.markdown("""
    1. Tell me what kind of trip you want
    2. Share your destination preference
    3. Let me know your travel duration
    4. Get a personalized itinerary!
    """)

# ----------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------
def add_message(role, content):
    """Add message to chat history"""
    st.session_state.messages.append({"role": role, "content": content})

def generate_itinerary_from_prefs(llm, query_handler):
    """Generate itinerary based on collected preferences"""
    prefs = st.session_state.user_prefs
    
    # Build query
    query_parts = []
    if prefs["duration"]:
        query_parts.append(f"{prefs['duration']} day")
    if prefs["trip_type"]:
        query_parts.append(prefs["trip_type"])
    if prefs["destination"] and prefs["destination"].lower() != "anywhere":
        query_parts.append(f"trip to {prefs['destination']}")
    else:
        query_parts.append("trip")
    
    full_query = " ".join(query_parts)
    
    # Retrieve destinations
    with st.spinner("🔍 Finding best destinations for you..."):
        retrieved_docs = query_handler.retrieve_destinations(full_query, top_k=5)
    
    st.session_state.current_docs = retrieved_docs
    
    if not retrieved_docs:
        return "Sorry, I couldn't find destinations matching your preferences. Would you like to try different options?"
    
    # Format context
    main = retrieved_docs[0]
    alternatives = retrieved_docs[1:3]
    
    prompt = f"""You are a friendly and helpful travel planner assistant.

User's preferences:
- Trip type: {prefs['trip_type']}
- Preferred destination: {prefs['destination']}
- Duration: {prefs['duration']} days

Main recommended destination:
Destination: {main['destination']}
Duration: {main['duration']} days
Best for: {main['interests']}

Detailed itinerary:
{main['itinerary']}

Alternative destinations:
{', '.join([d['destination'] + f" ({d['duration']} days)" for d in alternatives]) if alternatives else 'None'}

Create a comprehensive travel itinerary with:
1. Brief intro explaining why this matches their interests
2. Day-by-day breakdown (Morning, Afternoon, Evening)
3. Key highlights
4. Brief mention of 2 alternatives
5. Friendly closing

Keep it conversational and well-structured.
"""
    
    with st.spinner("✨ Creating your personalized itinerary..."):
        response = llm(prompt)
    
    return response

def generate_alternative(llm, index=1):
    """Generate itinerary for alternative destination"""
    docs = st.session_state.current_docs
    
    if len(docs) <= index:
        return "I don't have more alternatives. Would you like to change your preferences?"
    
    alt = docs[index]
    prefs = st.session_state.user_prefs
    
    prompt = f"""You are a friendly travel planner.

        User's preferences:
        - Destination: {prefs["destination"] }
        - Trip type: {prefs['trip_type']}
        - Duration: {prefs['duration']} days

        Alternative destination:
        Destination: {alt['destination']}
        Duration: {alt['duration']} days
        Best for: {alt['interests']}

        Itinerary details:
        {alt['itinerary']}

        Create a concise itinerary with day-by-day activities.
        Explain why this is a great choice.
        """
    
    with st.spinner("✨ Generating alternative itinerary..."):
        response = llm(prompt)
    

    return response

# ----------------------------------------------------------------
# Main App
# ----------------------------------------------------------------
def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 AI Travel Planner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your personal AI assistant for planning amazing trips!</p>', unsafe_allow_html=True)
    
    # Load resources
    if not st.session_state.initialized:
        llm = load_llm()
        vectorstore, nlp, query_handler = load_resources()
        st.session_state.llm = llm
        st.session_state.query_handler = query_handler
        st.session_state.initialized = True
        st.success("✅ System ready! Let's plan your trip!")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Initial welcome message
    if st.session_state.conversation_stage == "welcome" and len(st.session_state.messages) == 0:
        welcome_msg = "Hi! 👋 I'm your personal AI travel planner. Would you like me to help you plan an amazing trip?"
        add_message("assistant", welcome_msg)
        with st.chat_message("assistant"):
            st.markdown(welcome_msg)
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Display user message
        add_message("user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process based on conversation stage
        stage = st.session_state.conversation_stage
        
        # Welcome stage
        if stage == "welcome":
            if user_input.lower() in ["yes", "y", "sure", "ok", "yeah"]:
                response = "Great! 🎉 What type of trip are you looking for?\n\n*Examples: historical, beach, adventure, spiritual, nature, architecture*"
                st.session_state.conversation_stage = "trip_type"
            else:
                response = "No problem! Feel free to come back anytime you want to plan a trip. 👋"
            
            add_message("assistant", response)
            with st.chat_message("assistant"):
                st.markdown(response)
        
        # Trip type stage
        elif stage == "trip_type":
            st.session_state.user_prefs["trip_type"] = user_input
            response = "Nice choice! 🌟 Do you have a specific destination in mind?\n\n*(Or say 'anywhere' if you're flexible)*"
            st.session_state.conversation_stage = "destination"
            
            add_message("assistant", response)
            with st.chat_message("assistant"):
                st.markdown(response)
        
        # Destination stage
        elif stage == "destination":
            st.session_state.user_prefs["destination"] = user_input
            response = "Perfect! 📅 How many days do you want to travel?\n\n*Examples: 2, 3 days, one week*"
            st.session_state.conversation_stage = "duration"
            
            add_message("assistant", response)
            with st.chat_message("assistant"):
                st.markdown(response)
        
        # Duration stage - generate itinerary
        elif stage == "duration":
            st.session_state.user_prefs["duration"] = user_input
            
            # Generate itinerary
            response = generate_itinerary_from_prefs(
                st.session_state.llm,
                st.session_state.query_handler
            )
            st.session_state.conversation_stage = "feedback"
            
            add_message("assistant", response)
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Add feedback prompt
            feedback_msg = "\n\n---\n\n**What would you like to do?**\n- Say **'yes'** if you like this itinerary\n- Say **'alternatives'** to see other destinations\n- Say **'change'** to start over"
            add_message("assistant", feedback_msg)
            with st.chat_message("assistant"):
                st.markdown(feedback_msg)
        
        # Feedback stage
        elif stage == "feedback":
            if user_input.lower() in ["yes", "y", "perfect", "great", "love it"]:
                response = "Wonderful! 🎉 I hope you have an amazing trip!\n\nWould you like to plan another trip?"
                st.session_state.conversation_stage = "welcome"
                
            elif user_input.lower() in ["alternatives", "other", "more"]:
                response = generate_alternative(st.session_state.llm, index=1)
                # Stay in feedback stage
                response += "\n\n---\n\n**What do you think?**\n- Say **'yes'** if you like this\n- Say **'alternatives'** for more options\n- Say **'change'** to start over"
                
            elif user_input.lower() in ["change", "restart", "new"]:
                response = "Sure! Let's start fresh. 🔄\n\nWhat type of trip are you looking for?"
                st.session_state.user_prefs = {
                    "trip_type": None,
                    "destination": None,
                    "duration": None
                }
                st.session_state.conversation_stage = "trip_type"
            
            else:
                response = "I didn't quite understand. Please say:\n- **'yes'** to confirm\n- **'alternatives'** for other options\n- **'change'** to start over"
            
            add_message("assistant", response)
            with st.chat_message("assistant"):
                st.markdown(response)
        
        st.rerun()

    # Quick action buttons at bottom
    if st.session_state.conversation_stage in ["trip_type", "destination", "duration"]:
        st.markdown("---")
        st.markdown("**💡 Quick suggestions:**")
        
        if st.session_state.conversation_stage == "trip_type":
            cols = st.columns(4)
            suggestions = ["Historical", "Beach", "Adventure", "Spiritual"]
            for i, sug in enumerate(suggestions):
                if cols[i].button(sug, key=f"quick_{sug}"):
                    st.session_state.messages.append({"role": "user", "content": sug})
                    st.session_state.user_prefs["trip_type"] = sug
                    response = "Nice choice! 🌟 Do you have a specific destination in mind?\n\n*(Or say 'anywhere' if you're flexible)*"
                    st.session_state.conversation_stage = "destination"
                    add_message("assistant", response)
                    st.rerun()

if __name__ == "__main__":
    main()