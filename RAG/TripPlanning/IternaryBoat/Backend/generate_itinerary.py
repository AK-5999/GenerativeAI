# step4_chat_with_memory.py

import spacy
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from typing import List, Dict
from langchain.memory import ConversationBufferMemory
import torch
from QuerryHandlerRetrieval import TravelQueryHandler, load_vectorstore

# ----------------------------------------------------------------
# Load a small LLM (efficient for 6 GB VRAM)
# ----------------------------------------------------------------
def load_llm(model_name="meta-llama/Llama-3.2-3B-Instruct"):
    print(f"🤖 Loading LLM: {model_name} ...")

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

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    print("✅ LLM loaded successfully!\n")
    return llm


# ----------------------------------------------------------------
# Conversational Travel Chatbot with Memory
# ----------------------------------------------------------------
class TravelChatBot:
    def __init__(self, vectorstore, llm, query_handler):
        self.vectorstore = vectorstore
        self.llm = llm
        self.query_handler = query_handler
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Store user preferences during conversation
        self.user_prefs = {
            "trip_type": None,      # e.g., "historical", "beach", "adventure"
            "destination": None,    # e.g., "Agra", "anywhere"
            "duration": None        # e.g., "2 days", "3"
        }
        
        self.conversation_stage = "welcome"  # Track where we are in the conversation
        self.current_itinerary_docs = []     # Store last retrieved docs

    def welcome(self):
        """Initial greeting"""
        print("\n🤖 Bot: Hi! 👋 I'm your personal travel planner.")
        print("🤖 Bot: Would you like me to help you plan a trip? (yes/no)")
        return "welcome"

    def ask_trip_type(self):
        """Ask what kind of trip"""
        print("\n🤖 Bot: Great! What type of trip are you looking for?")
        print("       (Examples: historical, beach, adventure, spiritual, nature, architecture)")
        return "trip_type"

    def ask_destination(self):
        """Ask for preferred destination"""
        print("\n🤖 Bot: Nice choice! Do you have a specific destination in mind?")
        print("       (Or say 'anywhere' if you're flexible)")
        return "destination"

    def ask_duration(self):
        """Ask for trip duration"""
        print("\n🤖 Bot: Perfect! How many days do you want to travel?")
        print("       (Examples: 2 days, 3, one week)")
        return "duration"

    def generate_itinerary(self):
        """Generate itinerary based on collected preferences"""
        # Build query from preferences
        query_parts = []
        
        if self.user_prefs["duration"]:
            query_parts.append(f"{self.user_prefs['duration']} day")
        
        if self.user_prefs["trip_type"]:
            query_parts.append(self.user_prefs["trip_type"])
        
        if self.user_prefs["destination"] and self.user_prefs["destination"].lower() != "anywhere":
            query_parts.append(f"trip to {self.user_prefs['destination']}")
        else:
            query_parts.append("trip")
        
        full_query = " ".join(query_parts)
        
        print(f"\n🔍 Searching for: {full_query}")
        
        # Retrieve relevant destinations
        retrieved_docs = self.query_handler.retrieve_destinations(full_query, top_k=5)
        self.current_itinerary_docs = retrieved_docs
        
        if not retrieved_docs:
            print("\n🤖 Bot: Sorry, I couldn't find destinations matching your preferences.")
            print("       Would you like to try different options? (yes/no)")
            return "retry"
        
        # Format retrieved context
        main = retrieved_docs[0]
        alternatives = retrieved_docs[1:3]
        
        # Get conversation history
        chat_history = self.memory.load_memory_variables({}).get("chat_history", "")
        
        # Build prompt with memory
        prompt = f"""You are a friendly and helpful travel planner assistant.

Conversation history:
{chat_history}

User's preferences:
- Trip type: {self.user_prefs['trip_type']}
- Preferred destination: {self.user_prefs['destination']}
- Duration: {self.user_prefs['duration']} days

Main recommended destination:
Destination: {main['destination']}
Duration: {main['duration']} days
Best for: {main['interests']}
State: {main['state']}

Detailed itinerary information:
{main['itinerary']}

Alternative similar destinations:
{', '.join([d['destination'] + f" ({d['duration']} days, {d['interests']})" for d in alternatives]) if alternatives else 'None'}

Task: Create a comprehensive travel itinerary with:
1. A brief introduction explaining why this destination matches their interests
2. Day-by-day breakdown (Morning, Afternoon, Evening activities)
3. Key highlights and must-visit places
4. Brief mention of 2 alternative destinations they might also like
5. A friendly closing

Keep it conversational, engaging, and well-structured.
"""

        print("\n✨ Generating your personalized itinerary...\n")
        
        response = self.llm(prompt)
        
        print("=" * 80)
        print("🗺️  YOUR PERSONALIZED TRAVEL ITINERARY")
        print("=" * 80)
        print(response)
        print("=" * 80)
        
        # Save to memory
        self.memory.save_context(
            {"input": full_query},
            {"output": response}
        )
        
        return "feedback"

    def ask_feedback(self):
        """Ask if user is satisfied"""
        print("\n🤖 Bot: What do you think about this itinerary?")
        print("       - Type 'yes' if you like it")
        print("       - Type 'alternatives' to see other destinations")
        print("       - Type 'change' to start over with new preferences")
        return "feedback"

    def show_alternatives(self):
        """Show alternative destinations from last retrieval"""
        if len(self.current_itinerary_docs) <= 1:
            print("\n🤖 Bot: I don't have other alternatives for your exact preferences.")
            print("       Would you like to change your preferences? (yes/no)")
            return "retry"
        
        # Use 2nd best option
        alt = self.current_itinerary_docs[1]
        
        prompt = f"""You are a friendly travel planner.

        User's preferences:
        - Trip type: {self.user_prefs['trip_type']}
        - Duration: {self.user_prefs['duration']} days

        Alternative destination:
        Destination: {alt['destination']}
        Duration: {alt['duration']} days
        Best for: {alt['interests']}

        Itinerary details:
        {alt['itinerary']}

        Create a concise itinerary for this alternative destination with day-by-day activities.
        Explain why this is also a great choice.
        """
        
        print("\n✨ Here's an alternative option...\n")
        response = self.llm(prompt)
        
        print("=" * 80)
        print("🗺️  ALTERNATIVE ITINERARY")
        print("=" * 80)
        print(response)
        print("=" * 80)
        
        # Remove this destination from list
        self.current_itinerary_docs.pop(1)
        
        return "feedback"

    def reset_preferences(self):
        """Reset all preferences"""
        self.user_prefs = {
            "trip_type": None,
            "destination": None,
            "duration": None
        }

    def chat(self):
        """Main conversation loop"""
        self.conversation_stage = self.welcome()
        
        while True:
            user_input = input("\n🧑 You: ").strip()
            
            # Save user message to memory
            self.memory.save_context({"input": user_input}, {"output": ""})
            
            # Exit conditions
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("\n🤖 Bot: Thank you for chatting! Have a wonderful trip! 🌍✈️")
                break
            
            # Handle conversation stages
            if self.conversation_stage == "welcome":
                if user_input.lower() in ["yes", "y", "sure", "ok"]:
                    self.conversation_stage = self.ask_trip_type()
                elif user_input.lower() in ["no", "n"]:
                    print("\n🤖 Bot: No problem! Feel free to come back anytime. 👋")
                    break
                else:
                    print("🤖 Bot: Sorry, I didn't understand. Please say 'yes' or 'no'.")
            
            elif self.conversation_stage == "trip_type":
                self.user_prefs["trip_type"] = user_input
                self.conversation_stage = self.ask_destination()
            
            elif self.conversation_stage == "destination":
                self.user_prefs["destination"] = user_input
                self.conversation_stage = self.ask_duration()
            
            elif self.conversation_stage == "duration":
                self.user_prefs["duration"] = user_input
                self.conversation_stage = self.generate_itinerary()
                if self.conversation_stage == "feedback":
                    self.conversation_stage = self.ask_feedback()
            
            elif self.conversation_stage == "feedback":
                if user_input.lower() in ["yes", "y", "perfect", "great", "love it"]:
                    print("\n🤖 Bot: Wonderful! 🎉 I hope you have an amazing trip!")
                    print("       Would you like to plan another trip? (yes/no)")
                    another = input("🧑 You: ").strip()
                    if another.lower() in ["yes", "y"]:
                        self.reset_preferences()
                        self.conversation_stage = self.ask_trip_type()
                    else:
                        print("\n🤖 Bot: Safe travels! 🌏✈️")
                        break
                
                elif user_input.lower() in ["alternatives", "other", "more"]:
                    self.conversation_stage = self.show_alternatives()
                    self.conversation_stage = self.ask_feedback()
                
                elif user_input.lower() in ["change", "restart", "new"]:
                    print("\n🤖 Bot: Sure! Let's start fresh.")
                    self.reset_preferences()
                    self.conversation_stage = self.ask_trip_type()
                
                else:
                    print("🤖 Bot: I didn't quite get that. Please say 'yes', 'alternatives', or 'change'.")
            
            elif self.conversation_stage == "retry":
                if user_input.lower() in ["yes", "y"]:
                    self.reset_preferences()
                    self.conversation_stage = self.ask_trip_type()
                else:
                    print("\n🤖 Bot: Okay! Feel free to come back anytime. 👋")
                    break


# ----------------------------------------------------------------
# Main runner
# ----------------------------------------------------------------
"""
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🌍 WELCOME TO YOUR AI TRAVEL PLANNER 🌍")
    print("="*80)
    
    print("\n🚀 Initializing system...")
    
    # Load components
    vectorstore = load_vectorstore()
    nlp = spacy.load("en_core_web_md")
    llm = load_llm()
    query_handler = TravelQueryHandler(vectorstore, nlp=nlp)
    
    print("✅ All systems ready!\n")
    
    # Start chatbot
    chatbot = TravelChatBot(vectorstore, llm, query_handler)
    chatbot.chat()
"""