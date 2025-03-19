from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import sys
import os

# Add the parent directory to the path so we can import the RAG system
sys.path.append (os.path.dirname (os.path.dirname (os.path.dirname (os.path.abspath (__file__)))))
from rag_system.query_rag import LegalRagSystem

# Initialize the RAG system
# Initialize the RAG system with Ollama URL
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:12345")  # URL de l'API Ollama
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "embeddings/law_faiss_index")

# Initialiser le système RAG avec l'URL Ollama
rag_system = LegalRagSystem(model_name="llama3", vector_store_path=VECTOR_STORE_PATH, ollama_url=OLLAMA_URL)


class ActionAnswerLegalQuestion (Action) :
    def name(self) -> Text :
        return "action_answer_legal_question"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]] :
        # Get the user's question
        user_question = tracker.latest_message.get ("text")

        # Use the RAG system to answer the question
        result = rag_system.query (user_question)

        # Respond to the user
        dispatcher.utter_message (text=result["answer"])

        # Set source information as a slot
        return [SlotSet ("sources", result["sources"])]


class ActionAnalyzeLegalCase (Action) :
    def name(self) -> Text :
        return "action_analyze_legal_case"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]] :
        # Get the case description
        case_description = tracker.get_slot ("case_description")

        if not case_description :
            dispatcher.utter_message (text="يرجى تقديم وصف للقضية أولاً.")
            return []

        # Use the RAG system to analyze the case
        result = rag_system.analyze_case (case_description)

        # Respond to the user
        dispatcher.utter_message (text=result["decision"])

        # Set relevant information as slots
        return [
            SlotSet ("legal_basis", result["legal_basis"]),
            SlotSet ("sources", result["sources"])
        ]


class ActionExplainLaw (Action) :
    def name(self) -> Text :
        return "action_explain_law"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]] :
        # Get the law query
        law_query = tracker.get_slot ("law_query")

        if not law_query :
            # Try to get it from the latest message
            law_query = tracker.latest_message.get ("text")

        # Use the RAG system to explain the law
        result = rag_system.explain_law (law_query)

        # Respond to the user
        dispatcher.utter_message (text=result["explanation"])

        # Set source information as a slot
        return [SlotSet ("sources", result["sources"])]


class ActionShowSources (Action) :
    def name(self) -> Text :
        return "action_show_sources"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]] :

        # Get the sources from the slot
        sources = tracker.get_slot ("sources")

        if not sources or len (sources) == 0 :
            dispatcher.utter_message (text="لم يتم استخدام أي مصادر في الإجابة السابقة.")
            return []

        # Format the sources nicely
        sources_text = "المصادر المستخدمة:\n"
        for i, source in enumerate (sources, 1) :
            sources_text += f"{i}. {source}\n"

        # Respond to the user
        dispatcher.utter_message (text=sources_text)

        return []