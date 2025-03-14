import os
import requests
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel


class ArabertEmbeddings :
    def __init__(self, model_name="aubmindlab/bert-base-arabertv2") :
        self.tokenizer = AutoTokenizer.from_pretrained (model_name)
        self.model = AutoModel.from_pretrained (model_name)
        self.device = "cuda" if torch.cuda.is_available ( ) else "cpu"
        self.model.to (self.device)
        print (f"Using device: {self.device}")

    def get_embeddings(self, texts) :
        embeddings = []
        for text in texts :
            inputs = self.tokenizer (text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k : v.to (self.device) for k, v in inputs.items ( )}

            with torch.no_grad ( ) :
                outputs = self.model (**inputs)

            # Use CLS token embedding as document embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu ( ).numpy ( )[0]
            embeddings.append (embedding)

        return embeddings

    # Make the class compatible with LangChain's expected interface
    def embed_documents(self, texts) :
        return self.get_embeddings (texts)

    def embed_query(self, text) :
        return self.get_embeddings ([text])[0]

    # Add this method to make the class callable as expected by LangChain
    def __call__(self, text) :
        if isinstance (text, list) :
            return self.embed_documents (text)
        else :
            return self.embed_query (text)

class LegalRagSystem :
    def __init__(self, model_name="llama3", vector_store_path="embeddings/law_faiss_index",
                 ollama_url="http://localhost:12345") :
        # Use Ollama API endpoint instead of loading model directly
        self.model_name = model_name
        self.ollama_url = ollama_url

        # Test connection to Ollama
        try :
            response = requests.get (f"{self.ollama_url}/api/tags")
            if response.status_code != 200 :
                print (f"⚠️ Attention: Impossible de se connecter à Ollama. Statut: {response.status_code}")
            else :
                models = response.json ( ).get ("models", [])
                model_names = [model.get ("name") for model in models]
                if model_name not in model_names :
                    print (
                        f"⚠️ Le modèle {model_name} n'est pas disponible dans Ollama. Modèles disponibles: {model_names}")
                else :
                    print (f"✅ Connecté à Ollama. Modèle {model_name} disponible.")
        except Exception as e :
            print (f"⚠️ Erreur lors de la connexion à Ollama: {e}")
            print ("Assurez-vous qu'Ollama est en cours d'exécution avec la commande 'ollama serve'")

        # Initialize AraBERT embeddings
        self.embeddings = ArabertEmbeddings ( )

        # Load vector store
        try :
            self.vector_store = FAISS.load_local (vector_store_path, self.embeddings,
                                                  allow_dangerous_deserialization=True)
            print (f"✅ Base de données vectorielle chargée depuis {vector_store_path}")
        except Exception as e :
            print (f"⚠️ Erreur lors du chargement de la base de données vectorielle: {e}")
            print ("Tentative de reconstruction de la base...")
            # Add code here to rebuild the vector store if needed
            raise

    def _generate_with_ollama(self, prompt, max_tokens=1024, temperature=0.1, top_p=0.9) :
        """
        Generate text using Ollama API
        """
        try :
            response = requests.post (
                f"{self.ollama_url}/api/generate",
                json={
                    "model" : self.model_name,
                    "prompt" : prompt,
                    "stream" : False,
                    "options" : {
                        "num_predict" : max_tokens,
                        "temperature" : temperature,
                        "top_p" : top_p
                    }
                }
            )

            if response.status_code == 200 :
                return response.json ( ).get ("response", "")
            else :
                print (f"⚠️ Erreur lors de la génération: {response.status_code}")
                return "Erreur lors de la génération de la réponse."
        except Exception as e :
            print (f"⚠️ Exception lors de la génération: {e}")
            return f"Erreur: {str (e)}"

    def query(self, user_query, k=5) :
        """
        Query the RAG system with a user question
        """
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search (user_query, k=k)

        # Format context for Llama
        context = "\n\n".join ([doc.page_content for doc in relevant_docs])

        # Create prompt for Llama
        prompt = f"""
        أنت مساعد قانوني ذكي متخصص في القانون العربي. يرجى استخدام المعلومات القانونية أدناه للإجابة على السؤال.

        المعلومات القانونية:
        {context}

        السؤال: {user_query}

        الإجابة:
        """

        # Generate response using Ollama
        answer = self._generate_with_ollama (prompt, max_tokens=1024, temperature=0.1, top_p=0.9)

        return {
            "answer" : answer,
            "sources" : [doc.metadata.get ("source", "Unknown") for doc in relevant_docs]
        }

    def analyze_case(self, case_description) :
        """
        Analyze a legal case and provide a judge-like decision
        """
        # Retrieve relevant legal texts
        relevant_docs = self.vector_store.similarity_search (case_description, k=7)

        # Format context for Llama
        context = "\n\n".join ([doc.page_content for doc in relevant_docs])

        # Create prompt for Llama
        prompt = f"""
        أنت قاضٍ متمرس في القانون العربي. يرجى تحليل الحالة المعروضة وإصدار قرار قضائي مستند على القوانين والسوابق القضائية ذات الصلة.

        النصوص القانونية ذات الصلة:
        {context}

        وصف الحالة:
        {case_description}

        القرار القضائي:
        """

        # Generate response using Ollama
        decision = self._generate_with_ollama (prompt, max_tokens=2048, temperature=0.2, top_p=0.9)

        return {
            "decision" : decision,
            "legal_basis" : [doc.page_content for doc in relevant_docs[:3]],
            "sources" : [doc.metadata.get ("source", "Unknown") for doc in relevant_docs]
        }

    def explain_law(self, law_query) :
        """
        Explain a specific law or regulation
        """
        # Retrieve relevant legal texts
        relevant_docs = self.vector_store.similarity_search (law_query, k=3)

        # Format context for Llama
        context = "\n\n".join ([doc.page_content for doc in relevant_docs])

        # Create prompt for Llama
        prompt = f"""
        أنت خبير قانوني متخصص في شرح القوانين والتشريعات. يرجى شرح القانون أو اللائحة المطلوبة بلغة واضحة وسهلة الفهم.

        النصوص القانونية ذات الصلة:
        {context}

        الاستفسار حول القانون:
        {law_query}

        الشرح:
        """

        # Generate response using Ollama
        explanation = self._generate_with_ollama (prompt, max_tokens=1536, temperature=0.3, top_p=0.9)

        return {
            "explanation" : explanation,
            "sources" : [doc.metadata.get ("source", "Unknown") for doc in relevant_docs]
        }