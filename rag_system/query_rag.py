import os
import re
import requests
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class ArabertEmbeddings:
    def _init_(self, model_name="aubmindlab/bert-base-arabertv2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def get_embeddings(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Use CLS token embedding as document embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            embeddings.append(embedding)
        return embeddings

    def embed_documents(self, texts):
        return self.get_embeddings(texts)

    def embed_query(self, text):
        return self.get_embeddings([text])[0]

    def _call_(self, text):
        if isinstance(text, list):
            return self.embed_documents(text)
        else:
            return self.embed_query(text)

class LegalRagSystem:
    def _init_(self, model_name="llama3:latest", vector_store_path="embeddings/law_faiss_index",
                 ollama_url="http://localhost:12345"):
        self.model_name = model_name
        self.ollama_url = ollama_url

        # Test connection to Ollama
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                print(f"⚠ Attention: Impossible de se connecter à Ollama. Statut: {response.status_code}")
            else:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                if model_name not in model_names:
                    print(f"⚠ Le modèle {model_name} n'est pas disponible dans Ollama. Modèles disponibles: {model_names}")
                else:
                    print(f"✅ Connecté à Ollama. Modèle {model_name} disponible.")
        except Exception as e:
            print(f"⚠ Erreur lors de la connexion à Ollama: {e}")
            print("Assurez-vous qu'Ollama est en cours d'exécution avec la commande 'ollama serve'")

        # If the model is not available, attempt to download it.
        if 'model_names' in locals() and model_name not in model_names:
            print(f"⏳ Downloading {model_name}...")
            try:
                requests.post(f"{self.ollama_url}/api/pull", json={"name": model_name})
                print(f"✅ Model {model_name} downloaded")
            except Exception as e:
                print(f"🚨 Failed to download model: {e}")
                raise

        # Initialize AraBERT embeddings
        self.embeddings = ArabertEmbeddings()

        # Load vector store
        try:
            self.vector_store = FAISS.load_local(vector_store_path, self.embeddings)
            print(f"✅ Base de données vectorielle chargée depuis {vector_store_path}")
        except Exception as e:
            print(f"⚠ Erreur lors du chargement de la base de données vectorielle: {e}")
            print("Tentative de reconstruction de la base...")
            # You can add code here to rebuild the vector store if needed.
            raise

    def _generate_with_ollama(self, prompt, max_tokens=2048, temperature=0.3, top_p=0.7):
        try:
            truncated_prompt = prompt[:12000]
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": truncated_prompt,
                    "stream": False,
                    "num_predict": max_tokens,
                    "temperature":0.3,
                    "top_p": 0.7,
                    "repeat_penalty":1.5,
                    "stop":["\nEnglish","User:","assistant"]


                },

            )
            response.raise_for_status()
            if 'response' not in response.json():
                raise ValueError("Invalid Ollama response format")
            return response.json()['response']
        except requests.exceptions.HTTPError as e:
            print(f"Ollama API Error: {e.response.status_code} - {e.response.text}")
            return "خطأ في الخادم الداخلي"
        except Exception as e:
            print(f"Generation Failed: {str(e)}")
            return "فشل توليد الإجابة"

    def query(self, user_query, k=5):
        MAX_CONTEXT_LENGTH = 10000  # Characters
        relevant_docs = self.vector_store.similarity_search(user_query, k=k)
        context = "\n\n".join([d.page_content[:2000] for d in relevant_docs])[:MAX_CONTEXT_LENGTH]
        prompt = f"""
        [السياق القانوني]
        {context}

        [السؤال]
        {user_query}
        """
        for attempt in range(3):
            try:
                answer = self._generate_with_ollama(prompt)
                if len(answer) > 100:
                    return {
                        "answer": answer,
                        "sources": [d.metadata.get("source") for d in relevant_docs]
                    }
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                continue
        return {"answer": "تعذر توليد إجابة مناسبة", "sources": []}

    def analyze_case(self, case_description):
        relevant_docs = self.vector_store.similarity_search(case_description, k=7)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"""
        أنت قاضٍ متمرس في القانون العربي. يرجى تحليل الحالة المعروضة وإصدار قرار قضائي مستند على القوانين والسوابق القضائية ذات الصلة.

        النصوص القانونية ذات الصلة:
        {context}

        وصف الحالة:
        {case_description}

        القرار القضائي:
        """
        decision = self._generate_with_ollama(prompt, max_tokens=2048, temperature=0.2, top_p=0.9)
        return {
            "decision": decision,
            "legal_basis": [doc.page_content for doc in relevant_docs[:3]],
            "sources": [doc.metadata.get("source", "Unknown") for doc in relevant_docs]
        }

    def explain_law(self, law_query):
        relevant_docs = self.vector_store.similarity_search(law_query, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"""
        أنت خبير قانوني متخصص في شرح القوانين والتشريعات. يرجى شرح القانون أو اللائحة المطلوبة بلغة واضحة وسهلة الفهم.

        النصوص القانونية ذات الصلة:
        {context}

        الاستفسار حول القانون:
        {law_query}

        الشرح:
        """
        explanation = self._generate_with_ollama(prompt, max_tokens=1536, temperature=0.3, top_p=0.9)
        return {
            "explanation": explanation,
            "sources": [doc.metadata.get("source", "Unknown") for doc in relevant_docs]
        }

    # Additional helper methods can be added below (e.g. validate_index, sanitize functions, etc.)