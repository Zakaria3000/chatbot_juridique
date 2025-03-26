# test_rag.py
from rag_system.query_rag import LegalRagSystem

legal_system = LegalRagSystem(
    model_name="phi3:mini",
    vector_store_path="embeddings/law_faiss_index",
    ollama_url="http://localhost:12345"
)
result = legal_system.query("ما هو الفصل 41 ؟")
print(result)
