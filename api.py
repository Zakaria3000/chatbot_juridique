from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from rag_system.query_rag import LegalRagSystem

app = Flask(__name__)
CORS(app)  # Permet les requêtes cross-origin

# Initialiser le système RAG une seule fois au démarrage
# Utilisez les chemins adaptés à votre environnement
rag_system = LegalRagSystem(
    model_name="llama3",  # ou le nom de votre modèle
    vector_store_path="embeddings/law_faiss_index",
    ollama_url="http://localhost:12345"
)

@app.route('/')
def serve_index():
    with open('index.html', 'r', encoding='utf-8') as file:
        return file.read()

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query', '')
    result = rag_system.query(user_query)
    return jsonify(result)

@app.route('/api/analyze-case', methods=['POST'])
def analyze_case():
    data = request.json
    case_description = data.get('case', '')
    result = rag_system.analyze_case(case_description)
    return jsonify(result)

@app.route('/api/explain-law', methods=['POST'])
def explain_law():
    data = request.json
    law_query = data.get('law', '')
    result = rag_system.explain_law(law_query)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)