from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag_system.query_rag import LegalRagSystem

app = Flask(__name__)
CORS(app)

# Initialize the LegalRagSystem once when the server starts.
try:
    legal_system = LegalRagSystem(
        model_name="llama3:latest",
        vector_store_path="embeddings/law_faiss_index",
        ollama_url="http://localhost:12345"
    )
    print("✅ Legal RAG system initialized successfully")
except Exception as e:
    print(f"🔥 Failed to initialize legal system: {str(e)}")
    legal_system = None

@app.route('/')
def serve_index():
    with open('index.html', 'r', encoding='utf-8') as file:
        return file.read()

@app.route('/api/query', methods=['POST'])
def handle_query():
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        print("Received query:", user_query)
        if len(user_query.strip()) < 3:
            return jsonify({"error": "استفسار قصير جداً"}), 400
        result = legal_system.query(user_query)
        if len(result.get('answer', '')) < 50 or "خطأ" in result.get('answer', ''):
            return jsonify({
                "error": "جودة الإجابة غير كافية",
                "details": result
            }), 502
        return jsonify(result)
    except Exception as e:
        print("Exception in /api/query:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze-case', methods=['POST'])
def handle_case_analysis():
    if not legal_system:
        return jsonify({"error": "System not initialized"}), 500
    data = request.get_json()
    case_data = data.get('case', '')
    if len(case_data.strip()) < 10:
        return jsonify({"error": "Case description too short"}), 400
    try:
        analysis = legal_system.analyze_case(case_data)
        return jsonify({
            "decision": analysis.get("decision", "No analysis generated"),
            "sources": analysis.get("sources", [])
        })
    except Exception as e:
        print("Exception in /api/analyze-case:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/api/explain-law', methods=['POST'])
def handle_law_explanation():
    if not legal_system:
        return jsonify({"error": "System not initialized"}), 500
    data = request.get_json()
    law_query = data.get('law', '')
    if len(law_query.strip()) < 5:
        return jsonify({"error": "Law query too short"}), 400
    try:
        explanation = legal_system.explain_law(law_query)
        return jsonify({
            "explanation": explanation.get("explanation", "No explanation generated"),
            "sources": explanation.get("sources", [])
        })
    except Exception as e:
        print("Exception in /api/explain-law:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)