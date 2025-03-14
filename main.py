import os
import argparse
from rag_system.query_rag import LegalRagSystem


def test_query(query, llama_path, vector_store_path) :
    """
    Test the RAG system with a single query
    """
    print (f"\n🔍 Testing query: {query}")
    rag_system = LegalRagSystem (llama_path, vector_store_path)

    result = rag_system.query (query)

    print ("\n📝 Answer:")
    print (result["answer"])

    print ("\n📚 Sources:")
    for source in result["sources"] :
        print (f"- {source}")

    return result


def test_case_analysis(case_description, llama_path, vector_store_path) :
    """
    Test the case analysis functionality
    """
    print (f"\n⚖️ Testing case analysis: {case_description[:100]}...")
    rag_system = LegalRagSystem (llama_path, vector_store_path)

    result = rag_system.analyze_case (case_description)

    print ("\n📜 Decision:")
    print (result["decision"])

    print ("\n📚 Legal basis:")
    for i, basis in enumerate (result["legal_basis"], 1) :
        print (f"{i}. {basis[:200]}...")

    print ("\n📚 Sources:")
    for source in result["sources"] :
        print (f"- {source}")

    return result


def test_law_explanation(law_query, llama_path, vector_store_path) :
    """
    Test the law explanation functionality
    """
    print (f"\n📖 Testing law explanation: {law_query}")
    rag_system = LegalRagSystem (llama_path, vector_store_path)

    result = rag_system.explain_law (law_query)

    print ("\n📝 Explanation:")
    print (result["explanation"])

    print ("\n📚 Sources:")
    for source in result["sources"] :
        print (f"- {source}")

    return result


def interactive_mode(llama_path, vector_store_path) :
    """
    Run an interactive session with the RAG system
    """
    print ("\n🤖 Starting interactive mode with the legal chatbot")
    print ("Type 'exit' to quit, 'case' to analyze a case, 'law' to explain a law, or any question for general queries")

    rag_system = LegalRagSystem (llama_path, vector_store_path)

    while True :
        user_input = input ("\n👤 You: ")

        if user_input.lower ( ) == "exit" :
            print ("👋 Goodbye!")
            break

        elif user_input.lower ( ) == "case" :
            print ("📝 Please describe the legal case (type on multiple lines, end with END on a separate line):")
            case_lines = []
            while True :
                line = input ( )
                if line == "END" :
                    break
                case_lines.append (line)

            case_description = "\n".join (case_lines)
            result = rag_system.analyze_case (case_description)

            print ("\n⚖️ Decision:")
            print (result["decision"])

        elif user_input.lower ( ) == "law" :
            law_query = input ("📚 Which law would you like me to explain? ")
            result = rag_system.explain_law (law_query)

            print ("\n📖 Explanation:")
            print (result["explanation"])

        else :
            result = rag_system.query (user_input)

            print ("\n🤖 Bot:")
            print (result["answer"])


if __name__ == "__main__" :
    parser = argparse.ArgumentParser (description="Legal Chatbot Testing Tool")
    parser.add_argument ("--llama", type=str, default=os.getenv ("LLAMA_MODEL_PATH", "/path/to/llama-3-8b.gguf"),
                         help="Path to the Llama model file")
    parser.add_argument ("--vector-store", type=str, default="embeddings/law_faiss_index",
                         help="Path to the FAISS vector store")
    parser.add_argument ("--mode", type=str, choices=["interactive", "query", "case", "law"], default="interactive",
                         help="Operation mode")
    parser.add_argument ("--input", type=str, help="Input text for non-interactive modes")

    args = parser.parse_args ( )

    if args.mode == "interactive" :
        interactive_mode (args.llama, args.vector_store)

    elif args.mode == "query" :
        if not args.input :
            print ("Error: --input is required for query mode")
            exit (1)
        test_query (args.input, args.llama, args.vector_store)

    elif args.mode == "case" :
        if not args.input :
            print ("Error: --input is required for case mode")
            exit (1)
        test_case_analysis (args.input, args.llama, args.vector_store)

    elif args.mode == "law" :
        if not args.input :
            print ("Error: --input is required for law mode")
            exit (1)
        test_law_explanation (args.input, args.llama, args.vector_store)