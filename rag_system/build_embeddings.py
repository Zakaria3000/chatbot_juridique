import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import json


class ArabertEmbeddings:
    def _init_(self, model_name="aubmindlab/bert-base-arabertv2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model_name = model_name
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

    # Add this method to make the class compatible with LangChain's expectations
    def embed_documents(self, texts):
        return self.get_embeddings(texts)

    # Add this method for query embedding
    def embed_query(self, text):
        return self.get_embeddings([text])[0]


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into chunks with overlap
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def build_embeddings():
    # Load cleaned legal texts
    with open("data/cleaned_laws.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    # Split into documents and chunk them
    documents = []

    # First, try to split by document separators if they exist
    raw_docs = full_text.split("\n\n")

    for i, doc in enumerate(raw_docs):
        if len(doc.strip()) > 0:
            chunks = chunk_text(doc)

            for j, chunk in enumerate(chunks):
                chunk_id = f"doc_{i}chunk{j}"
                metadata = {
                    "id": chunk_id,
                    "source": f"doc_{i}chunk{j}",
                    "chunk": j,
                    "document": i
                }
                documents.append(Document(page_content=chunk, metadata=metadata))

    print(f"Created {len(documents)} document chunks")

    # Initialize AraBERT embeddings
    embeddings = ArabertEmbeddings()

    # Option 1: Use LangChain's expected interface
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    # Option 2: If you want to pre-compute embeddings manually
    # texts = [doc.page_content for doc in documents]
    # text_embeddings = embeddings.get_embeddings(texts)
    # embeddings_list = [emb.tolist() for emb in text_embeddings]
    # metadata_list = [doc.metadata for doc in documents]
    # vector_store = FAISS.from_embeddings(
    #    embedding_vectors=embeddings_list,
    #    texts=texts,
    #    metadatas=metadata_list,
    # )

    # Save the vector store
    vector_store.save_local("embeddings/law_faiss_index")

    print("✅ Embeddings created and saved successfully")

    # Save document metadata separately for easier reference
    metadata = [doc.metadata for doc in documents]
    with open("embeddings/document_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "_main_":
    build_embeddings()