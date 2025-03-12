from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Charger le texte
with open("data/cleaned_laws.txt", "r", encoding="utf-8") as f:
    text_data = f.readlines()

# Création des embeddings
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(text_data, embeddings)

# Sauvegarde du store
vectorstore.save_local("embeddings/law_faiss_index")
print("✅ Indexation terminée")
