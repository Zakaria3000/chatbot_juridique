import pymupdf
import os

def extract_text_from_pdf(pdf_path):
    """ Extrait le texte brut d'un fichier PDF """
    text = ""
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

def process_all_pdfs(input_folder, output_file):
    """ Extrait et enregistre le texte de tous les PDF d'un dossier """
    all_text = ""
    for file in os.listdir(input_folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(input_folder, file)
            all_text += extract_text_from_pdf(file_path) + "\n\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(all_text)

if __name__ == "__main__":
    process_all_pdfs("data", "data/extracted_laws.txt")
    print("✅ Extraction terminée")
