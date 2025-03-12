import re

def clean_text(text):
    """ Nettoie le texte en supprimant les caractères spéciaux et les espaces inutiles """
    text = re.sub(r"\s+", " ", text)  # Supprime les espaces multiples
    text = re.sub(r"[\t\n\r]+", " ", text)  # Supprime les retours à la ligne
    return text.strip()

if __name__ == "__main__":
    with open("data/extracted_laws.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned_text = clean_text(raw_text)

    with open("data/cleaned_laws.txt", "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print("✅ Nettoyage terminé")
