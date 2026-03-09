from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader

from src.config import DATA_DIR

def get_pdf_files(data_dir: Path = DATA_DIR) -> List[Path]:
    """
    Return all PDF files from the data directory.
    """
    return list(data_dir.glob("*.pdf"))

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a single PDF file.
    """
    reader = PdfReader(str(pdf_path))
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    return "\n".join(text_parts)

def load_documents() -> List[Dict[str, str]]:
    """
    Load all PDF documents from the data folder and return
    a list pf dictionaries with source name and extracted text.
    """
    documents = []
    pdf_files = get_pdf_files()

    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if text.strip():
            documents.append(
                {
                    "source": pdf_file.name,
                    "text": text,
                }
            )

    return documents

