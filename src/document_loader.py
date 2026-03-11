from pathlib import Path
from typing import List, Dict, Optional

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


def load_documents_from_paths(pdf_files: List[Path]) -> List[Dict[str, str]]:
    """
    Load PDF documents from a list of file paths.
    """
    documents = []

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


def load_documents(pdf_files: Optional[List[Path]] = None) -> List[Dict[str, str]]:
    """
    Load PDF documents either from provided paths or from the default data folder.
    """
    if pdf_files is None:
        pdf_files = get_pdf_files()

    return load_documents_from_paths(pdf_files)