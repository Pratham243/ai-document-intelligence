import io
from pathlib import Path

import pdfplumber
from docx import Document


def extract_text_from_pdf(file_path: str | None = None, file_bytes: bytes | None = None) -> str:
    pages = []
    if file_bytes:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
    elif file_path:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
    return "\n\n".join(pages)


def extract_text_from_docx(file_path: str | None = None, file_bytes: bytes | None = None) -> str:
    if file_bytes:
        doc = Document(io.BytesIO(file_bytes))
    elif file_path:
        doc = Document(file_path)
    else:
        return ""
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text_from_txt(file_path: str | None = None, file_bytes: bytes | None = None) -> str:
    if file_bytes:
        return file_bytes.decode("utf-8", errors="replace")
    elif file_path:
        return Path(file_path).read_text(encoding="utf-8", errors="replace")
    return ""


def extract_text(file_path: str | None = None, file_bytes: bytes | None = None, filename: str = "") -> str:
    ext = Path(filename or file_path or "").suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path=file_path, file_bytes=file_bytes)
    elif ext == ".docx":
        return extract_text_from_docx(file_path=file_path, file_bytes=file_bytes)
    elif ext == ".txt":
        return extract_text_from_txt(file_path=file_path, file_bytes=file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
