"""
Document Loaders for Automotive Service Manuals, TSBs & Recalls (Topic 3.19)
"""
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
import docx


class DocumentLoader:
    @staticmethod
    def load_pdf(file_path: Path) -> List[Dict[str, Any]]:
        pages_data = []
        doc = fitz.open(str(file_path))
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                pages_data.append({
                    "page_number": page_num + 1,
                    "raw_text": text
                })
        doc.close()
        return pages_data

    @staticmethod
    def load_docx(file_path: Path) -> List[Dict[str, Any]]:
        doc = docx.Document(str(file_path))
        full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return [{"page_number": 1, "raw_text": full_text}]

    @staticmethod
    def load_txt(file_path: Path) -> List[Dict[str, Any]]:
        text = file_path.read_text(encoding="utf-8")
        return [{"page_number": 1, "raw_text": text}]

    def load_file(self, file_path: str | Path) -> List[Dict[str, Any]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self.load_pdf(path)
        elif suffix == ".docx":
            return self.load_docx(path)
        elif suffix == ".txt":
            return self.load_txt(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")