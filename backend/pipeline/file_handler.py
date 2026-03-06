"""
File upload handler — extracts text from txt, docx, pdf files.
"""

import io
from fastapi import UploadFile, HTTPException


async def extract_text_from_file(file: UploadFile) -> str:
    """
    Read an uploaded file and return its text content.
    Supports .txt, .docx, .pdf
    """
    filename = file.filename.lower() if file.filename else ""
    content = await file.read()

    if filename.endswith(".txt"):
        # Try UTF-8 first, then latin-1
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1")

    elif filename.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n".join(paragraphs)
            if not text.strip():
                raise HTTPException(status_code=400, detail="The DOCX file appears to be empty.")
            return text
        except ImportError:
            raise HTTPException(status_code=500, detail="DOCX parsing not available. python-docx missing.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse DOCX file: {str(e)}")

    elif filename.endswith(".pdf"):
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(content))
            pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            text = "\n".join(pages)
            if not text.strip():
                raise HTTPException(status_code=400, detail="Could not extract text from PDF. It may be image-based.")
            return text
        except ImportError:
            raise HTTPException(status_code=500, detail="PDF parsing not available. PyPDF2 missing.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse PDF file: {str(e)}")

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {filename}. Supported: .txt, .docx, .pdf"
        )
