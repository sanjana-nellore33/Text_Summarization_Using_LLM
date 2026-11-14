
import re
from io import BytesIO
from pypdf import PdfReader


def extract_pdf_text(pdf_file):
    """Extract text from an uploaded PDF."""

    reader = PdfReader(BytesIO(pdf_file.getvalue()))

    extracted_text = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            extracted_text.append(page_text)

    return "\n".join(extracted_text)


def clean_text(text):
    """Clean unnecessary spaces and formatting."""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(text, chunk_size=1200, overlap=150):
    """Split lengthy documents into overlapping word chunks."""

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):

        end = start + chunk_size
        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks
