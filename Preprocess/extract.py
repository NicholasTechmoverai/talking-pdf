# Extract text from PDF

import fitz  # pymupdf

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text

text = extract_text("file.pdf")


print(text[:500])