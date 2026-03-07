from pypdf import PdfReader

def load_pdf_with_metadata(file_path):
    reader = PdfReader(file_path)
    pages_data = []
    
    for index, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # We store the text AND the page number together
            pages_data.append({
                "text": text,
                "metadata": {
                    "source": file_path.split("/")[-1], # e.g., "ml_notes.pdf"
                    "page": index + 1
                }
            })
    return pages_data