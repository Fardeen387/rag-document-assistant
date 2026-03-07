import ftfy
import re

def clean_pages(pages_data):
    cleaned_pages = []
    for item in pages_data:
        # 1. Fix encoding
        text = ftfy.fix_text(item["text"])
        # 2. Your fix for the diamond symbols
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        # 3. Fix whitespace
        text = " ".join(text.split())
        
        cleaned_pages.append({
            "text": text,
            "metadata": item["metadata"]
        })
    return cleaned_pages