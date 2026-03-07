from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def create_metadata_chunks(cleaned_pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    final_documents = []
    
    for item in cleaned_pages:
        # Extract the source name without the .pdf extension for the ID
        source_name = item["metadata"]["source"].replace(".pdf", "").lower()
        page_num = item["metadata"]["page"]
        
        # Split the text of this specific page
        page_chunks = splitter.split_text(item["text"])
        
        # Iterate with an index to create the chunk ID
        for i, chunk_text in enumerate(page_chunks):
            # Construct the unique ID: ml_notes_p10_c1, ml_notes_p10_c2, etc.
            chunk_id = f"{source_name}_p{page_num}_c{i+1}"
            
            # Create a copy of metadata and add the new ID
            chunk_metadata = item["metadata"].copy()
            chunk_metadata["chunk_id"] = chunk_id
            
            doc = Document(
                page_content=chunk_text,
                metadata=chunk_metadata
            )
            final_documents.append(doc)
            
    return final_documents