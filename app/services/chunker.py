from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_transcript(transcript: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = text_splitter.create_documents([transcript])
    return documents
