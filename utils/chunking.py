
from langchain.text_splitter import CharacterTextSplitter

class Chunking:

    def get_chunks(text,chunk_size,chunk_overlap):
        try:
            text_splitter = CharacterTextSplitter(
                separator="\n", 
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
        except Exception as e:
            raise Exception(f"Error occurred while chunking text. Error: {e}")

        return chunks