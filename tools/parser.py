from PyPDF2 import PdfReader
import docx2txt 

class parsers:

    def pdf_parser(pdf_file):

        text=""

        for pdf in pdf_file:
            try:
                pdf_reader = PdfReader(pdf)
            except Exception as e:
                raise Exception(f"Failed to read the PDF file: {pdf}.Error: {e}")
            
            try:
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                    else:
                        raise Exception(f"Failed to extract text from a page in {pdf}.")
            except Exception as e:
                raise Exception(f"Error occurred while extracting text from {pdf}. Error: {e}")
            
        return text

    def doc_parser(doc_file):
        
        text = ""

        for doc in doc_file:
            try:
                text += docx2txt.process(doc)
            except Exception as e:
                raise Exception(f"Error occured while extracting text from {doc}. Error: {e} ")
     
        return text