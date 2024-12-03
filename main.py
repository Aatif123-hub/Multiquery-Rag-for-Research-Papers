import os
import streamlit as st
from utils.parser import parsers
from utils.chunking import Chunking
from rag.embedding import Embeddings
from rag.vectorstore import VectorStore
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from rag.llm_models import LLM

def rag_pipeline(selected_file, embedding_model, vector_store, llm_model, main_question):

    if selected_file.endswith('.pdf'):
        file_text = parsers.pdf_parser([selected_file])
    elif selected_file.endswith('.docx'):
        file_text = parsers.doc_parser([selected_file])
    else:
        raise ValueError("Unsupported file type. Please select a PDF or DOCX file.")

    if not file_text.strip():
        raise ValueError("No text extracted from the selected file.")

    text_chunks = Chunking.get_chunks(file_text)
    embeddings = Embeddings.get_embeddings(embedding_model)
    vectorstore = VectorStore.vectorization(vector_store, text_chunks, embeddings)

    llm = LLM.get_llm(llm_model)
    with open('/Users/aatif/Multiquery_RAG/prompt/Researcher.txt', 'r') as file:
        prompt_template = file.read()
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    multi_query_retriever = MultiQueryRetriever.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=multi_query_retriever,
        memory=memory,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    response = conversation_chain({"question": main_question})
    return response['answer']

if __name__ == "__main__":
    st.title("Research Paper Q&A with RAG Pipeline")

    input_folder = "/Users/aatif/Multiquery_RAG/input"
    output_folder = "/Users/aatif/Multiquery_RAG/output"
    os.makedirs(output_folder, exist_ok=True)
    available_files = [f for f in os.listdir(input_folder) if f.endswith(('.pdf', '.docx'))]

    uploaded_file = st.file_uploader("Upload a PDF or DOCX file:", type=["pdf", "docx"],accept_multiple_files=True)

    if uploaded_file is not None:
        with open(os.path.join(input_folder, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        available_files.append(uploaded_file.name)

    if not available_files:
        st.write("No PDF or DOCX files found in the folder.")
    else:
        selected_file = st.selectbox("Select a file:", available_files)
        selected_file_path = os.path.join(input_folder, selected_file)
        
        available_embeddings = Embeddings.get_available_embeddings()
        embedding_model = st.selectbox("Select an embedding model:", available_embeddings)

        available_vectorstores = VectorStore.get_available_vectorstores()
        vector_store = st.selectbox("Select a vector storage type:", available_vectorstores)

        available_llms = LLM.get_available_llm()
        llm_model = st.selectbox("Select an LLM model:", available_llms)

        main_question = st.text_input("What do you need to know about this research paper:")

        if st.button("Run Pipeline"):
            try:
                combined_response = rag_pipeline(selected_file_path, embedding_model, vector_store, llm_model, main_question)
                
                output_file_path = os.path.join(output_folder, "output_summary.md")
                with open(output_file_path, "w") as output_file:
                    output_file.write(f"# Summary\n\n{combined_response}\n")
                    output_file.write("\n#\n")

                st.success(f"Summary saved to {output_file_path}")
                st.write("## Summary\n", combined_response)
            except Exception as e:
                st.error(f"Error: {e}")
