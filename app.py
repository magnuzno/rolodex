import os
import tempfile

import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores.faiss import FAISS

# Initialize Streamlit app
st.title("Document-based Q&A System")
# Streamlit interface to allow users to upload PDF files
uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type=['pdf'])
# uploaded_files = os.listdir('/home/pvcdata/bravo11bot/docs')
# Temporary directory for uploaded files
temp_dir = tempfile.mkdtemp()
# Placeholder for documents and vector store
documents = []
vector_store = None
# Process uploaded files and load/create vector store
if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_path = os.path.join(temp_dir, uploaded_file.name)
        loader = UnstructuredPDFLoader(pdf_path, mode="single")
        documents.extend(loader.load())
    if documents:
        # Split and process documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(documents)
        for doc in documents:
            doc.page_content = doc.page_content.replace("\n", "")
        # Load or create vector store
        try:
            vector_store = FAISS.load_local('faiss_index')
        except:
            model_path = '/home/pvcdata/bravo11bot/mistral/llama2_13b_chat.gguf'  # Update this path
            embedding_model = LlamaCppEmbeddings(model_path=model_path, n_ctx=7000, n_batch=100, verbose=False, n_gpu_layers=-1)
            vector_store = FAISS.from_documents(documents, embedding_model)
            vector_store.save_local('faiss_index')
        st.success("Documents processed and vector store created/loaded.")





        # Setup LLM and prompt template
        llm = LlamaCpp(model_path=model_path, temperature=0.9, max_tokens=300, n_ctx=7000, top_p=1, n_gpu_layers=-1, n_batch=100, verbose=False, repeat_penalty=1.9)
        template = """[INST]<<SYS>>You are a helpful assistant. Answer the question with the context provided. Use only information from the context and answer succintly in short sentences.<</SYS>>
History: {history}
Context: {context}
Question: {question}
Answer: [/INST]"""
        prompt = PromptTemplate(template=template, input_variables=["history", "context", "question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
        # Streamlit interface for user to input a question
        question = st.text_input("Type your question here:")
        if question:
            # Perform search and generate answer
            context = vector_store.similarity_search(question)
            context_str = [f"{i}: " + doc.page_content for i, doc in enumerate(context)]
            response = llm_chain.invoke({'history': [], 'context': context_str, 'question': question})
            st.markdown("### Answer")
            st.write(response["text"])
    else:
        st.error("No documents were processed. Please upload valid PDF files.")
else:
    st.write("Please upload one or more PDF documents.")
