import os
import tempfile

import streamlit as st

from bravobot import ChatBot, DocumentLoader, VectorStoreManager

# Initialize Streamlit app
st.title("Document-based Q&A System")

try:
    vector_store = VectorStoreManager(faiss_index='/home/pvcdata/bravo11bot/faiss_index')
except FileNotFoundError:
    # Streamlit interface to allow users to upload PDF files
    uploaded_files = st.file_uploader("Upload PDF, Excel, or JSON documents", accept_multiple_files=True)

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
        document_loader = DocumentLoader(pdf_path)
        documents.extend(document_loader.load_documents())
    if documents:
        model_path = '/home/pvcdata/bravo11bot/mistral/llama2_13b_chat.gguf'  # Update this path
        vector_store_manager = VectorStoreManager(documents, model_path)
        vector_store = vector_store_manager.create_vector_store()
        st.success("Documents processed and vector store created/loaded.")

# Setup ChatBot
chat_bot = ChatBot(vector_store, model_path)

# Streamlit interface for user to input a question
while True:
    question = st.text_input("Type your question here:")
    if question:
        # Perform search and generate answer
        response, context = chat_bot.chat(question)
        st.markdown("### Answer")
        st.write(response)
        st.markdown("### Context")
        for context_str in context:
            st.write(context_str)

