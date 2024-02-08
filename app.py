import os
import tempfile

import streamlit as st

from bravobot import ChatBot, DocumentLoader, VectorStoreManager

model_path = '/home/pvcdata/bravo11bot/mistral/llama2_13b_Q8_chat.gguf'
faiss_path = '/home/pvcdata/bravo11bot/faiss_index'

st.title("Bravo11Bot - RAG for Acquisitions")

# Define your options
# options = ["Local Llama2", "GPT4"]

# # Create the dropdown menu
# model_option = st.selectbox("Choose an option", options)

# # Display the selected option
# st.write(f"You selected {model_option}")
# openai_key = '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'
openai_key = None

try:
    vector_store = VectorStoreManager(model_path = model_path, faiss_index_path=faiss_path, openai_key=openai_key).create_vector_store()
    st.success("Vector store found and loaded from disk (faiss_index)")
except TypeError: # documents is None, not iterable
    # Streamlit interface to allow users to upload PDF files
    uploaded_files = st.file_uploader("Upload PDF, Excel, or JSON documents", accept_multiple_files=True)

    # Temporary directory for uploaded files
    temp_dir = tempfile.mkdtemp()

    # Placeholder for documents and vector store
    documents = []
    vector_store = None

# Process uploaded files and load/create vector store
    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_path = os.path.join(temp_dir, uploaded_file.name)
        document_loader = DocumentLoader(pdf_path)
        documents.extend(document_loader.load_documents())
    if documents:
        vector_store_manager = VectorStoreManager(documents, model_path, faiss_path)
        vector_store = vector_store_manager.create_vector_store()
        st.success("Documents processed and vector store created/loaded.")

chat_bot = ChatBot(vector_store, model_path, openai_key=openai_key)

## Chat interface with streamlit
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your question here"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

if prompt:
    response, context = chat_bot.chat(prompt)

    # TODO add side window with context

    with st.chat_message("assistant"):
        st.markdown(response)
        # response = st.write_stream(response_generator()) streaming

    st.session_state.messages.append({"role":"assistant", "content": response})

#TODO add button to clear chat
