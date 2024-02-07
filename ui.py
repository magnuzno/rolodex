from langchain_community.vectorstores.chroma import Chroma
import tempfile
# Set up the environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Title for the Streamlit app
st.title("ChatGPT-like Clone with Document Interaction")
def process_uploaded_files(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".pdf"):
            # Use a temporary file to load the PDF content
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                loader = UnstructuredPDFLoader(tmp_file.name, mode="elements")
                documents.extend(loader.load())
    return documents
def setup_chain(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    documents = text_splitter.split_documents(documents)
    embedding_model = OllamaEmbeddings(model="dolphin-phi")
    vector_store = Chroma(documents, embedding_model)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=Ollama(model="dolphin-phi"),
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )
    return qa_chain
# User uploads PDF files
uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type='pdf')
if st.button("Load Documents"):
    if uploaded_files:
        documents = process_uploaded_files(uploaded_files)
        qa_chain = setup_chain(documents)
        st.success("Documents loaded successfully. You can now ask questions.")
    else:
        st.error("Please upload at least one PDF file.")
# Placeholder for chat history
chat_history = []
# User input for asking questions
user_query = st.text_input("Ask a question about the documents:")
if st.button("Ask"):
    if user_query and 'qa_chain' in locals():
        result = qa_chain.invoke(
            {"question": user_query, "chat_history": chat_history}
        )
        st.text_area("Answer:", value=result["answer"], height=100)
        chat_history.append((user_query, result["answer"]))
    else:
        st.error("Please load documents before asking questions or ensure your question is not empty.")
