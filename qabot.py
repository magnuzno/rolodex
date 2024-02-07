import os
import sys

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores.faiss import FAISS

here = os.path.dirname(os.path.abspath(__file__))

documents = []
for file in os.listdir(here + "/docs"):
    if file.endswith(".pdf"):
        pdf_path = here + '/docs/' + file
        loader = UnstructuredPDFLoader(pdf_path, mode="elements")
        documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

mistral_path = '/home/pvcdata/bravo11bot/mistral/mistral-7b-v0.1.Q5_K_S.gguf'
embedding_model = LlamaCppEmbeddings(model_path=mistral_path, n_ctx=7000, n_batch=100, verbose=False, n_gpu_layers=1000)

try:
    vector_store = FAISS.load_local('faiss_index')
except:
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local('faiss_index')

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=LlamaCpp(model_path=mistral_path, temperature=0.95, max_tokens=7000, top_p=1, n_gpu_layers=1000),
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    verbose=True)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the BravoBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa_chain.invoke(
        {"question": query, "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))
