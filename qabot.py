import os
import sys

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import LlamaCppEmbeddings

# from langchain_community.llms.ollama import Ollama
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores.faiss import FAISS

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["LANGCHAIN_HANDLER"] = ""

here = os.path.dirname(os.path.abspath(__file__))

documents = []
count = 0
for file in os.listdir(here + "/docs"):
    if count > 10:
        break
    if file.endswith(".pdf"):
        pdf_path = here + '/docs/' + file
        loader = UnstructuredPDFLoader(pdf_path, mode="elements")
        documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

mistral_path = '/home/pvcdata/bravo11bot/mistral/mistral-7b-v0.1.Q5_K_S.gguf'
# embedding_model = OllamaEmbeddings(model="dolphin-phi")
embedding_model = LlamaCppEmbeddings(model_path=mistral_path, n_ctx=8000, n_batch=10)
vector_store = FAISS.from_documents(documents, embedding_model)


qa_chain = ConversationalRetrievalChain.from_llm(
    llm=LlamaCpp(model_path=mistral_path, temperature=0.75, max_tokens=2000, top_p=1),
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

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
