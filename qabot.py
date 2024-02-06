import os
import sys

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores.chroma import Chroma

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

documents = []
count = 0
for file in os.listdir("docs"):
    if count > 10:
        break
    if file.endswith(".pdf"):
        pdf_path = "./docs/" + file
        loader = UnstructuredPDFLoader(pdf_path, mode="elements")
        documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

embedding_model = OllamaEmbeddings(model="dolphin-phi")
vector_store = Chroma(documents, embedding_model)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=Ollama(model="dolphin-phi"),
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
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
