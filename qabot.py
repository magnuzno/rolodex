import os
import sys

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    JSONLoader,
    UnstructuredExcelLoader,
    UnstructuredPDFLoader,
)
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores.faiss import FAISS

try:
    vector_store = FAISS.load_local('faiss_index')
except:
    here = os.path.dirname(os.path.abspath(__file__))
    documents = []
    for file in os.listdir(here + "/docs"):
        file_path = here + '/docs/' + file
        if file.endswith(".pdf"):
            loader = UnstructuredPDFLoader(file_path, mode="single")
        if file.endswith('.json'):
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.[].abstract', #extracts only the 'abstract' value
                text_content=False)
        if file.endswith('.csv') or file.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(file_path, mode="single")
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(documents)
    for doc in documents:
        doc.page_content = doc.page_content.replace("\n", "")

    model_path = '/home/pvcdata/bravo11bot/mistral/llama2_70b.gguf'
    embedding_model = LlamaCppEmbeddings(model_path=model_path, n_ctx=7000, n_batch=100, verbose=False, n_gpu_layers=-1)
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local('faiss_index')

llm = LlamaCpp(model_path=model_path, temperature=0.9, max_tokens=300, n_ctx=7000, top_p=1, n_gpu_layers=-1, n_batch=100, verbose=False, repeat_penalty=1.9, stop=["[INST]", "User:"])

template = """[INST]<<SYS>>You are a helpful assistant. Answer the question with the context provided. Use only information from the context and answer succintly in short sentences.<</SYS>>
History: {history}
Context: {context}
Question: {question}
Answer: [/INST]"""

prompt = PromptTemplate(template=template, input_variables=["history", "context", "question"])

llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

# question = "In Unity State, the flooding was expected to put how many people at risk of further displacement?"

#

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the BravoBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
while True:
    question = input(f"{green}Prompt: ")
    if question == "exit" or question == "quit" or question == "q" or question == "f":
        print('Exiting')
        sys.exit()
    if question == '':
        continue

    ## LLM Chain prototype (no chat, single question)
    context = vector_store.similarity_search(question, k=10, fetch_k=40)
    context_str = [f"{i}: " + doc.page_content for i, doc in enumerate(context)]
    response = llm_chain.invoke({'history':chat_history, 'context': context_str, 'question': question})
    chat_history.append(f"User: {question} \n Assistant:{response['text']})")
    print(f"{white}Answer: " + response["text"])
