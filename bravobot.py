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


class DocumentLoader:
    def __init__(self, directory):
        self.directory = directory
        self.documents = []

    def load_documents(self):
        for file in os.listdir(self.directory):
            file_path = self.directory + '/' + file
            if file.endswith(".pdf"):
                loader = UnstructuredPDFLoader(file_path, mode="single")
            elif file.endswith('.json'):
                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema='.[]',
                    text_content=False)
            elif file.endswith('.csv') or file.endswith('.xlsx'):
                loader = UnstructuredExcelLoader(file_path, mode="single")
            self.documents.extend(loader.load())
        return self.documents

class VectorStoreManager:
    def __init__(self, documents, model_path, document_size=3000, doc_chunk_overlap=500):
        self.documents = documents
        self.model_path = model_path
        self.document_size = document_size
        self.doc_chunk_overlap = doc_chunk_overlap

    def create_vector_store(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.document_size,
            chunk_overlap=self.doc_chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self.documents = text_splitter.split_documents(self.documents)
        for doc in self.documents:
            doc.page_content = doc.page_content.replace("\n", "")

        embedding_model = LlamaCppEmbeddings(model_path=self.model_path, n_ctx=7000, n_batch=100, verbose=False, n_gpu_layers=-1)
        vector_store = FAISS.from_documents(self.documents, embedding_model)
        vector_store.save_local('faiss_index')
        return vector_store

class ChatBot:
    def __init__(self, vector_store, model_path):
        self.vector_store = vector_store
        self.model_path = model_path
        self.llm = LlamaCpp(model_path=model_path, temperature=0.9, max_tokens=300, n_ctx=7000, top_p=1, n_gpu_layers=-1, n_batch=100, verbose=False, repeat_penalty=1.9, stop=["[INST]", "User:"], f16_kv=True)
        template = """[INST]<<SYS>>You are a helpful assistant. Answer the question with the context provided. Use only information from the context and answer succintly in short sentences.<</SYS>>
        History: {history}
        Context: {context}
        Question: {question}
        Answer: [/INST]"""
        prompt = PromptTemplate(template=template, input_variables=["history", "context", "question"])
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm, verbose=True)
        self.chat_history = []

    def chat(self):
        while True:
            question = input("Prompt: ")
            if question in ["exit", "quit", "q", "f"]:
                print('Exiting')
                sys.exit()
            if question == '':
                continue

            context = self.vector_store.similarity_search(question, k=10, fetch_k=40)
            context_str = [f"{i}: " + doc.page_content for i, doc in enumerate(context)]
            response = self.llm_chain.invoke({'history':self.chat_history, 'context': context_str, 'question': question})
            self.chat_history.append(f"User: {question} \n Assistant:{response['text']})")
            print("Answer: " + response["text"])
