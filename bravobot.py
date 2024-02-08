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
from langchain_openai import OpenAI, OpenAIEmbeddings


class DocumentLoader:
    def __init__(self, path):
        self.path = path
        self.documents = []

    def load_documents(self):
        # file_path = self.directory + '/' + file
        #TODO implement folder loading in streamlit app.py
        if self.path.endswith(".pdf"):
            loader = UnstructuredPDFLoader(self.path, mode="single")
        elif self.path.endswith('.json'):
            loader = JSONLoader(
                file_path=self.path,
                jq_schema='.[]',
                text_content=False)
        elif self.path.endswith('.csv') or self.path.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(self.path, mode="single")
        self.documents.extend(loader.load())
        return self.documents

class VectorStoreManager:
    def __init__(self, documents=None, model_path=None,  faiss_index_path = None, document_size=3000, doc_chunk_overlap=500, openai_key=None):
        self.documents = documents
        self.model_path = model_path
        self.document_size = document_size
        self.doc_chunk_overlap = doc_chunk_overlap
        self.faiss_index_path = faiss_index_path
        self.openai_key = openai_key

    def create_vector_store(self):
        if self.openai_key is not None:
            embedding_model = OpenAIEmbeddings()
        else:
            embedding_model = LlamaCppEmbeddings(model_path=self.model_path, n_ctx=7000, n_batch=100, verbose=False, n_gpu_layers=-1)
        if not os.path.isdir(self.faiss_index_path):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.document_size,
                chunk_overlap=self.doc_chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            self.documents = text_splitter.split_documents(self.documents)
            for doc in self.documents:
                doc.page_content = doc.page_content.replace("\n", "")

            vector_store = FAISS.from_documents(self.documents, embedding_model)
            vector_store.save_local(self.faiss_index_path)
        else:
            vector_store = FAISS.load_local(self.faiss_index_path,embedding_model)
        return vector_store

class ChatBot:
    def __init__(self, vector_store: FAISS, model_path: str, openai_key=None):
        self.vector_store = vector_store
        self.model_path = model_path
        if openai_key is not None:
            pass
        else:
            self.llm = LlamaCpp(model_path=model_path, temperature=0.01, max_tokens=300, n_ctx=7000, n_gpu_layers=-1, verbose=False, repeat_penalty=4, stop=["[INST]", "User:"], f16_kv=True, streaming=False)
        template = """[INST]<<SYS>>You are a helpful assistant. Answer the question with the context provided. Use only information from the context and answer succintly in short sentences.<</SYS>>
        History: {history}
        Context: {context}
        Question: {question}
        Assistant:[/INST]"""
        prompt = PromptTemplate(template=template, input_variables=["history", "context", "question"])
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm, verbose=True)
        self.chat_history = []

    def chat(self, question):
        context = self.vector_store.similarity_search(question, k=10, fetch_k=40)
        # TODO add document title in context_str
        context_str = [f"{i}: " + doc.page_content for i, doc in enumerate(context)]
        response = self.llm_chain.invoke({'history':self.chat_history, 'context': context_str, 'question': question})
        self.chat_history.append(f"User: {question} \n Assistant:{response['text']})")
        return response["text"], context_str
