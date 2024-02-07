import os
import sys

from langchain.chains import ConversationalRetrievalChain, ConversationChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores.faiss import FAISS

try:
    vector_store = FAISS.load_local('faiss_index')
except:
    here = os.path.dirname(os.path.abspath(__file__))
    documents = []
    for file in os.listdir(here + "/docs"):
        if file.endswith(".pdf"):
            pdf_path = here + '/docs/' + file
            loader = UnstructuredPDFLoader(pdf_path, mode="single")
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        # separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(documents)
    for doc in documents:
        doc.page_content = doc.page_content.replace("\n", "")

    model_path = '/home/pvcdata/bravo11bot/mistral/llama2_13b_chat.gguf'
    embedding_model = LlamaCppEmbeddings(model_path=model_path, n_ctx=7000, n_batch=100, verbose=False, n_gpu_layers=-1)
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local('faiss_index')

llm = LlamaCpp(model_path=model_path, temperature=0.9, max_tokens=300, n_ctx=7000, top_p=1, n_gpu_layers=-1, n_batch=100, verbose=False, repeat_penalty=1.9)

template = """ Answer the question with the context provided. Use only information from the context and answer succintly in short sentences.
Context: {context}
Question: {question}
Answer: """

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

history_template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:{history}
Context: {context}
Human: {question}
AI: """
conv_prompt = PromptTemplate(template=history_template, input_variables=["history","context", "question"])

conv_chain = ConversationChain(prompt=prompt, llm=llm)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    verbose=True)

# question = "In Unity State, the flooding was expected to put how many people at risk of further displacement?"

# https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-Q8_0.gguf?download=true

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
    # context = vector_store.similarity_search(question)
    # print([doc.page_content for doc in context])
    # response = llm_chain.invoke({'context': context, 'question': question}
    # print(f"{white}Answer: " + response["text"])

    ## Conversation chain with custom template
    context = vector_store.similarity_search(question)
    print([doc.page_content for doc in context])
    response = conv_chain.invoke({'history': chat_history, 'context': context, 'question': question})
    chat_history.append((question, response["text"]))
    print(f"{white}Answer: " + response["text"])

    ## QA Conversation Retrieval Protoype
    # result = qa_chain.invoke(
    #     {"question": question, "chat_history": chat_history})
    # print(f"{white}Answer: " + result["answer"])
    # chat_history.append((question, result["answer"]))
    # print(f"{white}Answer: " + result["answer"])
    # chat_history.append((question, result["answer"]))
