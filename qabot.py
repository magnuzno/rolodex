from langchain.document_loaders import DocumentLoader
from langchain.document_transformers import TextSplitter
from langchain.vector_stores import VectorStore
from langchain.text_embedding_models import TextEmbeddingModel
from langchain.retrievers import Retriever
from langchain.chat_models import ChatModel
from langchain import hub

loader = DocumentLoader(...)
documents = loader.load_documents()

splitter = TextSplitter(...)
split_documents = splitter.split(documents)


embedding_model = TextEmbeddingModel(...)
vector_store = VectorStore(...)

for doc in split_documents:
    embedding = embedding_model.embed(doc)
    vector_store.store(doc.id, embedding)

retriever = Retriever(vector_store, embedding_model)

llm = ChatModel(model_name="gpt-3.5-turbo", temperature=0)
prompt = hub.pull("rlm/rag-prompt")

def rag_chain(question):
    context = retriever.retrieve(question)
    prompt_input = {"context": context, "question": question}
    prompt_output = prompt.invoke(prompt_input).to_string()
    answer = llm(prompt_output)
    return answer

print(rag_chain("Your question here"))