from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def ingest_data(filepath, JobDescription):
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    print(format_docs(documents))

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs
    )

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save and reload the vector store
    vectorstore.save_local("faiss_index_")
    persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)

    llm = OllamaLLM(model="llama3.2")

    template = """
    You are a resume evaluation assistant. Given the job description and context extracted from a candidate's resume, evaluate the match.
    
    Job Description:
    {JobDescription}

    Resume Content:
    {context}

    Please do the following:
    - Evaluate how well the candidate's resume matches the job description.
    - Give a match score between 0 to 100%.
    - Provide detailed feedback covering skills match, experience alignment, missing qualifications and suggestions for improvement.
    
    Answer in the following format:
    Match Score: <score>%
    Feedback:
    <detailed evaluation and suggestions>
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Create a retriever
    qa_chain = (
        {
            "context": persisted_vectorstore.as_retriever() | format_docs,
            "JobDescription": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    response = qa_chain.invoke(JobDescription)
    return response