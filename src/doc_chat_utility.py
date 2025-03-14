import os
from PyPDF2 import PdfReader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

working_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize the LLM
llm = Ollama(
    model="llama3",
    temperature=0
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()


def extract_text_from_pdf(file_path):
    """Extract text from a PDF file using PyPDF2."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def initialize_knowledge_base(file_path):
    """Initialize the knowledge base from a PDF file."""
    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(file_path)

    # Split text into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(pdf_text)

    # Convert chunks to LangChain Document objects
    documents = [Document(page_content=chunk) for chunk in text_chunks]

    # Create vector embeddings
    knowledge_base = FAISS.from_documents(documents, embeddings)

    return knowledge_base


def get_answer(knowledge_base, query, memory):
    """Get an answer from the knowledge base with conversational memory."""
    retriever = knowledge_base.as_retriever()

    # Initialize RetrievalQA chain with memory
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    # Generate the response based on the number of chunks of bina
    response = qa_chain.run(query)
    return response
