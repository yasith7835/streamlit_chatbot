import os
import streamlit as st
from doc_chat_utility import initialize_knowledge_base, get_answer
from langchain.memory import ConversationBufferMemory

working_dir = os.path.dirname(os.path.abspath(__file__))

# Configure Streamlit page
st.set_page_config(
    page_title="Chat with Document :)",
    page_icon="📄",
    layout="centered"
)

st.title("Document Q&A - Llama 3 - Ollama")

# Initialize session state for knowledge base, memory, and chat history
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File uploader
uploaded_file = st.file_uploader(label="Upload your file", type=["pdf"])

if uploaded_file and st.button("Process Document"):
    # Save uploaded file to working directory
    bytes_data = uploaded_file.read()
    file_name = uploaded_file.name
    file_path = os.path.join(working_dir, file_name)
    with open(file_path, "wb") as f:
        f.write(bytes_data)

    # Initialize knowledge base
    st.session_state.knowledge_base = initialize_knowledge_base(file_path)
    st.session_state.chat_history = []  # Clear chat history for a new document
    st.success("Document processed successfully! You can now ask questions.")

# Chat interface
st.write("### Chat History")
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["user"])
    with st.chat_message("assistant"):
        st.write(chat["bot"])

# User input for new query
if user_query := st.chat_input("Ask your question"):
    if st.session_state.knowledge_base is None:
        st.error("Please upload and process a document first.")
    else:
        # Get the answer
        answer = get_answer(
            st.session_state.knowledge_base,
            user_query,
            st.session_state.memory
        )

        # Update chat history
        st.session_state.chat_history.append({"user": user_query, "bot": answer})

        # Display the latest question and response immediately
        with st.chat_message("user"):
            st.write(user_query)
        with st.chat_message("assistant"):
            st.write(answer)
