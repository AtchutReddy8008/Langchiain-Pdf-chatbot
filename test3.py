import streamlit as st
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")


# Main Streamlit interface
st.title("PDF Question Answering")
st.write("Upload a PDF document and ask questions about its content.")

# Upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:

    # Load the PDF document
    try:
        loader = PyPDFLoader(file_path=uploaded_file)
        data = loader.load()
        st.write(f"Loaded {len(data)} document chunks from the uploaded PDF.")
    except Exception as e:
        st.error(f"Error loading PDF: {e}")

    # Split the document into text chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(data)
    st.write(f"Split the document into {len(text_chunks)} chunks.")
    st.write(f"Split the document into {(text_chunks)} chunks.")
    
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={"device": device}
)
st.write("Embeddings model loaded successfully.")

# Initialize the Llama model for answer generation
llm_answer_gen = LlamaCpp(
    streaming=False,
    model_path=r"./mistral-7b-openorca.Q4_0.gguf",  # Ensure this path is correct
    temperature=0.7,
    top_p=0.95,
    f16_kv=True,
    verbose=True,
    n_ctx=4096
)
st.write("LLM model initialized.")

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


    # Create vector store from the document chunks and persist locally
try:
    vector_store = Chroma.from_documents(
        text_chunks, 
        embeddings,
        persist_directory="./chroma_db"  # This will create the directory if it doesn't exist
    )
    st.write("Vector store created for document retrieval.")
    
    # Initialize answer generation chain
    answer_gen_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_answer_gen,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    
    # User input for questions
    user_input = st.text_input("Ask a question about the PDF content:")

    if user_input:
        st.write("Preparing to invoke answer generation...")
        try:
            # Run the conversational retrieval chain using invoke
            result = answer_gen_chain.run({"question": user_input})
            st.write("Result structure:", result)  # Print the entire result to see its structure
            # Access the answer if the key is present
            if "answer" in result:
                st.write(result["answer"])
            else:
                st.write("No answer key found in the result.")
        except Exception as e:
            st.error(f"An error occurred while retrieving the answer: {e}")
except Exception as e:
    st.error(f"Error creating vector store: {e}")

