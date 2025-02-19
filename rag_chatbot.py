# app.py
import streamlit as st
import os
from typing import List
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import logging
from datetime import datetime

# Set up logging
if not os.path.exists('logs'):
    os.makedirs('logs')

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'logs/chatbot_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(page_title="RAG Chatbot", layout="wide")

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def process_file(self, file) -> List:
        logger.info(f"Processing file: {file.name}")
        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Choose loader based on file type
            if file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            else:  # Assume text file
                loader = TextLoader(tmp_path)
            
            # Load and split the document
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            
            # Add source metadata
            for chunk in chunks:
                chunk.metadata['source'] = file.name
                
            logger.info(f"Successfully processed {len(chunks)} chunks from {file.name}")
            return chunks
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

def initialize_rag_chain():
    logger.info("Initializing RAG chain")
    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key="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",
        model_name="mixtral-8x7b-32768",
        temperature=0.7,
    )
    
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    logger.info("Successfully initialized RAG components")
    return llm, embeddings

def create_chatbot(llm, embeddings, documents):
    logger.info("Creating chatbot with vector store")
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Initialize conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        verbose=True,
        output_key="answer"  # Explicitly set output key for memory
    )
    
    logger.info("Successfully created chatbot chain")
    return chain

def main():
    st.title("📚 RAG Chatbot")
    
    # Initialize session state
    if 'chain' not in st.session_state:
        st.session_state.chain = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=['pdf', 'txt']
        )
        
        if uploaded_files:
            with st.spinner('Processing documents...'):
                try:
                    # Initialize document processor
                    processor = DocumentProcessor()
                    
                    # Process all documents
                    all_chunks = []
                    for file in uploaded_files:
                        chunks = processor.process_file(file)
                        all_chunks.extend(chunks)
                    
                    # Initialize RAG components
                    llm, embeddings = initialize_rag_chain()
                    
                    # Create chatbot chain
                    st.session_state.chain = create_chatbot(llm, embeddings, all_chunks)
                    
                    st.success(f'Processed {len(uploaded_files)} documents')
                    logger.info(f"Successfully processed {len(uploaded_files)} documents")
                except Exception as e:
                    error_msg = f"Error processing documents: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)
    
    # Main chat interface
    if st.session_state.chain is None:
        st.info("Please upload documents to start chatting!")
    else:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents"):
            logger.info(f"Received user prompt: {prompt}")
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Get response from chain
                        response = st.session_state.chain.invoke({"question": prompt})
                        response_text = response["answer"]
                        
                        # Add source information
                        sources = []
                        for doc in response["source_documents"]:
                            source = doc.metadata.get("source", "Unknown")
                            if source not in sources:
                                sources.append(source)
                        
                        if sources:
                            response_text += f"\n\nSources: {', '.join(sources)}"
                        
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        logger.info("Successfully generated response")
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        logger.error(error_msg)
                        st.error(error_msg)

if __name__ == "__main__":
    main()