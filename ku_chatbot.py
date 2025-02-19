import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    CSVLoader
)
from langchain_community.document_loaders import WebBaseLoader
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Union
import tempfile

# Set up logging
def setup_logging():
    """Configure logging with both file and console handlers."""
    if not os.path.exists('logs'):
        os.makedirs('logs')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/chatbot_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None

class DocumentProcessor:
    """Handles different document types and their processing."""
    
    SUPPORTED_FORMATS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader,
        '.doc': UnstructuredWordDocumentLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.ppt': UnstructuredPowerPointLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.xls': UnstructuredExcelLoader,
        '.csv': CSVLoader,
        '.html': UnstructuredHTMLLoader
    }

    @staticmethod
    def get_loader_class(file_extension: str):
        """Get the appropriate loader class for a file extension."""
        return DocumentProcessor.SUPPORTED_FORMATS.get(file_extension.lower())

    @staticmethod
    def process_file(file_path: str) -> List[str]:
        """Process a file and return its content as text."""
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DocumentProcessor.get_loader_class(file_extension)
        
        if not loader_class:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        loader = loader_class(file_path)
        documents = loader.load()
        
        # Extract text content from documents
        if isinstance(documents, list):
            return [doc.page_content for doc in documents]
        return [documents.page_content]

class RAGChatbot:
    def __init__(self):
        """Initialize the RAG chatbot with HuggingFace embeddings."""
        logger.info("Initializing RAG Chatbot")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Successfully initialized HuggingFace embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        logger.info("Text splitter initialized")

    def process_uploaded_file(self, file_path: str) -> List[str]:
        """Process uploaded file into text chunks."""
        logger.info(f"Processing uploaded file: {file_path}")
        try:
            # Get document content using DocumentProcessor
            document_texts = DocumentProcessor.process_file(file_path)
            combined_text = "\n\n".join(document_texts)
            
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(combined_text)
            logger.info(f"Successfully split document into {len(text_chunks)} chunks")
            return text_chunks

        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            raise

    def process_urls(self, urls: List[str]) -> List[str]:
        """Process URLs using WebBaseLoader."""
        logger.info(f"Processing {len(urls)} URLs")
        try:
            loader = WebBaseLoader(urls)
            documents = loader.load()
            combined_text = "\n\n".join([doc.page_content for doc in documents])
            text_chunks = self.text_splitter.split_text(combined_text)
            logger.info(f"Successfully processed URLs into {len(text_chunks)} chunks")
            return text_chunks
        except Exception as e:
            logger.error(f"Error processing URLs: {str(e)}")
            raise

    def combine_knowledge_sources(self, text_chunks_list: List[List[str]]) -> List[str]:
        """Combine multiple sources of text chunks into a single list."""
        combined_chunks = []
        for chunks in text_chunks_list:
            combined_chunks.extend(chunks)
        return combined_chunks

    def initialize_vector_store(self, text_chunks: List[str]) -> FAISS:
        """Initialize FAISS vector store with text chunks."""
        logger.info("Initializing vector store")
        try:
            vector_store = FAISS.from_texts(text_chunks, self.embeddings)
            logger.info("Successfully created vector store")
            return vector_store
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def setup_conversation_chain(self, vector_store: FAISS) -> ConversationalRetrievalChain:
        """Set up the conversational chain with the vector store."""
        logger.info("Setting up conversation chain")
        try:
            memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True
            )

            chain = ConversationalRetrievalChain.from_llm(
                llm = ChatGroq(
                    groq_api_key="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",
                    model_name="mixtral-8x7b-32768",
                    temperature=0.7,
                ),
                retriever=vector_store.as_retriever(),
                memory=memory
            )
            logger.info("Successfully created conversation chain")
            return chain
        except Exception as e:
            logger.error(f"Error setting up conversation chain: {str(e)}")
            raise

def main():
    logger.info("Starting application")
    st.set_page_config(page_title="Enhanced RAG Chatbot", page_icon="🤖")
    st.title("Enhanced RAG Chatbot with Multi-Document Support")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Document upload section
        st.subheader("Document Upload")
        supported_formats = list(DocumentProcessor.SUPPORTED_FORMATS.keys())
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=[fmt.replace('.', '') for fmt in supported_formats],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.info(f"Uploaded {len(uploaded_files)} file(s)")
            
        # Web scraping section
        st.subheader("Web Content")
        urls_input = st.text_area("Enter URLs (one per line)")
        process_urls = st.button("Process URLs")

        # Initialize knowledge base
        st.subheader("Knowledge Base")
        initialize_kb = st.button("Initialize Knowledge Base")

        if initialize_kb:
            logger.info("Initializing knowledge base")
            with st.spinner("Initializing knowledge base..."):
                try:
                    chatbot = RAGChatbot()
                    all_text_chunks = []
                    
                    # Process uploaded documents
                    if uploaded_files:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            for uploaded_file in uploaded_files:
                                file_path = os.path.join(temp_dir, uploaded_file.name)
                                
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                
                                try:
                                    doc_chunks = chatbot.process_uploaded_file(file_path)
                                    all_text_chunks.append(doc_chunks)
                                    logger.info(f"Successfully processed {uploaded_file.name}")
                                except Exception as e:
                                    logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
                                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    
                    # Process URLs if provided
                    if urls_input:
                        urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
                        if urls:
                            try:
                                url_chunks = chatbot.process_urls(urls)
                                all_text_chunks.append(url_chunks)
                                logger.info("Successfully processed URLs")
                            except Exception as e:
                                logger.error(f"Error processing URLs: {str(e)}")
                                st.error(f"Error processing URLs: {str(e)}")

                    if not all_text_chunks:
                        logger.warning("No content found to process")
                        st.error("Please either upload documents or provide URLs first!")
                        return

                    # Combine all text chunks and initialize vector store
                    combined_chunks = chatbot.combine_knowledge_sources(all_text_chunks)
                    st.session_state.vector_store = chatbot.initialize_vector_store(combined_chunks)
                    st.session_state.conversation_chain = chatbot.setup_conversation_chain(st.session_state.vector_store)
                    st.success("Knowledge base initialized!")
                    logger.info("Knowledge base initialization completed")
                except Exception as e:
                    logger.error(f"Error initializing knowledge base: {str(e)}")
                    st.error(f"Error initializing knowledge base: {str(e)}")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the content"):
        logger.info(f"Received user prompt: {prompt}")
        if not st.session_state.conversation_chain:
            logger.warning("Attempted to chat without initialized knowledge base")
            st.error("Please initialize the knowledge base first!")
            return

        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    logger.info("Generating response")
                    response = st.session_state.conversation_chain({"question": prompt})
                    response_text = response['answer']
                    st.write(response_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    logger.info("Successfully generated and displayed response")
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}")
                    st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()