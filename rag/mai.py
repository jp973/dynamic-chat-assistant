import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import tempfile
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Initialize embeddings and model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = GoogleGenerativeAI(model="gemini-1.5-flash")

# Initialize session state
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'memory' not in st.session_state:
    st.session_state.memory = []
if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None

def reset_state():
    """Reset all state variables when switching documents"""
    if st.session_state.processed_docs:  # Only show message if there was a previous document
        st.warning("🔄 Switching documents - Previous conversations and embeddings cleared!")
    st.session_state.processed_docs = None
    st.session_state.retriever = None
    st.session_state.memory = []
    st.session_state.current_pdf = None

# Set page configuration and styling
st.set_page_config(page_title="RAG Chat Assistant", layout="wide", initial_sidebar_state="expanded")

# Update the CSS with dark theme styling
st.markdown("""
    <style>
    /* Dark theme color scheme and base styles */
    :root {
        --primary-color: #8B5CF6;
        --secondary-color: #6D28D9;
        --background: #111827;
        --card-background: #1F2937;
        --text-primary: #F9FAFB;
        --text-secondary: #D1D5DB;
        --border-color: #374151;
        --hover-color: #2D3748;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--background) 0%, #0F172A 100%);
    }
    
    /* Header and title styling */
    h1 {
        background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.5rem !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background-color: var(--card-background) !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: var(--card-background) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    .stChatMessage:hover {
        background: var(--hover-color) !important;
    }
    
    /* Chat input styling */
    .stTextInput input {
        background: var(--card-background) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .stTextInput input:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2) !important;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: var(--card-background) !important;
        color: var (--text-primary) !important;
        border: 2px dashed var(--primary-color) !important;
    }
    
    /* Conversation container styling */
    .conversation-container {
        background: var(--card-background) !important;
        border-left: 4px solid var(--primary-color) !important;
        color: var(--text-primary) !important;
    }
    
    /* Welcome screen styling */
    .upload-text {
        background: var(--card-background) !important;
        color: var(--text-primary) !important;
    }
    
    /* General text colors */
    p, span, div {
        color: var(--text-primary) !important;
    }
    
    /* Spinner and progress styling */
    .stSpinner > div {
        border-color: var(--primary-color) transparent transparent !important;
    }
    
    /* Success/Error message styling */
    .stSuccess, .stInfo {
        background-color: var(--card-background) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--primary-color) !important;
    }
    
    .stError {
        background-color: var(--card-background) !important;
        color: #EF4444 !important;
        border: 1px solid #DC2626 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.title("📚 Dynamic RAG Chat Assistant")
    st.markdown("---")

with col2:
    # Add a sidebar with information
    with st.sidebar:
        st.markdown("### 📖 About")
        st.info("This application uses the Gemini model with FAISS for document question-answering.")
        st.markdown("### 🔍 How to use")
        st.markdown("""
        1. Upload a PDF document
        2. Wait for processing
        3. Ask questions about the content
        """)

# File upload with better UI
uploaded_file = st.file_uploader("📤 Upload your PDF document", type="pdf")

if uploaded_file:
    # Check if a new PDF is uploaded
    if st.session_state.current_pdf != uploaded_file.name:
        reset_state()
        st.session_state.current_pdf = uploaded_file.name
        st.info(f"📚 Now processing: {uploaded_file.name}")
        
    if st.session_state.processed_docs is None:
        try:
            with st.spinner("🔄 Processing your document... Please wait."):
                # Clean up any existing data
                if 'vectorstore' in locals() and 'vectorstore' in globals():
                    del vectorstore
                if 'docs' in locals():
                    del docs
                
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                # Process document
                loader = PyPDFLoader(temp_file_path)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(data)
                
                # Store processed documents in session state
                vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
                st.session_state.retriever = vectorstore.as_retriever(
                    search_type="similarity", 
                    search_kwargs={"k": 10}
                )
                st.session_state.processed_docs = True
                
            st.success("✅ New document processed successfully! You can now start asking questions.")
        except Exception as e:
            st.error(f"❌ An error occurred while processing the new document: {str(e)}")

# Display chat interface only if document is processed
if st.session_state.processed_docs:
    # Display conversation history
    if st.session_state.memory:
        st.markdown("<h3 style='color: var(--primary-color);'>Previous Conversations</h3>", unsafe_allow_html=True)
        for conv in st.session_state.memory:
            with st.container():
                st.markdown(f"""
                    <div class="conversation-container">
                        <p style="color: var(--primary-color); font-weight: 600;">Question:</p>
                        <p style="color: var(--text-primary);">{conv['query']}</p>
                        <p style="color: var(--primary-color); font-weight: 600; margin-top: 1rem;">Answer:</p>
                        <p style="color: var(--text-secondary);">{conv['response']}</p>
                    </div>
                """, unsafe_allow_html=True)

    query = st.chat_input("Type your question here...")

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Prepare conversation history context
                    memory_context = "\n".join([
                        f"User: {m['query']}\nAssistant: {m['response']}"
                        for m in st.session_state.memory[-5:]  # Keep last 5 conversations
                    ])

                    system_prompt = """
                    You are a highly intelligent and concise assistant. Use both the context and 
                    recent conversation history to provide accurate answers.

                    Context: {context}
                    
                    Recent Conversation History:
                    {memory_context}

                    Remember to:
                    1. Use the provided context as primary source
                    2. Consider conversation history for context
                    3. If information isn't in the document, say so and offer to explain
                    4. Keep responses clear and focused

                    Now, respond to the user's question.
                    """.format(context="{context}", memory_context=memory_context)

                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}")
                    ])

                    question_answer_chain = create_stuff_documents_chain(
                        llm,
                        prompt
                    )
                    
                    rag_chain = create_retrieval_chain(
                        st.session_state.retriever,
                        question_answer_chain
                    )

                    response = rag_chain.invoke({
                        "input": query
                    })
                    
                    if isinstance(response, dict):
                        answer = response.get('answer', '')
                    else:
                        answer = str(response)

                    if answer and answer.strip():
                        st.write(answer)
                        # Update memory with new conversation
                        st.session_state.memory.append({
                            "query": query,
                            "response": answer
                        })
                        # Keep only last 5 conversations
                        st.session_state.memory = st.session_state.memory[-5:]
                    else:
                        fallback_message = "I apologize, but I couldn't find relevant information to answer your question. Please try rephrasing your question."
                        st.write(fallback_message)
                        st.session_state.memory.append({
                            "query": query,
                            "response": fallback_message
                        })

                except Exception as e:
                    st.error(f"An error occurred during response generation: {e}")
else:
    # Center-aligned upload prompt
    st.markdown("""
        <div class="upload-text">
            <h3>👋 Welcome!</h3>
            <p>Please upload a PDF document to get started.</p>
        </div>
    """, unsafe_allow_html=True)
