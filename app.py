import gradio as gr
import os
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_pinecone import PineconeVectorStore
import uuid
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_RtTvf_FprhEhvmE6aJGckZg9P14Rny1q19p1QnGuRt67eCDWLMzJe3yW1LqaJ8rs1RcyE")
PINECONE_INDEX_NAME = "pdfrag"  # Name of the Pinecone index

# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define the embedding model for similarity search
embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'

# Initialize HuggingFace Embeddings model
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"  # or another supported model
)

# Global chat history storage
chat_sessions = {}
current_session_id = None

def load_vector_store():
    """Load the Pinecone vector store."""
    print(f"Looking for Pinecone index '{PINECONE_INDEX_NAME}'")
    
    try:
        # Get the Pinecone index
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Create LangChain vector store from Pinecone index
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings
        )
        
        print(f"Vector store '{PINECONE_INDEX_NAME}' loaded successfully.")
        return vector_store
    except Exception as e:
        print(f"Error loading Pinecone index: {e}")
        return None


def create_qa_chain(vector_store):
    """Create a question-answering chain."""
    try:
        # Create retriever from vector store
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Retrieve top 5 most similar chunks
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        return None


def query_pdfs(query):
    """Query the Pinecone vector database."""
    if not query.strip():
        return "Please enter a question.", []
    
    vector_store = load_vector_store()  # Load the pre-created vector store
    
    if vector_store is None:
        return "Error: Vector store is not available.", []
    
    # Create QA chain with vector store
    qa_chain = create_qa_chain(vector_store)
    
    if qa_chain is None:
        return "Error: Could not create QA chain.", []
    
    try:
        # Perform the query and retrieve results
        response = qa_chain.invoke({"query": query})
        sources = []

        # Prepare sources in a structured format to be displayed in the Gradio interface
        if 'source_documents' in response:
            for doc in response['source_documents']:
                # Try multiple ways to get page number
                page_number = None
                
                # Check different possible metadata keys
                if hasattr(doc, 'metadata') and doc.metadata:
                    page_number = (doc.metadata.get('page') or 
                                 doc.metadata.get('page_number') or 
                                 doc.metadata.get('source_page') or 
                                 doc.metadata.get('page_label'))
                
                # If still no page number, try to extract from source if available
                if not page_number and hasattr(doc, 'metadata') and doc.metadata.get('source'):
                    source_info = doc.metadata.get('source', '')
                    # Try to extract page number from source string
                    import re
                    page_match = re.search(r'page[_\s]*(\d+)', source_info.lower())
                    if page_match:
                        page_number = page_match.group(1)
                
                # Default to document index if no page found
                if not page_number:
                    page_number = f"Doc {len(sources) + 1}"
                
                snippet = doc.page_content[:300]  # Only show the first 300 characters of the chunk
                sources.append([str(page_number), snippet])

        # If no sources are found, display a "No sources found" message
        if not sources:
            sources.append(["No sources found", ""])

        return response['result'], sources
        
    except Exception as e:
        print(f"Error during query: {e}")
        return f"Error processing your query: {str(e)}", []


def create_new_chat():
    """Create a new chat session."""
    global current_session_id
    session_id = str(uuid.uuid4())[:8]
    chat_sessions[session_id] = {
        'title': 'New Chat',
        'messages': [],
        'created_at': datetime.now().strftime("%H:%M")
    }
    current_session_id = session_id
    return update_chat_history(), [], ""


def update_chat_history():
    """Update the chat history sidebar."""
    if not chat_sessions:
        return gr.update(choices=[], value=None)
    
    choices = []
    for session_id, session_data in reversed(list(chat_sessions.items())):
        title = session_data['title']
        time = session_data['created_at']
        display_text = f"{title} • {time}"
        choices.append((display_text, session_id))
    
    return gr.update(choices=choices, value=current_session_id if current_session_id else None)


def select_chat(session_id):
    """Select a chat from history."""
    global current_session_id
    if session_id and session_id in chat_sessions:
        current_session_id = session_id
        return chat_sessions[session_id]['messages'], ""
    return [], ""


def chat_response(message, history):
    """Process chat message and return response with sources."""
    global current_session_id
    
    if not message.strip():
        return history, "", update_chat_history()
    
    # Create new session if none exists
    if not current_session_id:
        create_new_chat()
    
    # Get answer and sources from existing query_pdfs function
    answer, sources = query_pdfs(message)
    
    # Format sources for display in chat
    sources_text = ""
    if sources and sources[0][0] != "No sources found":
        sources_text = "\n\n📚 **Sources:**\n"
        for page, snippet in sources:
            # Handle different page number formats
            if page == "N/A" or page == "No sources found":
                sources_text += f"• **Document:** {snippet[:150]}...\n"
            else:
                sources_text += f"• **Page {page}:** {snippet[:150]}...\n"
    
    # Combine answer with sources
    full_response = answer + sources_text
    
    # Add to chat history
    history.append([message, full_response])
    
    # Update session data
    if current_session_id in chat_sessions:
        chat_sessions[current_session_id]['messages'] = history
        # Update title based on first message
        if len(history) == 1:
            chat_sessions[current_session_id]['title'] = message[:30] + ("..." if len(message) > 30 else "")
    
    return history, "", update_chat_history()


def create_gradio_interface():
    """Create a Claude AI-style interface with sidebar and main chat area."""
    
    # Custom CSS for Claude AI-style interface
    css = """
/* Main container */
.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    background: #0f0f0f !important;
}

/* Sidebar styling */
.sidebar {
    background: #1a1a1a;
    border-right: 1px solid #2a2a2a;
    height: 100vh;
    padding: 16px;
    min-width: 280px;
    max-width: 320px;
}

.sidebar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid #2a2a2a;
}

.logo {
    font-size: 18px;
    font-weight: 600;
    color: #f5f5f5;
    display: flex;
    align-items: center;
    gap: 8px;
}

.new-chat-btn {
    background: #ff6b35 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 18px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 2px 8px rgba(255, 107, 53, 0.3) !important;
}

.new-chat-btn:hover {
    background: #e55a2b !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(255, 107, 53, 0.4) !important;
}

.chat-history {
    margin-top: 16px;
}

.chat-history label {
    font-size: 14px;
    font-weight: 500;
    color: #a0a0a0;
    margin-bottom: 8px;
}

/* Main chat area */
.main-chat {
    background: #0f0f0f;
    height: 100vh;
    display: flex;
    flex-direction: column;
}

.chat-header {
    background: #161616;
    border-bottom: 1px solid #2a2a2a;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    backdrop-filter: blur(10px);
}

.chat-title {
    font-size: 16px;
    font-weight: 600;
    color: #f5f5f5;
    letter-spacing: -0.01em;
}

/* Chat messages */
.chat-container {
    flex: 1;
    background: #0f0f0f;
    border: none;
    border-radius: 0;
    overflow-y: auto;
}

.message-user {
    background: linear-gradient(135deg, #2a2a2a 0%, #1f1f1f 100%);
    margin: 12px 20px;
    padding: 16px 20px;
    border-radius: 20px 20px 6px 20px;
    max-width: 80%;
    margin-left: auto;
    font-size: 15px;
    line-height: 1.6;
    color: #f5f5f5;
    border: 1px solid #333;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
}

.message-assistant {
    background: linear-gradient(135deg, #1a1a1a 0%, #141414 100%);
    margin: 12px 20px;
    padding: 16px 20px;
    border-radius: 20px 20px 20px 6px;
    max-width: 85%;
    font-size: 15px;
    line-height: 1.7;
    color: #e5e5e5;
    border: 1px solid #2a2a2a;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
}

/* Input area */
.input-area {
    background: #161616;
    border-top: 1px solid #2a2a2a;
    padding: 20px 24px;
    backdrop-filter: blur(10px);
}

.chat-input {
    border: 1px solid #3a3a3a;
    border-radius: 16px;
    padding: 16px 20px;
    font-size: 15px;
    resize: none;
    background: #1a1a1a;
    color: #f5f5f5;
    width: 100%;
    min-height: 52px;
    max-height: 200px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
}

.chat-input::placeholder {
    color: #666;
}

.chat-input:focus {
    outline: none;
    border-color: #ff6b35;
    box-shadow: 0 0 0 3px rgba(255, 107, 53, 0.2), inset 0 2px 4px rgba(0, 0, 0, 0.3);
    background: #1f1f1f;
}

.send-btn {
    background: linear-gradient(135deg, #ff6b35 0%, #e55a2b 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 20px !important;
    margin-left: 12px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 2px 12px rgba(255, 107, 53, 0.3) !important;
}

.send-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 20px rgba(255, 107, 53, 0.4) !important;
}

.send-btn:disabled {
    background: linear-gradient(135deg, #404040 0%, #333 100%) !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .sidebar {
        display: none;
    }
    
    .message-user, .message-assistant {
        margin: 8px 12px;
        padding: 12px 16px;
    }
    
    .input-area {
        padding: 16px;
    }
}

/* Hide default gradio elements */
.wrap {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

/* Radio button styling for chat history */
.chat-history .gradio-radio {
    background: transparent;
    border: none;
}

.chat-history .gradio-radio > label {
    background: linear-gradient(135deg, #1f1f1f 0%, #1a1a1a 100%);
    border: 1px solid #333;
    border-radius: 12px;
    padding: 14px 16px;
    margin: 6px 0;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-size: 13px;
    line-height: 1.4;
    color: #d0d0d0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.chat-history .gradio-radio > label:hover {
    background: linear-gradient(135deg, #2a2a2a 0%, #252525 100%);
    border-color: #444;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.chat-history .gradio-radio > label[data-selected="true"] {
    background: linear-gradient(135deg, rgba(255, 107, 53, 0.2) 0%, rgba(255, 107, 53, 0.1) 100%);
    border-color: #ff6b35;
    color: #ff8c66;
    box-shadow: 0 2px 12px rgba(255, 107, 53, 0.3);
}

/* Scrollbar styling for dark theme */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: #1a1a1a;
}

::-webkit-scrollbar-thumb {
    background: #404040;
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: #505050;
}

/* Additional modern touches */
body {
    background: #0f0f0f !important;
    color: #f5f5f5 !important;
}

/* Smooth animations for all interactive elements */
* {
    transition: background-color 0.3s cubic-bezier(0.4, 0, 0.2, 1),
                border-color 0.3s cubic-bezier(0.4, 0, 0.2, 1),
                transform 0.3s cubic-bezier(0.4, 0, 0.2, 1),
                box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
"""
    
    with gr.Blocks(css=css, title="Daffo-Pilot : Agentic-RAG by Ashik", theme=gr.themes.Base()) as demo:
        
        with gr.Row(elem_classes=["main-layout"]):
            # Sidebar
            with gr.Column(scale=1, elem_classes=["sidebar"]):
                # Sidebar header
                with gr.Row(elem_classes=["sidebar-header"]):
                    gr.HTML('<div class="logo"> 🗐 History </div>')
                    new_chat_btn = gr.Button("+ New Chat", elem_classes=["new-chat-btn"], size="sm")
                
                # Chat history
                with gr.Column(elem_classes=["chat-history"]):
                    gr.Markdown("### Recent Chats")
                    chat_history_list = gr.Radio(
                        label="",
                        choices=[],
                        value=None,
                        interactive=True,
                        show_label=False
                    )
            
            # Main chat area
            with gr.Column(scale=3, elem_classes=["main-chat"]):
                # Chat header
                with gr.Row(elem_classes=["chat-header"]):
                    gr.HTML('<div class="chat-title"> 🛠     Daffo-Pilot : Agentic-RAG chat </div>')
                
                # Chat interface
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    height=600,
                    show_label=False,
                    container=True,
                    elem_classes=["chat-container"],
                    show_copy_button=True,
                    show_share_button=False
                )
                
                # Input area
                with gr.Row(elem_classes=["input-area"]):
                    with gr.Column(scale=10):
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="Ask me anything.....",
                            container=False,
                            elem_classes=["chat-input"],
                            lines=1,
                            max_lines=6
                        )
                    with gr.Column(scale=1, min_width=80):
                        submit_btn = gr.Button(
                            "Send",
                            elem_classes=["send-btn"],
                            size="sm"
                        )
        
        # Event handlers
        def respond(message, history):
            return chat_response(message, history)
        
        def handle_new_chat():
            return create_new_chat()
        
        def handle_chat_select(selected_session):
            if selected_session:
                return select_chat(selected_session)
            return [], ""
        
        # Connect events
        msg.submit(respond, [msg, chatbot], [chatbot, msg, chat_history_list])
        submit_btn.click(respond, [msg, chatbot], [chatbot, msg, chat_history_list])
        new_chat_btn.click(handle_new_chat, [], [chat_history_list, chatbot, msg])
        chat_history_list.change(handle_chat_select, [chat_history_list], [chatbot, msg])
        
        # Initialize with first chat
        demo.load(handle_new_chat, [], [chat_history_list, chatbot, msg])
    
    return demo


def main():
    """Create and launch Gradio interface."""
    # Check if required API keys are set
    if GROQ_API_KEY == "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":
        print("Warning: Please set your Groq API key in the GROQ_API_KEY variable")
    
    print("Starting PDF Question Answering System...")
    demo = create_gradio_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=10000)


if __name__ == "__main__":
    main()
