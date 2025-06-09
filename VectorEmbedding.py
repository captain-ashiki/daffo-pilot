import os
import pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# Folder where PDFs are stored
DOCUMENTS_FOLDER = 'documents'  # Ensure you have your PDFs in this folder


# Pinecone Configuration
PINECONE_API_KEY = "pcsk_RtTvf_FprhEhvmE6aJGckZg9P14Rny1q19p1QnGuRt67eCDWLMzJe3yW1LqaJ8rs1RcyE"  # Replace with your Pinecone API Key
PINECONE_ENV = "us-east-1"  # Replace with your Pinecone environment (you can find it on the Pinecone dashboard)
PINECONE_HOST = "https://pdfrag-bxptziy.svc.aped-4627-b74a.pinecone.io"  # The host URL provided by Pinecone
PINECONE_INDEX_NAME = "pdfrag"  # Name of the Pinecone index

# Initialize Pinecone using the new method
# Create a Pinecone client instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check and create the Pinecone index with dimension 384 if it doesn't exist
def create_pinecone_index():
    """Create Pinecone index if not exists."""
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        # Create a new index with dimension 384 (matching HuggingFace embeddings)
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  # Set dimension to 384 (as your embeddings are of size 384)
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' created with dimension 384.")
    else:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

# Initialize HuggingFace Embeddings model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def load_pdf(file_path: str) -> list[Document]:
    """Load a PDF file and return its documents."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

def split_documents(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> list[Document]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def insert_documents_into_pinecone(documents: list[Document]):
    """Insert document embeddings into Pinecone."""
    try:
        # Connect to the Pinecone index
        index = pc.Index(PINECONE_INDEX_NAME)

        # Prepare the embeddings and store them in Pinecone
        batch_size = 100  # Define the batch size
        batch = []

        # Get the embeddings for all documents
        document_texts = [doc.page_content for doc in documents]  # Extract text from documents
        embeddings_list = embeddings.embed_documents(document_texts)  # Use embed_documents method

        for i, doc in enumerate(documents):
            embedding = embeddings_list[i]  # Get the embedding for the current document
            metadata = {
                "text": doc.page_content
            }
            # Use document ID (e.g., a simple ID based on the index) to store the embeddings
            batch.append((f"doc_{i}", embedding, metadata))
            
            # Insert embeddings into Pinecone in batches
            if len(batch) >= batch_size:
                index.upsert(vectors=batch)
                print(f"Upserted {len(batch)} vectors into Pinecone.")
                batch = []  # Clear the batch

        # Upsert any remaining documents
        if batch:
            index.upsert(vectors=batch)
            print(f"Upserted {len(batch)} vectors into Pinecone.")

    except Exception as e:
        print(f"Error inserting documents into Pinecone: {e}")

def process_pdfs_and_create_pinecone():
    """Process PDFs from the 'documents' folder and create a vector store in Pinecone."""
    all_documents = []

    # Load all PDFs from the predefined 'documents' folder
    pdf_files = [os.path.join(DOCUMENTS_FOLDER, f) for f in os.listdir(DOCUMENTS_FOLDER) if f.endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in the folder '{DOCUMENTS_FOLDER}'.")
        return

    # Load and process each PDF
    for pdf_file in pdf_files:
        documents = load_pdf(pdf_file)
        if not documents:
            print(f"Failed to load PDF: {pdf_file}")
            continue
        all_documents.extend(documents)

    # Split documents into smaller chunks
    split_docs = split_documents(all_documents)

    # Create Pinecone index if it doesn't already exist
    create_pinecone_index()

    # Insert documents into Pinecone
    insert_documents_into_pinecone(split_docs)

if __name__ == "__main__":
    process_pdfs_and_create_pinecone()
