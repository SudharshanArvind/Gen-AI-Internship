import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Hardcoded Google API Key (for testing purposes only)
GOOGLE_API_KEY = "AIzaSyBTCgR0R3MB2Iz-btzr_qqmmky4DQNQYsk"
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # ✅ Fix: Set environment variable

# Initialize Google Palm LLM
st.write("Initializing Google Palm LLM...")
try:
    llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
    st.write("Google Palm LLM initialized.")
except Exception as e:
    st.error(f"Error initializing Google Palm LLM: {e}")

# Initialize instructor embeddings using Hugging Face
st.write("Loading Hugging Face embeddings...")
try:
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    st.write("Hugging Face embeddings loaded.")
except Exception as e:
    st.error(f"Error loading Hugging Face embeddings: {e}")

# Paths
VECTORDATABASE_PATH = "faiss_index"
DATASET_PATH = "data/bitext_dataset.csv"  # Ensure this file exists

def create_vector_db():
    """Loads dataset, creates FAISS vector DB, and saves it locally."""
    st.write("Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        st.error(f"❌ Error: Dataset file '{DATASET_PATH}' not found!")
        return
    
    try:
        loader = CSVLoader(file_path=DATASET_PATH, source_column="prompt")
        data = loader.load()
        st.write(f"✅ Loaded {len(data)} documents.")
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return

    # Check if the FAISS index already exists
    if os.path.exists(VECTORDATABASE_PATH):
        st.write("FAISS vector database already exists, skipping creation.")
        return
    
    st.write("Creating FAISS vector database...")
    try:
        vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
        vectordb.save_local(VECTORDATABASE_PATH)
        st.write("✅ FAISS vector database created and saved.")
    except Exception as e:
        st.error(f"❌ Error creating FAISS vector database: {e}")

def get_qa_chain():
    """Loads FAISS DB and returns a LangChain-based RetrievalQA model."""
    st.write("Loading FAISS vector database...")
    try:
        vectordb = FAISS.load_local(VECTORDATABASE_PATH, embedding=instructor_embeddings)
        st.write("✅ FAISS vector database loaded.")
    except Exception as e:
        st.error(f"❌ Error loading FAISS vector database: {e}")
        return None

    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    Try to provide as much relevant information as possible from the "response" section of the dataset.
    If an answer is not found, reply with "I don't know."
    CONTEXT: {context}
    QUESTION: {question}
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    st.write("Initializing QA chain...")
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        st.write("✅ QA chain initialized.")
        return qa_chain
    except Exception as e:
        st.error(f"❌ Error initializing QA chain: {e}")
        return None

# Streamlit UI
st.title("LangChain QA Bot")
st.write("Ask any question based on the dataset!")

# Initialize the vector database and QA chain
create_vector_db()
chain = get_qa_chain()

# User input
user_query = st.text_input("Enter your question:")

if user_query:
    if chain is not None:
        st.write("Processing your question...")
        try:
            response = chain({"query": user_query})  # FIXED
            st.write("Answer:", response["result"])
        except Exception as e:
            st.error(f"❌ Error processing your question: {e}")
    else:
        st.error("QA chain is not initialized. Please check the logs for errors.")
