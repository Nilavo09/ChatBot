import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import google.cloud
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load the environment variables (Google API Key)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)  # Pass the text variable to split_text
    return chunks

# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="model/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, 
    make sure to provide all the details, if the answer is not in the provided context just say,
    "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="model/embedding-001")
    
    # Load the FAISS vector store
    new_db = FAISS.load_local("faiss_index", embeddings)
    
    # Perform a similarity search on the vector store
    docs = new_db.similarity_search(user_question)

    # Get the conversational chain
    chain = get_conversational_chain()

    # Get the response to the user's question
    response = chain(
       {"input_documents": docs, "question": user_question},
       return_only_outputs=True
    )

    # Output the response
    st.write("Reply: ", response["output_text"])

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Chat With Multiple PDF", page_icon="😼")
    st.header("Chat with Multiple PDF using GeminiGemini 😼")
   
    # Get the user's question
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        # Allow uploading of multiple PDF files
        pdf_docs = st.file_uploader("Upload your PDF Files", type=["pdf"], accept_multiple_files=True)
        
        # Button to submit and process the PDFs
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # Extract text from uploaded PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Split the raw text into chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Create the vector store from the chunks
                    get_vector_store(text_chunks)
                    
                    # Show success message
                    st.success("Processing Complete!")
            else:
                st.error("Please upload at least one PDF.")

# Run the app
if __name__ == "__main__": 
    main()
