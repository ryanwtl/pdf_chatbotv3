
import streamlit as st  
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import requests
import tempfile
import uuid
import pandas as pd
import re
import base64
from dotenv import load_dotenv
import os
import time

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
groqcloud_api_key = os.getenv("GROQCLOUD_API_KEY")

# st.write(google_api_key)

# Ensure API key is set
if not groqcloud_api_key:
    st.error("GroqCloud API key not found. Please set GROQCLOUD_API_KEY in your .env file.")

# # Initialize the API key in session state if it doesn't exist
# if 'api_key' not in st.session_state:
#     st.session_state.api_key = ''

def clean_filename(filename):
    """
    Remove "(number)" pattern from a filename 
    (because this could cause error when used as collection name when creating Chroma database).

    Parameters:
        filename (str): The filename to clean

    Returns:
        str: The cleaned filename
    """
    # Regular expression to find "(number)" pattern
    new_filename = re.sub(r'\s\(\d+\)', '', filename)
    return new_filename

def display_pdf(uploaded_file):

    """
    Display a PDF file that has been uploaded to Streamlit.

    The PDF will be displayed in an iframe, with the width and height set to 700x1000 pixels.

    Parameters
    ----------
    uploaded_file : UploadedFile
        The uploaded PDF file to display.

    Returns
    -------
    None
    """
    # Read file as bytes:
    bytes_data = uploaded_file.getvalue()
    
    # Convert to Base64
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    
    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    
    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)

def split_document(documents, chunk_size, chunk_overlap):    
    """
    Function to split generic text into smaller chunks.
    chunk_size: The desired maximum size of each chunk (default: 400)
    chunk_overlap: The number of characters to overlap between consecutive chunks (default: 20).

    Returns:
        list: A list of smaller text chunks created from the generic text
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                          chunk_overlap=chunk_overlap,
                                          length_function=len,
                                          separators=["\n\n", "\n", " "])
    
    return text_splitter.split_documents(documents)

def get_embedding_function():
    """
    Return an OpenAIEmbeddings object, which is used to create vector embeddings from text.
    The embeddings model used is "text-embedding-ada-002" and the OpenAI API key is provided
    as an argument to the function.

    Parameters:
        api_key (str): The OpenAI API key to use when calling the OpenAI Embeddings API.

    Returns:
        OpenAIEmbeddings: An OpenAIEmbeddings object, which can be used to create vector embeddings from text.
    """
    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-ada-002", openai_api_key=api_key
    # )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings

def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):

    """
    Create a vector store from a list of text chunks.

    :param chunks: A list of generic text chunks
    :param embedding_function: A function that takes a string and returns a vector
    :param file_name: The name of the file to associate with the vector store
    :param vector_store_path: The directory to store the vector store

    :return: A Chroma vector store object
    """

    # Create a list of unique ids for each document based on the content
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
    # Ensure that only unique docs with unique ids are kept
    unique_ids = set()
    unique_chunks = []
    
    unique_chunks = [] 
    for chunk, id in zip(chunks, ids):     
        if id not in unique_ids:       
            unique_ids.add(id)
            unique_chunks.append(chunk)        

    # Create a new Chroma database from the documents
    # vectorstore = Chroma.from_documents(documents=unique_chunks, 
    #                                     collection_name=clean_filename(file_name),
    #                                     embedding=embedding_function, 
    #                                     ids=list(unique_ids), 
    #                                     persist_directory = vector_store_path)

    vectorstore = Chroma.from_documents(documents=unique_chunks, 
                                        embedding=embedding_function, 
                                        ids=list(unique_ids), 
                                        persist_directory = vector_store_path)

    # The database should save automatically after we create it
    # but we can also force it to save using the persist() method
    vectorstore.persist()
    
    return vectorstore


def create_vectorstore_from_texts(documents, api_key, file_name):
    
    # Step 2 split the documents  
    """
    Create a vector store from a list of texts.

    :param documents: A list of generic text documents
    :param api_key: The OpenAI API key used to create the vector store
    :param file_name: The name of the file to associate with the vector store

    :return: A Chroma vector store object
    """
    docs = split_document(documents, chunk_size=2000, chunk_overlap=400)
    
    # Step 3 define embedding function
    embedding_function = get_embedding_function()

    # Step 4 create a vector store  
    vectorstore = create_vectorstore(docs, embedding_function, file_name)
    
    return vectorstore


def get_pdf_text(uploaded_file): 
    """
    Load a PDF document from an uploaded file and return it as a list of documents

    Parameters:
        uploaded_file (file-like object): The uploaded PDF file to load

    Returns:
        list: A list of documents created from the uploaded PDF file
    """
    try:
        # Read file content
        input_file = uploaded_file.read()

        # Create a temporary file (PyPDFLoader requires a file path to read the PDF,
        # it can't work directly with file-like objects or byte streams that we get from Streamlit's uploaded_file)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()
        loader = PyPDFLoader(temp_file.name)
        documents = loader.load() 

        return documents
    
    finally:
        # Ensure the temporary file is deleted when we're done with it
        os.unlink(temp_file.name)

# Prompt template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.

{context}

---

Answer the question based on the above context: {question}
"""

def query_document1(vectorstore, query, api_key, model_name):

    """
    Query a vector store with a question and return a structured response.

    :param vectorstore: A Chroma vector store object
    :param query: The question to ask the vector store
    :param api_key: The OpenAI API key to use when calling the OpenAI Embeddings API

    :return: A pandas DataFrame with three rows: 'answer', 'source', and 'reasoning'
    """
    # llm = OllamaLLM(model="llama3.2:latest")

    retriever=vectorstore.as_retriever(search_type="similarity")
    topic = query

    relevant_chunks = retriever.invoke(topic)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Step 4: Prepare the API request
    url = "https://api.groq.com/openai/v1/chat/completions"  # Replace with the API endpoint you are using
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }


    data = {
        "model": model_name,  # Use the model name from the mapping
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {relevant_chunks}\nQuestion: {topic}"} # prompt_template
        ]
    }

    # Step 4: Make the API request
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        # return {"output_text": result}
    elif response.status_code == 429:  # Handle rate limits
        retry_after = response.json().get("error", {}).get("message", "")
        # return {"output_text": f"Rate limit exceeded. {retry_after}"}
    else:
        raise ValueError(f"GroqCloud API request failed with status {response.status_code}: {response.text}")
    
    # Step 6: Create a structured response in DataFrame format
    answer_row = [result]  # LLM's answer
    source_row = [relevant_chunks]  # The context used as a source
    reasoning_row = [relevant_chunks]  # Reasoning can be the context for now (can be improved)

    return result

def load_streamlit_page():

    """
    Load the streamlit page with two columns. The left column contains a text input box for the user to input their OpenAI API key, and a file uploader for the user to upload a PDF document. The right column contains a header and text that greet the user and explain the purpose of the tool.

    Returns:
        col1: The left column Streamlit object.
        col2: The right column Streamlit object.
        uploaded_file: The uploaded PDF file.
    """
    st.set_page_config(layout="wide", page_title="LLM Tool")

    # Design page layout with 2 columns: File uploader on the left, and other interactions on the right.
    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        st.header("WELCOME")
        # st.text_input('OpenAI API key', type='password', key='api_key',
        #             label_visibility="collapsed", disabled=False)
        st.header("Upload file")
        uploaded_file = st.file_uploader("Please upload your PDF document:", type= "pdf")

    return col1, col2, uploaded_file


# Make a streamlit page
col1, col2, uploaded_file = load_streamlit_page()

# Process the input
if uploaded_file is not None:
    with col2:
        display_pdf(uploaded_file)
        
    # Load in the documents
    documents = get_pdf_text(uploaded_file)
    st.session_state.vector_store = create_vectorstore_from_texts(documents, 
                                                                  api_key=google_api_key,
                                                                  file_name=uploaded_file.name)
    st.write("Input Processed")

# Generate answer
with col1:
    user_question = st.text_input("Ask a Question from the PDF Files")
    model_name = st.selectbox("Select Model", ["Llama3","Llama3.1","Llama3.3"])
    model_mapping = {
        "Llama3": "llama3-70b-8192",
        "Llama3.1": "llama-3.1-70b-versatile", 
        "Llama3.3": "llama-3.3-70b-versatile" 
    }
    if st.button("Generate response"):
        start_time = time.time()
        with st.spinner("Generating answer"):
            # Load vectorstore:
            answer = query_document1(vectorstore = st.session_state.vector_store, 
                                    query = user_question,
                                    api_key = groqcloud_api_key,
                                    model_name = model_mapping[model_name])
                            
            placeholder = st.empty()
            placeholder = st.write(answer)
        elapsed_time = time.time() - start_time
        st.write(f"Response Time: {elapsed_time:.2f} seconds")