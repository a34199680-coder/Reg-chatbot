import os
import streamlit as st
import google.generativeai as genai
from pdf_extractor import text_extractor
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Lets Configures the Models
# LLM Model
gemini_key = os.getenv('GEMINI-API-KEY')
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Embdeding Model
embeding_model = HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L6-v2')


# Lets Create the Main Page
st.title(":orange[CHATBOT:]:blue[AI Assisted chatbot using RAG.]")

tips = '''
Follow the steps to use the application:-
* Upload your PDF Document in sidebar.
* Write a query and start the chat.
'''
st.text(tips)


# Lets create the sidebar
st.sidebar.title(":rainbow[Upload Your File ]")
st.sidebar.subheader(":red[Upload PDF file only]")
pdf_file =st.sidebar.file_uploader("Upload Here",type=['pdf'])
if pdf_file :
    st.sidebar.success('File Uploaded successfully')

    file_text = text_extractor(pdf_file)

    # Step 1: Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size= 1000,chunk_overlap=200)
    chunks = splitter.split_text(file_text)

    # Step 2: Create the vector DB
    vector_store = FAISS.from_texts(chunks,embeding_model)
    retriever = vector_store.as_retriever(search_kwargs={'k':3})

    def generate_content(query):
        # Step 3: Retrieveral (R)
        retrived_docs = retriever.invoke(query)
        context = '\n'.join([d.page_content for d in retrived_docs])

        #Step 4: Augumenting (A)
        augmented_prompt = f''' 
        <Role> You are a helpful assistant using RAG.
        <Goal> Answer the question asked by the user. Here are the questions {query}.
        <Context> Here are the documents retrieve from the Vector database to support the answer which you have to generate {context}
        '''

        # Step 5 : Generation (G)
        response  = model.generate_content(augmented_prompt)
        return response.text
    
    # Create Chatbot in order to start the conversion
    # For intialize a chat create history if not created.
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Display the history
    for msg in st.session_state.history:
        if msg['role']=='user':
            st.info(f"USER: {msg['text']}")
        else:
            st.warning(f"CHATBOT: {msg['text']}")
    
    # Input from the user using Streamlit form
    with st.form('Chatbot Form',clear_on_submit=True):
        user_query = st.text_area('Ask Anything.')
        send = st.form_submit_button('Send')
    
    # We have to start the conversation and append output and query in history
    if user_query and send:
        st.session_state.history.append({'role':'user','text':user_query})
        st.session_state.history.append({'role':'chatbot','text':generate_content(user_query)})
        st.rerun()









