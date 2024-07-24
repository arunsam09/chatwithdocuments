import streamlit as st

from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub
import regex as re
import os


def generate_response(uploaded_file, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        pdfloader = PyPDFLoader(uploaded_file.name)
        documents = pdfloader.load_and_split()
        # Split documents into chunks

        # Select embeddings
        embeddings = HuggingFaceInferenceAPIEmbeddings(api_key="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", model_name="thenlper/gte-large")
        # Create a vectorstore from documents
        db = Chroma.from_documents(documents, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        model = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", huggingfacehub_api_token="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", model_kwargs={"temperature":0.01, "do_sample": True, "max_new_tokens":1024},)
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=model, chain_type='stuff', retriever=retriever)
        return qa.run(query_text)


# Page title
st.set_page_config(page_title='🦜🔗 Document Chat Powered by Huggingface')
st.title('🦜🔗 Ask the Document !!!!')


# File upload
uploaded_file = st.file_uploader('Upload an article', type='pdf')
# Query text
query_text = st.text_input('Enter your question:')

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted :
        with st.spinner('Calculating...'):
            with open(uploaded_file.name, mode='wb') as w:
                w.write(uploaded_file.getvalue())
            response = generate_response(uploaded_file, query_text)
            result.append(response)

            match = re.search(r'Helpful Answer: (.+)', response)
            helpful_answer = match.group(1).strip()
            print(helpful_answer)


if len(result):
    if os.path.exists(uploaded_file.name):
        os.remove(uploaded_file.name)

    st.info(helpful_answer)
