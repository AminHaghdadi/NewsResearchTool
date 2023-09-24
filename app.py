from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import time
import streamlit as st
import torch
from key import openai_API
import os
os.environ["OPENAI_API_KEY"]=openai_API

# Check if GPU is available, otherwise use CPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu" 

device = torch.device(device)

llm = OpenAI(temperature=0.1, max_tokens=500)

DB_PATH="vectorstore/db_faiss"

st.title("News Research Tool 📰")
st.sidebar.title("News Article URLs")

urls=[]
for i in range(3):
    url=st.sidebar.text_input(f'url{i+1}')
    urls.append(url)


side_button=st.sidebar.button("Start Process")
st.sidebar.subheader("Powered by chatGPT3.5",divider='rainbow')
placeholder=st.empty()

if side_button :
    print(urls)
    loader=UnstructuredURLLoader(urls=urls)
    placeholder.text("Data Loading...Started...✅✅✅")
    
    data=loader.load()
    
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000)
    
    placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
  
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device': device}) 

 
    db = FAISS.from_documents(docs, embeddings)
    placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)
    db.save_local(DB_PATH)
    
query=placeholder.text_input("Question: ")

if query:
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device': device}) 
    vectorstore=FAISS.load_local(DB_PATH,embeddings)
    
    chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorstore.as_retriever())
    result = chain({"question": query}, return_only_outputs=True)
    st.header("Answer")
    st.write(result["answer"])
    
            
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  
        for source in sources_list:
                st.write(source)