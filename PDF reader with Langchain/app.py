from dotenv import load_dotenv
import streamlit as st
import altair as alt
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback





def main():
    load_dotenv() #loads openai secret key
    #print(os.getenv("OPENAI_API_KEY"))
    st.set_page_config(page_title="PDF reader")
    st.header("Enter you PDF here")

    #Takes and uploading the pdf
    pdf = st.file_uploader("Upload you pdf", type="pdf")
    
    #Extracts and reading the pdf
    if pdf is not None:
        pdf_reader = PdfReader(pdf) #this is just for taking the text using the PyPDF2 module/library
        text="" #creats an empty string

        for page in pdf_reader.pages:
            text += page.extract_text() #this for extarcts all the texts as one big singular string

        
        #Chunks and vectorising data
        text_splitter= CharacterTextSplitter(
            separator="\n",
            chunk_size = 500,
            chunk_overlap = 100,
            length_function=len    
        )  #this above piece of code creates a tuple and specifically instructs how to split the code using Langchain

        chunks = text_splitter.split_text(text) #split.text() returns a list displayed as chunks
        st.write(chunks)

        #note that the above part is still in text and not yet vectorised for the embedding process

        vector_embeddings = OpenAIEmbeddings() #langchain is the wrapper and the openai does the actual vector embedding
        knowledge_base = FAISS.from_texts(chunks, vector_embeddings) #(data, function(method executed)) #this creates the relation for the sematic search developed by Facebook

        #The below part is for taking user input question after PDF vectorinsing and creating the sematic search database

        user_question = st.text_input("Enter you text input: ")
        if user_question:
         docs = knowledge_base.similarity_search(user_question)
        
         llm = OpenAI()
         chain = load_qa_chain(llm, chain_type="stuff")
         with get_openai_callback() as cb:
           response = chain.run(input_documents=docs, question=user_question)
           print(cb)
           
        st.write(response)



if __name__=='__main__':
    main()

