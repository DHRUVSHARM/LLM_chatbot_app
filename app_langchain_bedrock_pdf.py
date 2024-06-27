import json
import os
import sys
import boto3
import streamlit as st
from fpdf import FPDF
import os

# using titan embeddings model for creating vectors / embeddings

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# data ingestion / loading

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# vector embeddings and vector store
from langchain_community.vectorstores import FAISS

## LLM models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from python.botocore.exceptions import NoCredentialsError, PartialCredentialsError

bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock
)


def save_text_to_pdf(text, filename):
    # saving the output from the llms as a pdf to be used to retrieve from the vectorstore
    if not os.path.exists("data"):
        print("creating the data directory ...")
        os.makedirs("data")

    print("creating a pdf to store for vectorstore ...")
    # Create a PDF document
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Set font
    pdf.set_font("Arial", size=12)

    # Add text to the PDF
    pdf.multi_cell(0, 10, text)

    # Save the PDF to the specified filename
    pdf.output(filename)

    print(f"pdf : {filename} , stored ...")


# data ingestion step :
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    print("the loader is : ", loader)
    documents = loader.load()

    # recursive text splitting step
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

    docs = text_splitter.split_documents(documents)
    print("the docs are : ", docs)
    return docs


# Vector embeddings and vector store
def get_vector_store(docs):
    print("loading vectorstore from docs : ", docs)
    try:
        vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        # saving locally , can also be done in db
        vectorstore_faiss.save_local("faiss_index")
    except Exception as e:
        print("error caught : ", e)


# in our app we will have 2 options , this can be extended if required , jurassic and llama2


def get_jurassic_llm():
    # get the jurassic model
    llm = Bedrock(
        model_id="ai21.j2-mid-v1", client=bedrock, model_kwargs={"maxTokens": 512}
    )
    return llm


def get_llama2_llm():
    # get the llam2 model
    llm = Bedrock(
        model_id="meta.llama2-13b-chat-v1",
        client=bedrock,
        model_kwargs={"max_gen_len": 512},
    )
    return llm


# defining the prompt template to interact with the llms

prompt_template = """

Human: Use the following pieces of context to provide a concise answer to the question at the end but use atleast
250 words and summarize with detailed explanation. If you do not know the answer, say you don't know but don't 
try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    answer = qa({"query": query})
    return answer["result"]


def main():
    st.set_page_config("Chat PDF")

    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    jp_index, lama_index = 1, 1
    if st.button("Jurassic Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(
                "faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True
            )
            llm = get_jurassic_llm()

            # faiss_index = get_vector_store(docs)
            res = get_response_llm(llm, faiss_index, user_question)
            st.write(res)
            save_text_to_pdf(res, f"data/jurassic_op{jp_index}.pdf")
            jp_index += 1
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(
                "faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True
            )
            llm = get_llama2_llm()

            # faiss_index = get_vector_store(docs)
            res = get_response_llm(llm, faiss_index, user_question)
            st.write(res)
            save_text_to_pdf(res, f"data/llama2_op{lama_index}.pdf")
            lama_index += 1
            st.success("Done")


def check_aws_credentials():
    try:
        client = boto3.client("sts")
        response = client.get_caller_identity()
        print("AWS Credentials are set up correctly. User ID: ", response["UserId"])
    except NoCredentialsError:
        print("No AWS credentials found.")
    except PartialCredentialsError:
        print("Incomplete AWS credentials found.")
    except Exception as e:
        print("An error occurred: ", e)


if __name__ == "__main__":
    check_aws_credentials()
    main()
