# **Streamlit LLM Chatbot App**

**This is a Streamlit app that uses AWS Bedrock to work with different LLM models. It is hosted on: [https://dhruvsharma19191.com/](https://dhruvsharma19191.com/)**

Welcome to my Streamlit LLM Chatbot App! This application leverages the power of AWS Bedrock and various large language models (LLMs) to provide a robust Q&A chatbot functionality. Here are the key features and workflows of this app:

## Features

1. **Data Ingestion**: Reads PDF documents from a folder.
2. **Document Processing**:
   - Splits PDFs into chunks.
   - Creates embeddings using FAISS vector store.
   - Utilizes Amazon Titan for advanced processing.
3. **Question Answering**:
   - Checks for similarity from the vector store.
   - Retrieves relevant chunks.
   - Asks the LLM with the provided prompt for answers.
4. **Interactive Q&A Chatbot**: Provides an interactive interface for users to ask questions and receive answers based on the ingested documents.

### Prerequisites

- Python 3.7+
- Streamlit
- AWS Account
- AWS CLI configured with appropriate permissions

### Installation

1. Clone the repository
   ```sh
   git clone 
