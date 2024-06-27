1) data ingestion : read from the folder the pdfs
2) take the pdf docs , split chunks , create embeddings and use faiss vectorstore , to create amazon titan
3) ask a question , check for similarity from vectorstore , get relevant chunks and ask the llm with the prompt 
4) basically it's like a q and a chatbot 

