from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.memory import VectorStoreRetrieverMemory

from langchain.chains import ConversationChain
# from langchain.chains import RunnableWithMessageHistory

from langchain.llms.bedrock import Bedrock
# from langchain_aws import BedrockLLM

from langchain.embeddings import BedrockEmbeddings
# from langchain_aws import BedrockEmbeddings

import boto3

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
llm = Bedrock(client=bedrock_client, model_id="amazon.titan-text-premier-v1:0")
embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v2:0",)

docs = [Document(page_content="Me chamo Matheus e gosto de IoT.")]

vectorstore = FAISS.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

memory = VectorStoreRetrieverMemory(retriever=retriever)
chain = ConversationChain(llm=llm, memory=memory)

result = chain.invoke({"input": "O que você sabe sobre mim?"})
print(result)