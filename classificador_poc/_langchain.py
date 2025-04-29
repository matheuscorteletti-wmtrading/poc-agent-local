import json
from typing import List

import boto3
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS


class BedrockEmbeddings(Embeddings):
    def __init__(self, model_id="amazon.titan-embed-text-v1", region_name="us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self.model_id = model_id

    def embed_query(self, text: str) -> List[float]:
        body = {"inputText": text}
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response["body"].read())
        return response_body["embedding"]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings

with open("docs/Tabela_NCM_Vigente_20250415_hierarquico.json", "r", encoding="utf-8") as f:
    json_ncm_hierarquico = json.load(f)

docs = []

for codigo, descricao in json_ncm_hierarquico.items():
    docs.append(
        Document(
            page_content=descricao,
            metadata={'ncm', codigo}
        )
    )

embeddings = BedrockEmbeddings()

db = FAISS.from_documents(docs, embeddings)

db.save_local("faiss_ncm_index")

