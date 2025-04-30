import json
import boto3
from typing import List

from langchain.embeddings.base import Embeddings
bedrock = boto3.client('bedrock-runtime', region_name="us-east-1")

def build_embedding(texto):
    body = {
        "inputText": texto
    }
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response['body'].read())
    return response_body['embedding']  # extrai vetor

class BedrockEmbeddings(Embeddings):
    def __init__(self, model_id="amazon.titan-embed-text-v2:0", region_name="us-east-1"):
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