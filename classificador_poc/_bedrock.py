import json
import boto3

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