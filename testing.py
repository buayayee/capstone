import boto3, json

# Uses IAM role / ~/.aws/credentials automatically — no API key needed
client = boto3.client("bedrock-runtime", region_name="us-east-1")

response = client.converse(
    modelId="openai.gpt-oss-120b-1:0",
    messages=[
        {"role": "user", "content": [{"text": "Can you explain the features of Amazon Bedrock?"}]}
    ],
    inferenceConfig={"maxTokens": 512, "temperature": 0.0},
)

print(response["output"]["message"]["content"][0]["text"])