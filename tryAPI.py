import boto3
import os

# Set the API key as an environment variable
os.environ['AWS_BEARER_TOKEN_BEDROCK']

# Create the Bedrock client
client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Define the model and message
model_id = "mistral.voxtral-mini-3b-2507"
messages = [{"role": "user", "content": [{"text": "Hello! Can you tell me about Amazon Bedrock?"}]}]

# Make the API call
response = client.converse(
    modelId=model_id,
    messages=messages,
)

# Print the response
print(response['output']['message']['content'][0]['text'])