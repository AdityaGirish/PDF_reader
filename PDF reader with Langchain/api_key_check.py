import openai

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
api_key = 'API_KEY'

# Initialize the OpenAI API client with your key
openai.api_key = api_key

# Test the API by making a sample request
response = openai.Completion.create(
    engine="davinci",
    prompt="Once upon a time",
    max_tokens=5
)

# Print the response from the API
print(response)
