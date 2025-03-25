import os
import re
from dotenv import load_dotenv
from rosebud_chat_model import rosebud_chat_model

# Function to read API key directly from .env file
def read_api_key_from_env_file(env_file_path='./.env'):
    try:
        with open(env_file_path, 'r') as f:
            for line in f:
                if line.startswith('OPENAI_API_KEY='):
                    # Extract the key value after the equals sign
                    api_key = line.split('=', 1)[1].strip()
                    return api_key
        return None
    except Exception as e:
        print(f"Error reading .env file: {e}")
        return None

# Load environment variables from .env file
load_dotenv()

# Verify the API key is loaded correctly
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not found in environment variables")
    exit(1)
else:
    # Check if the API key starts with "sk-"
    if not api_key.startswith("sk-"):
        print(f"Warning: API key doesn't look like an OpenAI key (should start with 'sk-')")
        print(f"Current value starts with: {api_key[:20]}...")
        
        # Try to read the correct key directly from .env file
        correct_key = read_api_key_from_env_file()
        
        if correct_key and correct_key.startswith("sk-"):
            print(f"Found correct API key in .env file, starting with: {correct_key[:10]}...")
            os.environ["OPENAI_API_KEY"] = correct_key
        else:
            print("Could not find valid OpenAI API key in .env file.")
            exit(1)
    else:
        print(f"API Key looks good, starting with: {api_key[:10]}...")

# Initialize the chat model after fixing the API key
chat_model = rosebud_chat_model()
query = "Recommend some films similar to star wars movies but not part of the star wars universe"

print("Running query constructor...")
query_constructor = chat_model.query_constructor.invoke(query)

print(f"query_constructor: {query_constructor}")

print("response:")
for chunk in chat_model.rag_chain_with_source.stream(query):
    print(chunk)
