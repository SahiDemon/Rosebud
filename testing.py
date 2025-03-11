import os
from rosebud_chat_model import rosebud_chat_model

os.environ["OPENAI_API_KEY"] = "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"
chat_model = rosebud_chat_model()
query = "Recommend some films similar to star wars movies but not part of the star wars universe"

query_constructor = chat_model.query_constructor.invoke(query)


print(f"query_constructor: {query_constructor}")

print("response:")
for chunk in chat_model.rag_chain_with_source.stream(query):
    print(chunk)
