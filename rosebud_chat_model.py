# Langchain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.query_constructor.base import AttributeInfo, StructuredQuery
from langchain_community.query_constructors.pinecone import PineconeTranslator
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSerializable
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

# Pinecone
from pinecone import Pinecone

# General
import json
from dotenv import load_dotenv
import os
from typing import Optional
from typing import Dict
import logging

# Weave
import weave


# Changed from inheritance to regular class to fix infinite recursion
class rosebud_chat_model:
    def __init__(self, **kwargs):
        # Removed super().__init__(**kwargs) call
        load_dotenv()
        self.RETRIEVER_MODEL_NAME = None
        self.SUMMARY_MODEL_NAME = None
        self.EMBEDDING_MODEL_NAME = None
        self.constructor_prompt = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain_with_source = None
        self.query_constructor = None
        self.context = None
        self.top_k = None
        self.dev_mode = kwargs.get('dev_mode', False)
        
        with open('./config.json') as f:
            config = json.load(f)
            self.RETRIEVER_MODEL_NAME = config["RETRIEVER_MODEL_NAME"]
            self.SUMMARY_MODEL_NAME = config["SUMMARY_MODEL_NAME"]
            self.EMBEDDING_MODEL_NAME = config["EMBEDDING_MODEL_NAME"]
            self.top_k = config["top_k"]
        self.initialize_query_constructor()
        
        try:
            self.initialize_vector_store()
            self.initialize_retriever()
            self.initialize_chat_model(config)
        except Exception as e:
            logging.error(f"Error initializing vector store or retriever: {str(e)}")
            if not self.dev_mode:
                raise Exception(f"Failed to initialize Pinecone vector store. Please check your API key and environment variables: {str(e)}")
            else:
                print(f"Running in dev mode, continuing without vector store: {str(e)}")

    def initialize_query_constructor(self):
        document_content_description = "Brief overview of a movie, along with keywords"

        # Define allowed comparators list
        allowed_comparators = [
            "$eq",  # Equal to (number, string, boolean)
            "$ne",  # Not equal to (number, string, boolean)
            "$gt",  # Greater than (number)
            "$gte",  # Greater than or equal to (number)
            "$lt",  # Less than (number)
            "$lte",  # Less than or equal to (number)
            "$in",  # In array (string or number)
            "$nin",  # Not in array (string or number)
        ]

        # Define allowed operators list
        allowed_operators = [
            "AND",
            "OR"
        ]

        examples = [
            (
                "Recommend some films similar to star wars movies but not part of the star wars universe.",
                {
                    "query": "space opera, adventure, epic battles",
                    "filter": "and(nin('Title', ['Star Wars']), in('Genre', ['Science Fiction', 'Adventure']))"
                }
            ),
            (
                "Show me critically acclaimed dramas without Tom Hanks.",
                {
                    "query": "critically acclaimed drama",
                    "filter": "and(eq('Genre', 'Drama'), nin('Actors', ['Tom Hanks']), gt('Rating', 7))",
                },
            ),
            (
                "Recommend some films by Yorgos Lanthimos.",
                {
                    "query": "Yorgos Lanthimos",
                    "filter": 'in("Directors", ["Yorgos Lanthimos]")',
                },
            ),
            (
                "Films similar to Yorgos Lanthmios movies.",
                {
                    "query": "Dark comedy, absurd, Greek Weird Wave",
                    "filter": 'NO_FILTER',
                },
            ),
            (
                "Find me thrillers with a strong female lead released between 2015 and 2020.",
                {
                    "query": "thriller strong female lead",
                    "filter": "and(eq('Genre', 'Thriller'), gt('Release Year', 2015), lt('Release Year', 2021))",
                },
            ),
            (
                "Find me highly rated drama movies in English that are less than 2 hours long",
                {
                    "query": "Highly rated drama English",
                    "filter": 'and(eq("Genre", "Drama"), eq("Language", "English"), lt("Runtime (minutes)", 120), gt("Rating", 7))',
                },
            ),
            (
                "Short films that discuss the meaning of life.",
                {
                    "query": "meaning of life",
                    "filter": 'lt("Runtime (minutes)", 40))',
                },
            ),
        ]

        metadata_field_info = [
            AttributeInfo(name="Title", description="The title of the movie",
                          type="string"),
            AttributeInfo(name="Runtime (minutes)", description="The runtime of the movie in minutes",
                          type="integer"),
            AttributeInfo(name="Language", description="The language of the movie",
                          type="string"),
            AttributeInfo(name="Release Year", description="The release year of the movie",
                          type="integer"),
            AttributeInfo(name="Genre", description="The genre of the movie",
                          type="string or list[string]"),
            AttributeInfo(name="Actors", description="The actors in the movie",
                          type="string or list[string]"),
            AttributeInfo(name="Directors", description="The directors of the movie",
                          type="string or list[string]"),
            AttributeInfo(name="Stream", description="The streaming platforms for the movie",
                          type="string or list[string]"),
            AttributeInfo(name="Buy", description="The platforms where the movie can be bought",
                          type="string or list[string]"),
            AttributeInfo(name="Rent", description="The platforms where the movie can be rented",
                          type="string or list[string]"),
            AttributeInfo(name="Production Companies",
                          description="The production companies of the movie", type="string or list[string]"),
            AttributeInfo(name="Rating",
                          description="Rating of a film, out of 10", type="float"),
        ]

        self.constructor_prompt = get_query_constructor_prompt(
            document_content_description,
            metadata_field_info,
            allowed_comparators=allowed_comparators,
            allowed_operators=allowed_operators,
            examples=examples,
        )

    def initialize_vector_store(self):
        # Create empty index
        PINECONE_KEY = os.getenv('PINECONE_API_KEY')
        PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        
        # Verify we have the correct API key format
        if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-"):
            # Try to get the correct key directly from the .env file
            env_file_path = os.path.join(os.path.dirname(__file__), '.env')
            try:
                with open(env_file_path, 'r') as f:
                    for line in f:
                        if line.startswith('OPENAI_API_KEY='):
                            OPENAI_API_KEY = line.split('=', 1)[1].strip()
                            if OPENAI_API_KEY.startswith("sk-"):
                                os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
                                logging.info("Loaded corrected API key from .env file")
                            break
            except Exception as e:
                logging.error(f"Error reading API key from .env file: {str(e)}")
        
        if not PINECONE_KEY or not PINECONE_INDEX_NAME:
            raise ValueError("Missing Pinecone API key or index name. Please check your .env file.")
        
        try:
            pc = Pinecone(api_key=PINECONE_KEY)
            
            # Target index and check status
            pc_index = pc.Index(PINECONE_INDEX_NAME)
            
            embeddings = OpenAIEmbeddings(
                model=self.EMBEDDING_MODEL_NAME,
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                organization=os.getenv('OPENAI_ORGANIZATION')
            )
            
            namespace = "film_search_prod"
            self.vectorstore = PineconeVectorStore(
                index=pc_index,
                embedding=embeddings,
                namespace=namespace
            )
        except Exception as e:
            logging.error(f"Failed to initialize Pinecone: {str(e)}")
            raise

    def initialize_retriever(self):
        if not self.vectorstore:
            if self.dev_mode:
                # Skip retriever initialization in dev mode if vector store failed
                return
            raise ValueError("Vector store not initialized. Cannot create retriever.")
            
        query_model = ChatOpenAI(
            model=self.RETRIEVER_MODEL_NAME,
            temperature=0,
            streaming=True,
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            organization=os.getenv('OPENAI_ORGANIZATION')
        )

        output_parser = StructuredQueryOutputParser.from_components()
        self.query_constructor = self.constructor_prompt | query_model | output_parser

        self.retriever = SelfQueryRetriever(
            query_constructor=self.query_constructor,
            vectorstore=self.vectorstore,
            structured_query_translator=PineconeTranslator(),
            search_kwargs={'k': self.top_k}
        )

    def initialize_chat_model(self, config):
        if not self.retriever and not self.dev_mode:
            raise ValueError("Retriever not initialized. Cannot create chat model.")
            
        def format_docs(docs):
            return "\n\n".join(f"{doc.page_content}\n\nMetadata: {doc.metadata}" for doc in docs)

        chat_model = ChatOpenAI(
            model=self.SUMMARY_MODEL_NAME,
            temperature=config['TEMPERATURE'],
            streaming=True,
            max_retries=10,
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            organization=os.getenv('OPENAI_ORGANIZATION')
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    """
                    Your goal is to recommend films to users based on their
                    query and the retrieved context. If a retrieved film doesn't seem
                    relevant, omit it from your response. If your context is empty
                    or none of the retrieved films are relevant, do not recommend films, but instead
                    tell the user you couldn't find any films that match their query.
                    Aim for three to five film recommendations, as long as the films are relevant. You cannot
                    recommend more than five films. Your recommendation should
                    be relevant, original, and at least two to three sentences
                    long.

                    YOU CANNOT RECOMMEND A FILM IF IT DOES NOT APPEAR IN YOUR
                    CONTEXT.

                    # TEMPLATE FOR OUTPUT
                    - **Title of Film**:
                        - **Runtime:**
                        - **Release Year:**
                        - **Streaming:**
                        - Your reasoning for recommending this film

                    Question: {question}
                    Context: {context}
                    """
                ),
            ]
        )

        # Create a chatbot Question & Answer chain from the retriever
        if self.retriever:
            rag_chain_from_docs = (
                RunnablePassthrough.assign(
                    context=(lambda x: format_docs(x["context"]))) | prompt | chat_model | StrOutputParser()
            )
            
            self.rag_chain_with_source = RunnableParallel(
                {"context": self.retriever, "question": RunnablePassthrough(), "query_constructor": self.query_constructor}
            ).assign(answer=rag_chain_from_docs)
        elif self.dev_mode:
            # In dev mode with no retriever, create a simple chat model that explains the situation
            self.rag_chain_with_source = RunnablePassthrough() | (lambda x: {
                "answer": "I'm running in development mode without Pinecone. Please check your API keys and configuration.",
                "context": "",
                "query_constructor": {}
            })

    # Modified to work without Weave decorator
    def predict_stream(self, query: str):
        try:
            # Initialize Weave only if needed for tracking
            try:
                weave.init('film-search')
            except Exception as e:
                logging.warning(f"Could not initialize Weave: {str(e)}")
            
            if not self.rag_chain_with_source:
                yield "Error: Chat model not initialized. Please check API keys and configuration."
                return
                
            for chunk in self.rag_chain_with_source.stream(query):
                if 'answer' in chunk:
                    yield chunk['answer']
                elif 'context' in chunk:
                    docs = chunk['context']
                    self.context = "\n\n".join(f"{doc.page_content}\n\nMetadata: {doc.metadata}" for doc in docs)
                elif 'query_constructor' in chunk:
                    self.query_constructor = chunk['query_constructor'].json()

        except Exception as e:
            yield f"An error occurred: {e}"

    # Modified to work without Weave decorator
    async def predict(self, query: str):
        try:
            # Initialize Weave only if needed for tracking
            try:
                weave.init('film-search')
            except Exception as e:
                logging.warning(f"Could not initialize Weave: {str(e)}")
            
            if not self.rag_chain_with_source:
                return {
                    'answer': "Error: Chat model not initialized. Please check API keys and configuration.",
                    'context': ""
                }
                
            result = self.rag_chain_with_source.invoke(query)
            return {
                'answer': result['answer'],
                'context': "\n".join(f"{doc.page_content}\n\nMetadata: {doc.metadata}" for doc in result['context']) if 'context' in result and result['context'] else ""
            }
        except Exception as e:
            return {'answer': f"An error occurred: {e}", 'context': ""}
