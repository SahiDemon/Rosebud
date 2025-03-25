import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Pinecone
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.client.exceptions import NotFoundException

# Langchain
from langchain_community.document_loaders import DirectoryLoader
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Prefect
from prefect import task, flow
from prefect.deployments import DeploymentImage

# General
import os
from dotenv import load_dotenv
import csv
from utils import get_id_list, get_data, write_file
import json

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

@task
def start():
    """
    Start-up: check everything works or fail fast!
    """

    # Print out some debug info
    print("Starting flow!")

    # Loading environment variables
    try:
        load_dotenv(verbose=True, dotenv_path='.env')
    except ImportError:
        print("Env file not found!")
    
    # Check if OPENAI_API_KEY is loaded correctly
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        # Try to get the correct key directly from the .env file
        api_key = read_api_key_from_env_file()
        if api_key and api_key.startswith("sk-"):
            os.environ["OPENAI_API_KEY"] = api_key
            print("Loaded API key from .env file")
        else:
            print("Warning: Could not find valid OpenAI API key")

    # Ensure user has set the appropriate env variables
    assert os.environ['LANGCHAIN_API_KEY']
    assert os.environ['TMBD_API_KEY']
    assert os.environ['PINECONE_API_KEY']
    assert os.environ['PINECONE_INDEX_NAME']
    assert os.environ['TMDB_BEARER_TOKEN']
    assert os.environ['LANGCHAIN_TRACING_V2']
    assert os.environ['WANDB_API_KEY']


@task(retries=3, retry_delay_seconds=[1, 10, 100])
def pull_data_to_csv(config):
    TMBD_API_KEY = os.getenv('TMBD_API_KEY')
    YEARS = range(config["years"][0], config["years"][-1] + 1)
    CSV_HEADER = ['Title', 'Runtime (minutes)', 'Language', 'Overview',
                  'Release Year', 'Genre', 'Keywords',
                  'Actors', 'Directors', 'Stream', 'Buy', 'Rent',
                  'Production Companies', 'Rating']

    for year in YEARS:
        # Grab list of ids for all films made in {YEAR}
        movie_list = list(set(get_id_list(TMBD_API_KEY, year)))

        FILE_NAME = f'./data/{year}_movie_collection_data.csv'

        # Creating file
        with open(FILE_NAME, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

        # Iterate through list of ids to get data
        for id in movie_list:
            dict = get_data(TMBD_API_KEY, id)
            write_file(FILE_NAME, dict)

    print("Successfully pulled data from TMDB and created csv files in data/")


@task
def convert_csv_to_docs():
    # Loading in data from all csv files
    docs = []
    for filename in os.listdir("./data"):
        if filename.endswith(".csv"):
            filepath = os.path.join("./data", filename)
            with open(filepath, "r", encoding="utf-8") as f:
                loader = CSVLoader(
                    file_path=filepath,
                    encoding="utf-8"
                )
                docs.extend(loader.load())
    
    print("Successfully took csv files and created docs")
    return docs


@task
def upload_docs_to_pinecone(docs, config):
    # Create empty index
    PINECONE_KEY, PINECONE_INDEX_NAME = os.getenv(
        'PINECONE_API_KEY'), os.getenv('PINECONE_INDEX_NAME')

    pc = Pinecone(api_key=PINECONE_KEY)

    # Delete the old index if it exists
    try:
        pc.delete_index(PINECONE_INDEX_NAME)
    except Exception as e:
        print(f"Error deleting index: {e}")

    # Create new index with correct dimensions
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # Match text-embedding-3-small dimensions
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ))

    # Target index and check status
    pc_index = pc.Index(PINECONE_INDEX_NAME)
    index_stats = pc_index.describe_index_stats()
    print(f"Pinecone index stats: {index_stats}")

    embeddings = OpenAIEmbeddings(model=config['EMBEDDING_MODEL_NAME'])
    namespace = "film_search_prod"

    try:
        pc_index.delete(namespace=namespace, delete_all=True)
    except NotFoundException:
        print(f"Namespace '{namespace}' not found. Not deleting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    else:
        print("Namespace deleted successfully.")

    PineconeVectorStore.from_documents(
        docs,
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace
    )
    print("Successfully uploaded documents to Pinecone")

    print("Successfully uploaded docs to Pinecone vector store")


@task
def publish_dataset_to_weave(docs):
    # Temporarily disabled due to Weave compatibility issues
    print("Weave integration temporarily disabled")
    pass


@flow(log_prints=True)
def pinecone_flow():
    with open('./config.json') as f:
        config = json.load(f)

    start()
    pull_data_to_csv(config)
    docs = convert_csv_to_docs()
    upload_docs_to_pinecone(docs, config)
    publish_dataset_to_weave(docs)


if __name__ == "__main__":
    # pinecone_flow.deploy(
    #     name="pinecone-flow-deployment",
    #     work_pool_name="my-aci-pool",
    #     cron="0 0 * * 0",
    #     image=DeploymentImage(
    #         name="prefect-flows:latest",
    #         platform="linux/amd64",
    #     )
    # )

    # For testing purposes
    pinecone_flow()
