# Rosebud 🌹

### Let's discover films.

## Getting API Keys 🔑

To use Rosebud, you'll need API keys for various services. Follow the steps in [this guide](https://scribehow.com/page/API_Key_Acquisition_Steps_For_Multiple_Services__AGK8btnlRziqnZoUGjvRFA) to acquire the necessary keys for:

- **Pinecone**: For vector storage and retrieval.
- **OpenAI**: For the language model powering the chat system.
- **TMDB**: For fetching movie data.
- **Weights & Biases (W&B)**: For evaluation and observability.

Once you have the keys, add them to your `.env` file as described in the Docker Installation section.

## Docker Installation 🐳

The easiest way to run Rosebud is using Docker. You can run the entire application with a single command:

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed on your machine
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)

### Running with Docker Compose

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SahiDemon/Rosebud.git
   cd Rosebud
   ```

2. **Set up your environment variables:**
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file with your API keys for Pinecone, OpenAI, etc.

3. **Start the application:**
   ```bash
   docker-compose up -d
   ```

4. **Access the application:**
   Open your browser and navigate to `http://localhost:8000`

5. **Stop the application:**
   ```bash
   docker-compose down
   ```

### Running with Docker directly

If you prefer to use Docker without Docker Compose:

```bash
# Build the image
docker build -t rosebud .

# Run the container
docker run -p 8000:8000 --env-file .env rosebud
```

### Docker Troubleshooting

If you encounter issues when running the Docker container:

1. **Docker Desktop not running**: Ensure Docker Desktop is running on your system. On Windows, check the system tray for the Docker icon.
   ```bash
   # Verify Docker is running with:
   docker info
   ```

2. **Docker Engine connection errors**: If you see errors like "unable to get image" or "error during connect", restart Docker Desktop.
   
3. **Environment variables not loading**: Make sure your `.env` file is in the same directory as your `docker-compose.yml` file.
   ```bash
   # You can verify environment variables with:
   docker-compose config
   ```

4. **Port already in use**: If port 8000 is already in use, change the port mapping in `docker-compose.yml`:
   ```yaml
   ports:
     - "8001:8000"  # Maps host port 8001 to container port 8000
   ```

5. **Windows path issues**: If using Windows and encountering path-related errors, try using Linux-style paths in Docker volume mappings.

## Overview of Tools Used 🛠️
- **LangChain**: Framework to create the `rosebud_chat_model`. Important in the creation of the self-querying retriever as well as connecting the OpenAI Chat model to the Pinecone vector store for retrieval augmented generation (RAG).
- **Weights and Biases (W&B)/Weave**: Used for evaluation and reproducibility. W&B and Weave two separate tools unified into a single platform for ease of use. Both offline and online evaluations are stored in W&B/Weave. Offline evaluation is performed using RAGAS (see below). Online evaluation is facilitated via the use of 👍 and 👎 buttons attached to the end of each response. See the image below for an example of the observability:
![Image of online feedback](images/wandb.png)
- **Prefect**: A workflow orchestrator that is used to schedule weekly updates to the data used by the Rosebud chat model (see below).
- **RAGAS**: A popular RAG evaluation framework. For this project, RAGAS was used for offline evaluation. A series of 15-20 questions are each passed to the Rosebud chat model, and the responses are parsed by an LLM judge (`gpt-4o-mini`) for evaluation. There are three metrics that are being tested:
    - `AnswerRelevancy()`: Measures how relevant the answer is to the question being asked. A score from 0 to 1.
    - `ContextRelevancy()`: Measures how relevant the retrieved context is to the question being asked. A score from 0 to 1.
    - `Faithfulness()`: Measures how much the response from the summary model adheres to the retrieved context. A score from 0 to 1. 
    
    These three metrics cover the [RAG triad](https://www.trulens.org/trulens_eval/getting_started/core_concepts/rag_triad/). Check out `offline_eval.py` for the full evaluation code.
- **pytest**: Used for testing. Tests primarily check to make sure that the retrieved data fits the correct format. Check out the `tests/` folder for this code.
- **Pinecone**: The vector store used to hold the documents describing each film. The fact that Pinecone allows for filtering of films via metadata is critical for this app.
- **Streamlit**: Used to create the front-end for the site. Checkout `streamlit_app.py` for this code. 


## Uploading Docs to Pinecone
There are four steps necessary to take data from The Movie Database (TMDB) API and upload them to the Pinecone vector store. Code can be found in the `pinecone_flow.py` file.:

![Flow to upload docs to Pinecone](images/pinecone_flow.png)

1) **pull_data_to_csv**: Programatically pulls roughly 100 of the top films in each year, from 1950 to today, and creates csv files for each year. Makes use of the [TMDB API](https://developer.themoviedb.org/reference/intro/getting-started). This code pulls the following attributes from each film:

    - **Actors**: e.g. ['Christine Taylor', 'Ben Stiller', ...]
    - **Buy**: e.g. ['Apple TV', 'Amazon Video', ...]
    - **Directors**: e.g. 'Ben Stiller'
    - **Genre**: e.g. 'Comedy'
    - **Keywords**: e.g. 'funny, model, ...'
    - **Language**': e.g. 'English'
    - **Overview**: e.g. 'Film about a male model...'
    - **Production** Companies: e.g. ['Paramount Pictures', 'Village Roadshow Pictures', ...]
    - **Rating**: e.g. 6.2
    - **Release Year**: e.g. 2001
    - **Rent**: e.g. ['Apple TV', 'Amazon Video']
    - **Runtime (minutes)**: e.g. 90
    - **Stream**: e.g. ['Paramount Plus', ...]
    - **Title**: e.g. 'Zoolander'
2) **convert_csv_to_docs**: Uses LangChain to take all of the csv files corresponding to each year and creates [Documents](https://js.langchain.com/v0.1/docs/modules/chains/document/) for each film. Each document has two fields: **page_content** and **metadata**:
    - page_content: The primary content that the LLM will see for each document. In this project, the page_content contains the movie's `title`, `overview`, and `keywords`. When the RAG app performs similarity search between the user query and the documents in the database, it does so over this text.
    - metadata: Attached to each document, this field stores all of the attributes that can be used to filter out documents before similarity search is done. These fields are: `Actors`, `Buy`, `Directors`, `Genre`, `Keywords`, `Language`, `Production`, `Rating`, `Release Year`, `Rent`, `Runtime (minutes)`, `Stream`, and `Title`. 
2) **upload_docs_to_pinecone**: The docs are then embedded using the `text-embedding-3-small` model from OpenAI. The embeddings are then uploaded to the Pinecone vector database programatically.
4) **publish_dataset_to_weave**: Finally, we publish the documents to the Weave platform from Weights & Biases for reproducibility.

## Building the Self-Querying Retriever

The `rosebud_chat_model` class contains six methods:
- `initialize_query_constructor`: Creates the query constructor prompt. This chat model is capable of *self-querying retrieval*. This means that the user's query will be used to filter out documents if necessary. The query constructor prompt dictates what sorts of metadata filtering is possible. It also contains a variety of few-shot examples to help guide the model's behavior. Because we use the Pinecone vector store, the following **comparators** are allowed:
    - `$eq`: Equal to (number, string, boolean)
    - `$ne`: Not equal to (number, string, boolean)
    - `$gt`: Greater than (number)
    - `$gte`: Greater than or equal to (number)
    - `$lt`: Less than (number)
    - `$lte`: Less than or equal to (number)
    - `$in`: In array (string or number)
    - `$nin`: Not in array (string or number)

    The following **operators** are allowed:
    - `AND`
    - `OR`

    Currently, the `gpt-4o-mini` model is used to take the user's query and convert it into simplified query with attached metadata filter. For example, the input `Find me drama movies in English that are less than 2 hours long and feature dogs.` creates the following query: 

    ```
    {
        "query": "drama English dogs", 
        "filter": {
            "operator": "and", 
            "arguments": [
                {
                    "comparator": "eq", "attribute": "Genre", "value": "Drama"
                }, 
                {
                    "comparator": "eq", "attribute": "Language", "value": "English"
                }, 
                    
                {
                    "comparator": "lt", "attribute": "Runtime (minutes)", "value": 120
                }
            ]
        },
    }
    ```
- `initialize_vector_store`: Connects the chat bot to the Pinecone vectorstore containing all of the documents. Recall earlier that we used a Prefect flow to create and push the documents to Pinecone.  
- `initialize_retriever`: Creates the self-querying retriever, which incorporates the query constructor, choice of LLM (`gpt-4o-mini`), and the Pinecone vectorstore. 
- `initialize_chat_model`: Creates the summary model, which uses `gpt-4o-mini` to take in the retrieved film documents from Pinecone and crafts recommendations to answer the user's query. There is a basic template provided here so that the bot creates structured output. 
- `predict_stream`: The method used to stream predictions to the Streamlit front-end. Chunks are streamed from the model one at a time. 
- `predict`: The method used to perform offline evaluation using the RAGAS framework. Inputs and outputs to this function are tracked using Weave. The output here is not streamed, and is performed asynchronously to facilitate fast off-line evaluation.
