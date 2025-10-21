from fastapi import FastAPI, HTTPException
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
import faiss
import os
import logging
from typing import List

import config
import database
from models import ChatRequest, SentenceResponse, InitResponse, LLMResponse

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Training API",
    description="FastAPI RAG solution with LlamaIndex and FAISS for Polish sentence retrieval",
    version="1.0.0"
)

logger.info("FastAPI application initialized")
logger.info(f"LLM Model: {config.LLM_MODEL}")
logger.info(f"Embedding Model: {config.EMBEDDING_MODEL}")

# Configure LlamaIndex settings
logger.debug("Configuring LlamaIndex settings...")
Settings.llm = OpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY)
Settings.embed_model = VoyageEmbedding(model_name=config.EMBEDDING_MODEL, voyage_api_key=config.VOYAGE_API_KEY)
logger.info("LlamaIndex settings configured successfully")


@app.get("/")
async def root():
    """Root endpoint."""
    logger.info("GET / - Root endpoint accessed")
    return {
        "message": "RAG Training API",
        "endpoints": {
            "init": "POST /init - Initialize FAISS vector store",
            "chat": "POST /chat - Query the RAG system"
        }
    }


@app.post("/init", response_model=InitResponse)
async def init_vector_store():
    """
    Initialize FAISS vector store from Polish sentences in the database.
    
    This endpoint:
    1. Fetches all Polish sentences from PostgreSQL
    2. Creates LlamaIndex documents with metadata
    3. Builds FAISS vector store with OpenAI embeddings
    4. Persists the index to local disk
    
    Returns:
        InitResponse with success message and count of indexed sentences
    """
    logger.info("POST /init - Starting FAISS vector store initialization")
    try:
        # Test database connection
        logger.debug("Testing database connection...")
        if not database.test_connection():
            logger.error("Database connection test failed")
            raise HTTPException(status_code=500, detail="Database connection failed")
        logger.debug("Database connection successful")
        
        # Fetch all sentences from database
        logger.debug("Fetching all sentences from database...")
        sentences_data = database.fetch_all_sentences()
        logger.info(f"Retrieved {len(sentences_data)} sentences from database")
        
        if not sentences_data:
            logger.error("No sentences found in database")
            raise HTTPException(status_code=404, detail="No sentences found in database")
        
        # Create LlamaIndex documents with metadata
        logger.debug(f"Creating LlamaIndex documents for {len(sentences_data)} sentences...")
        documents = []
        for item in sentences_data:
            doc = Document(
                text=item["sentence"],
                metadata={
                    "sentence_id": item["id"]
                }
            )
            documents.append(doc)
        logger.debug(f"Created {len(documents)} LlamaIndex documents")
        
        # Create FAISS index
        # dimension of voyage-3-large is 1024
        logger.debug("Creating FAISS index...")
        dimension = 1024  # voyage-3-large embedding dimension
        faiss_index = faiss.IndexFlatL2(dimension)
        logger.debug(f"FAISS index created with dimension {dimension}")
        
        # Create vector store with text storage enabled
        logger.debug("Creating vector store with text storage enabled...")
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Build index from documents
        logger.info(f"Building FAISS index from {len(documents)} documents...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )
        logger.info("FAISS index built successfully")
        
        # Create directory if it doesn't exist
        logger.debug(f"Creating FAISS index directory: {config.FAISS_INDEX_PATH}")
        os.makedirs(config.FAISS_INDEX_PATH, exist_ok=True)
        
        # Persist the index
        logger.debug(f"Persisting index to {config.FAISS_INDEX_PATH}...")
        index.storage_context.persist(persist_dir=config.FAISS_INDEX_PATH)
        logger.info(f"FAISS index persisted successfully to {config.FAISS_INDEX_PATH}")
        
        response = InitResponse(
            message="FAISS vector store initialized successfully",
            sentences_indexed=len(sentences_data)
        )
        logger.info(f"Initialization complete: {len(sentences_data)} sentences indexed")
        return response
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error initializing vector store: {str(e)}")


@app.post("/chat", response_model=SentenceResponse)
async def chat(request: ChatRequest):
    """
    Query the RAG system with a Polish question.
    
    This endpoint:
    1. Loads FAISS vector store from disk
    2. Retrieves most relevant sentences
    3. Formats results as markdown table
    4. Queries OpenAI LLM with structured output
    5. Fetches original sentences from database
    6. Returns matching sentences
    
    Args:
        request: ChatRequest with Polish question
        
    Returns:
        SentenceResponse with sentence IDs and sentences
    """
    logger.info(f"POST /chat - Processing question: '{request.question}'")
    try:
        # Check if FAISS index exists
        logger.debug(f"Checking for FAISS index at {config.FAISS_INDEX_PATH}")
        if not os.path.exists(config.FAISS_INDEX_PATH):
            logger.error("FAISS index not found")
            raise HTTPException(
                status_code=404,
                detail="FAISS index not found. Please call /init endpoint first."
            )
        logger.debug("FAISS index found")
        
        # Load the vector store from disk
        logger.debug("Loading FAISS vector store...")
        vector_store = FaissVectorStore.from_persist_dir(config.FAISS_INDEX_PATH)
        
        # Load the index from storage
        logger.debug("Loading index from storage...")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=config.FAISS_INDEX_PATH
        )
        index = load_index_from_storage(storage_context)
        logger.debug("Index loaded successfully")
        
        # Create retriever
        logger.debug(f"Creating retriever with top_k={config.TOP_K_RESULTS}")
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=config.TOP_K_RESULTS
        )
        
        # Retrieve relevant nodes
        logger.info(f"Retrieving top {config.TOP_K_RESULTS} relevant sentences for question...")
        nodes = retriever.retrieve(request.question)
        logger.info(f"Retrieved {len(nodes)} nodes from vector store")
        
        if not nodes:
            logger.info("No relevant nodes found for question")
            return SentenceResponse(sentence_ids=[], sentences=[])
        
        # Format results as markdown table
        logger.debug("Formatting retrieved nodes as markdown table...")
        markdown_table = "| Sentence ID | Sentence |\n|-------------|----------|\n"
        for node in nodes:
            sentence_id = node.metadata.get("sentence_id", "N/A")
            sentence_text = node.text
            markdown_table += f"| {sentence_id} | {sentence_text} |\n"
            logger.debug(f"  Node: ID={sentence_id}, similarity_score={node.score:.4f}")
        
        # Create prompt for LLM
        logger.debug("Creating prompt for LLM...")
        prompt = f"""You are a helpful assistant that can answer questions about Polish sentences.
You are given a question in Polish language.
You are given a list of relevant sentences with their ids.

Question: Is there any sentence from the list that is answering the question: {request.question}?

Context (relevant sentences):
{markdown_table}

If there is, return sentence_ids of the sentences that are answering the question.
Return only the sentence IDs that directly answer or are relevant to the question.
If no sentences answer the question, return an empty list."""
        
        # Query LLM with structured output
        logger.info(f"Querying LLM ({config.LLM_MODEL}) with structured output...")
        llm = OpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY)
        
        # Use structured prediction to get Pydantic model
        structured_llm = llm.as_structured_llm(output_cls=LLMResponse)
        llm_response = structured_llm.complete(prompt)
        logger.debug("LLM response received")
        
        # Extract sentence IDs from LLM response
        llm_result = llm_response.raw
        logger.info(f"LLM returned {len(llm_result.sentence_ids)} sentence IDs: {llm_result.sentence_ids}")
        
        if not llm_result.sentence_ids:
            logger.info("No matching sentences found by LLM")
            return SentenceResponse(sentence_ids=[], sentences=[])
        
        # Fetch original sentences from database by IDs
        logger.debug(f"Fetching sentences from database for IDs: {llm_result.sentence_ids}")
        sentences_data = database.fetch_sentences_by_ids(llm_result.sentence_ids)
        logger.debug(f"Retrieved {len(sentences_data)} sentences from database")
        
        # Prepare response
        sentence_ids = [item["id"] for item in sentences_data]
        sentences = [item["sentence"] for item in sentences_data]
        
        logger.info(f"Returning {len(sentences)} matching sentences")
        return SentenceResponse(
            sentence_ids=sentence_ids,
            sentences=sentences
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("GET /health - Health check requested")
    db_status = database.test_connection()
    faiss_exists = os.path.exists(config.FAISS_INDEX_PATH)
    
    health_data = {
        "status": "healthy" if db_status else "unhealthy",
        "database": "connected" if db_status else "disconnected",
        "faiss_index": "initialized" if faiss_exists else "not_initialized"
    }
    logger.info(f"Health check result: {health_data['status']}")
    return health_data



