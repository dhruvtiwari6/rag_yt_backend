from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.messages import AIMessage, HumanMessage

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from app.schemas import IngestRequest, QueryRequest
from app.services.chunker import chunk_transcript
from app.services.transcript import get_transcript

import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://rag-yt-frontend.vercel.app", "https://askyt.dhruv-tiwari.me"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


vectorStore : FAISS | None = None
history = []


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "FastAPI backend is running 🚀"
    }


# @app.options("/ingest")
# def ingest_options():
#     return Response(status_code=200)

@app.post("/ingest")
def ingest_video(req : IngestRequest):
    global vectorStore
    global history

    try:
        # Validate video_id
        if not req.video_id or not req.video_id.strip():
            raise HTTPException(status_code=400, detail="Video ID cannot be empty")
        
        logger.info(f"Processing ingest request for video_id: {req.video_id}")
        
        # Check for required environment variables
        if not os.getenv("NVIDIA_API_KEY"):
            raise HTTPException(
                status_code=500, 
                detail="NVIDIA API key not configured. Please set NVIDIA_API_KEY environment variable."
            )
        
        # Get transcript
        try:
            transcript = get_transcript(req.video_id)
            if not transcript or not transcript.strip():
                raise HTTPException(
                    status_code=404, 
                    detail=f"No transcript found for video ID: {req.video_id}"
                )
        except RuntimeError as e:
            logger.error(f"Transcript fetch error: {str(e)}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error fetching transcript: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch transcript: {str(e)}")

        # Split into langchain documents
        try:
            docs = chunk_transcript(transcript)
            if not docs or len(docs) == 0:
                raise HTTPException(
                    status_code=500, 
                    detail="Failed to chunk transcript. Transcript may be empty or invalid."
                )
        except Exception as e:
            logger.error(f"Chunking error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to chunk transcript: {str(e)}")

        # Create embedding model and vector store
        try:
            embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
            vectorStore = FAISS.from_documents(
                documents=docs,
                embedding=embedding_model
            )
        except Exception as e:
            logger.error(f"Vector store creation error: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to create vector store. Please check your NVIDIA API key and connection: {str(e)}"
            )

        # Reset history for new video
        history = []

        logger.info(f"Successfully ingested {len(docs)} chunks for video_id: {req.video_id}")
        return {
            "status": "success",
            "video_id": req.video_id,
            "chunk_ingested": len(docs)
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ingest endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")





@app.post("/query")
def query_video(req : QueryRequest):
    global vectorStore
    global history
    
    try:
        # Validate vector store exists
        if vectorStore is None:
            raise HTTPException(
                status_code=400, 
                detail="No videos ingested yet. Please ingest a video first using the /ingest endpoint."
            )
        
        # Validate query
        if not req.query or not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Validate k parameter
        if req.k <= 0 or req.k > 50:
            raise HTTPException(
                status_code=400, 
                detail="Parameter 'k' must be between 1 and 50"
            )
        
        logger.info(f"Processing query: {req.query[:50]}...")
        
        # Check for required environment variables
        if not os.getenv("GOOGLE_API_KEY"):
            raise HTTPException(
                status_code=500, 
                detail="Google API key not configured. Please set GOOGLE_API_KEY environment variable."
            )
        
        # Add user message to history
        try:
            message = HumanMessage(content=req.query)
            history.append(message)
        except Exception as e:
            logger.error(f"Error adding message to history: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")
        
        # Create retriever
        try:
            retriever = vectorStore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": req.k, "lambda_mult": 0.5}
            )
        except Exception as e:
            logger.error(f"Error creating retriever: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to create retriever. Vector store may be corrupted: {str(e)}"
            )

        def format_docs(retrieved_docs):
            if not retrieved_docs:
                logger.warning("No documents retrieved")
                return "No relevant context found."
            try:
                context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                return context
            except Exception as e:
                logger.error(f"Error formatting documents: {str(e)}")
                # Re-raise to be caught by outer exception handler
                raise

        # Create parallel chain
        try:
            parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough(),
                "history": RunnableLambda(lambda _: history),
            })
        except Exception as e:
            logger.error(f"Error creating parallel chain: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to create processing chain: {str(e)}")

        # Create LLM and prompt
        try:
            parser = StrOutputParser()
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", 
                "You are a helpful assistant. "
                "Answer only from the provided transcript context. "
                "If the context is insufficient, say you don't know."
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human",
                "Context:\n{context}\n\n"
                "Question:\n{question}"
                ),
            ])
        except Exception as e:
            logger.error(f"Error initializing LLM or prompt: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize language model. Please check your Google API key: {str(e)}"
            )

        # Invoke chain
        try:
            mainChain = parallel_chain | prompt | llm | parser
            ans = mainChain.invoke(req.query)
            
            if not ans or not ans.strip():
                raise HTTPException(
                    status_code=500, 
                    detail="Received empty response from language model"
                )
        except Exception as e:
            logger.error(f"Error invoking chain: {str(e)}")
            # Remove the message from history if chain invocation failed
            if history and isinstance(history[-1], HumanMessage):
                history.pop()
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to generate response. Please check your API keys and try again: {str(e)}"
            )

        # Add AI response to history
        try:
            history.append(AIMessage(content=ans))
        except Exception as e:
            logger.warning(f"Error adding AI message to history: {str(e)}")
            # Don't fail the request if history update fails

        logger.info(f"Successfully processed query")
        return {
            "status": "success",
            "query": req.query,
            "context": ans
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in query endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



        






