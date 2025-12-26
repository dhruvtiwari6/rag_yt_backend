from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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



from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
origins = [
    "ragytbackend-production-37b6.up.railway.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorStore : FAISS | None = None
history = []


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "FastAPI backend is running 🚀"
    }



@app.post("/ingest")
def ingest_video(req : IngestRequest):
    global vectorStore
    global history

    try:
        print("req" , req)
        # get transcript
        transcript = get_transcript(req.video_id)

        # split into langchain documnets
        docs = chunk_transcript(transcript)

        from langchain_community.vectorstores import FAISS
        embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")


        vectorStore = FAISS.from_documents(
            documents = docs,
            embedding = embedding_model
        )

        return {
            "status" : "success",
            "video_id" : req.video_id,
            "chunk_ingested" : len(docs)
        }
    
    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))



@app.post("/query")
def query_video(req : QueryRequest):
    if vectorStore is None:
        raise HTTPException(status_code = 400 , detail = "No videos ingested yet")
    

    message = HumanMessage(content = req.query)
    history.append(message)

    print(req.query)
    
    # create retriever
    retriever = vectorStore.as_retriever(
        search_type= "mmr",
        search_kwargs = {"k" : req.k, "lambda_mult": 0.5}
    )


    def format_docs(retrieved_docs) :
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context
    


    parallel_chain = RunnableParallel({
        'context' : retriever |  RunnableLambda(format_docs),
        'question' : RunnablePassthrough(),
        "history": RunnableLambda(lambda _: history),

    })

    parser = StrOutputParser()

    llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , temperature=0.2)
    # prompt = PromptTemplate(
    #     template = """
    #             You are a helpful assistant.
    #             Answer only from the provided transcript context.
    #             if the context is insufficient , just say you don't know

    #             {context}
    #             Question : {question}
    #     """,
    #     input_variables=    ['context' , 'question']
    # )



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

    mainChain = parallel_chain | prompt | llm | parser
    ans = mainChain.invoke(req.query)

    history.append(AIMessage(content = ans))

    print(history)
    

    return {
        "status" : "success",
        "query": req.query,
        "context": ans
    }



        






