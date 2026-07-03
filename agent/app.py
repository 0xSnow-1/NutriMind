import os
import logging

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import uvicorn

from agent.agent import compiled as nutrimind
import agent.db as db

API_KEY = os.getenv("API_KEY", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api = FastAPI(title="NutriMind API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.on_event("startup")
async def startup():
    db.setup_tables()


@api.get("/health")
async def health():
    return {"status": "ok", "service": "NutriMind"}


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "USER_#01"


class ChatResponse(BaseModel):
    response: str
    thread_id: str


def verify_api_key(x_api_key: str = Header(None)):
    if API_KEY and (not x_api_key or x_api_key != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


@api.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, auth=Depends(verify_api_key)):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        result = nutrimind.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )

        for m in reversed(result["messages"]):
            if isinstance(m.content, str) and m.content.strip():
                return ChatResponse(response=m.content, thread_id=request.thread_id)

        return ChatResponse(
            response="No response generated.", thread_id=request.thread_id
        )

    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run("agent.app:api", host="0.0.0.0", port=8000)
