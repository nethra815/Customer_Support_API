import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from support_rag import answer_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VoltKart Customer Support API",
    description=(
        "A RAG-powered customer-support API using a VoltKart FAQ knowledge base, "
        "FAISS retrieval, LangGraph orchestration, and Groq-hosted LLM inference."
    ),
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Customer question about VoltKart support policies.",
        examples=["How do I request a refund?"],
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    relevance_score: float
    response_time_ms: float


@app.get("/", tags=["Health"])
def root() -> dict[str, str]:
    """Basic service information."""
    return {
        "service": "VoltKart Customer Support API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    """Health check for the API process."""
    return {"status": "running"}


@app.post("/ask", response_model=AskResponse, tags=["Customer Support"])
def ask(request: AskRequest) -> AskResponse:
    """Answer a customer-support question using the VoltKart knowledge base."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be blank.")

    started = time.perf_counter()

    try:
        answer, sources, relevance_score = answer_question(question)
    except Exception as exc:
        logger.exception("Failed to answer support question")
        # Do not return API keys, environment variables, or internal exception details.
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question right now. Please try again.",
        ) from exc

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "Answered question in %.2f ms; relevance=%.3f; sources=%d",
        elapsed_ms,
        float(relevance_score),
        len(sources or []),
    )

    return AskResponse(
        question=question,
        answer=str(answer),
        sources=list(sources or []),
        relevance_score=round(float(relevance_score), 4),
        response_time_ms=elapsed_ms,
    )
