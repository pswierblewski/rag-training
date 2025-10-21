from pydantic import BaseModel, Field
from typing import List


class SentenceResponse(BaseModel):
    """Response model for chat endpoint containing sentence IDs and sentences."""
    sentence_ids: List[int] = Field(description="List of sentence IDs")
    sentences: List[str] = Field(description="List of sentences")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str = Field(description="Question in Polish language")


class InitResponse(BaseModel):
    """Response model for init endpoint."""
    message: str
    sentences_indexed: int


class LLMResponse(BaseModel):
    """Pydantic model for LLM structured output."""
    sentence_ids: List[int] = Field(description="List of sentence IDs that answer the question")



