from pydantic import BaseModel
from typing import List

class DocumentInput(BaseModel):
    text: str

class SummaryResponse(BaseModel):
    summary: str
    total_chunks: int

class QueryInput(BaseModel):
    question: str
    document_text: str

class QueryResponse(BaseModel):
    answer: str
    relevant_chunks: List[str]