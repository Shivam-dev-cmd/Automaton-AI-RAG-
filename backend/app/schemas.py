from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the sender: 'user' or 'assistant'")
    content: str = Field(..., description="Text content of the message")

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User's query")
    session_id: Optional[str] = Field(default=None, description="Client session identifier")
    history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Recent conversation turns")

class Citation(BaseModel):
    id: str = Field(..., description="Unique ID of the cited section")
    title: str = Field(..., description="Title of the cited section")
    url: Optional[str] = Field(default=None, description="Official Automaton AI URL")
    category: str = Field(..., description="Category (product, service, faq, industry, etc.)")
    snippet: str = Field(..., description="Relevant snippet from knowledge base")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated answer grounded in company knowledge")
    citations: List[Citation] = Field(default_factory=list, description="Referenced official sources")
    sources: List[str] = Field(default_factory=list, description="Distinct source URLs or titles")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Retrieval confidence score")
    suggested_followups: List[str] = Field(default_factory=list, description="Helpful follow-up questions")
    is_fallback: bool = Field(default=False, description="Whether fallback response was triggered")
    guardrail_triggered: Optional[str] = Field(default=None, description="Guardrail rule triggered if applicable")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query string to search")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")

class SearchResult(BaseModel):
    id: str
    title: str
    category: str
    score: float
    content: str
    url: Optional[str] = None
    caveats: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

class HealthResponse(BaseModel):
    status: str
    version: str
    chunks_count: int
    llm_provider: str
    knowledge_base_loaded: bool

class KnowledgeSummaryResponse(BaseModel):
    company_name: str
    tagline: str
    categories: Dict[str, int]
    total_chunks: int
    products: List[str]
    services: List[str]
    datasets: List[str]
    models: List[str]
