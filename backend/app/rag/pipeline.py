import logging
from typing import List, Dict, Any, Optional
from backend.app.config import settings
from backend.app.schemas import ChatResponse, Citation, ChatMessage
from backend.app.rag.document_store import DocumentStore, DocumentChunk
from backend.app.rag.retriever import HybridRetriever
from backend.app.rag.guardrails import GuardrailEngine
from backend.app.rag.generator import ResponseGenerator

logger = logging.getLogger("rag_pipeline")

class RAGPipeline:
    """
    Core RAG Pipeline for Automaton AI Chatbot.
    Enforces strict grounding, hybrid retrieval, guardrails, and citation verification.
    """
    def __init__(self):
        self.doc_store = DocumentStore()
        self.retriever = HybridRetriever(self.doc_store)
        self.generator = ResponseGenerator()

    async def process_chat(
        self, message: str, history: Optional[List[ChatMessage]] = None
    ) -> ChatResponse:
        query = message.strip()
        if not query:
            return ChatResponse(
                response="Please provide a question or topic regarding Automaton AI.",
                citations=[],
                sources=[],
                confidence=0.0,
                suggested_followups=["What is ADVIT Studio?", "What datasets are in Data Zoo?", "Where is Automaton AI based?"],
                is_fallback=True
            )

        # 1. Pre-Retrieval Intent Guardrail Check
        guardrail_result = GuardrailEngine.pre_retrieval_check(query)
        if guardrail_result:
            logger.info(f"Pre-retrieval guardrail triggered: {guardrail_result.get('rule')}")
            citations = [Citation(**c) for c in guardrail_result.get("citations", [])]
            return ChatResponse(
                response=guardrail_result["response"],
                citations=citations,
                sources=guardrail_result.get("sources", []),
                confidence=guardrail_result.get("confidence", 1.0),
                suggested_followups=guardrail_result.get("suggested_followups", []),
                is_fallback=guardrail_result.get("is_fallback", False),
                guardrail_triggered=guardrail_result.get("rule")
            )

        # 2. Hybrid Retrieval (BM25 + TF-IDF + Keyword Booster)
        retrieved = self.retriever.search(query, top_k=settings.MAX_RETRIEVED_CHUNKS)

        if not retrieved or retrieved[0][1] < settings.CONFIDENCE_THRESHOLD:
            # Low confidence fallback
            logger.warning(f"Low retrieval confidence for query '{query}'. Top score: {retrieved[0][1] if retrieved else 0.0}")
            return ChatResponse(
                response=(
                    "I could not locate verified details regarding that specific question in Automaton AI's official documentation.\n\n"
                    "Automaton AI focuses on enterprise DLOps/LLMOps platforms (ADVIT Studio), automated machine learning (ADAPT AI), "
                    "managed data labeling, and computer vision solutions.\n\n"
                    f"To discuss your specific query with an Automaton AI specialist, please use the [Connect Form]({settings.CONNECT_URL}) "
                    f"or reach out directly at **{settings.INFO_EMAIL}**."
                ),
                citations=[],
                sources=[settings.CONNECT_URL],
                confidence=retrieved[0][1] if retrieved else 0.0,
                suggested_followups=[
                    "What is ADVIT Studio?",
                    "What is the process for Custom AI Applications?",
                    "Tell me about the Model Gallery"
                ],
                is_fallback=True,
                guardrail_triggered="low_confidence_fallback"
            )

        # Extract top chunks and confidence
        top_chunks = [item[0] for item in retrieved]
        top_confidence = retrieved[0][1]

        # 3. Asynchronous Generation
        hist_dicts = [{"role": h.role, "content": h.content} for h in (history or [])]
        raw_response = await self.generator.generate(query, top_chunks, hist_dicts)

        # 4. Post-Generation Guardrail Check
        final_response, guardrail_warning = GuardrailEngine.post_generation_check(raw_response, top_chunks)

        # 5. Build Structured Citations & Suggested Followups
        citations: List[Citation] = []
        sources_set = set()
        followups_set = set()

        for chunk in top_chunks[:3]:
            if chunk.url:
                sources_set.add(chunk.url)
            snippet = chunk.content.split("\n")[0][:180] + "..." if len(chunk.content) > 180 else chunk.content
            citations.append(Citation(
                id=chunk.id,
                title=chunk.title,
                url=chunk.url,
                category=chunk.category,
                snippet=snippet
            ))
            for q in chunk.suggested_questions:
                if q.lower() != query.lower():
                    followups_set.add(q)

        suggested_followups = list(followups_set)[:3]
        if not suggested_followups:
            suggested_followups = [
                "What is ADVIT Studio V2?",
                "How do I request a demo?",
                "Explore Data Zoo datasets"
            ]

        return ChatResponse(
            response=final_response,
            citations=citations,
            sources=list(sources_set),
            confidence=top_confidence,
            suggested_followups=suggested_followups,
            is_fallback=False,
            guardrail_triggered=guardrail_warning
        )
