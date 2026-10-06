import asyncio
from backend.app.rag.document_store import DocumentStore
from backend.app.rag.retriever import HybridRetriever
from backend.app.rag.guardrails import GuardrailEngine
from backend.app.rag.pipeline import RAGPipeline

def test_document_store_loading():
    store = DocumentStore()
    assert len(store.chunks) > 20, "Knowledge base should yield more than 20 structured chunks"
    
    # Check essential IDs exist
    chunk_ids = [c.id for c in store.chunks]
    assert "company-overview" in chunk_ids
    assert "contact-info" in chunk_ids
    assert "company-leadership" in chunk_ids
    assert "prod-advit-studio" in chunk_ids
    assert "prod-tenderai" in chunk_ids
    assert "data-zoo-catalog" in chunk_ids
    assert "model-gallery-catalog" in chunk_ids

def test_retriever_bm25_and_boost():
    store = DocumentStore()
    retriever = HybridRetriever(store)
    
    # Test ADVIT query
    results = retriever.search("What is ADVIT Studio V2?", top_k=3)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert "advit" in top_chunk.id
    assert score > 0.4

    # Test Data Zoo dataset query
    results = retriever.search("Where can I find annotated cotton data or corn leaf datasets?", top_k=3)
    assert any("data-zoo" in chunk.id for chunk, _ in results)

    # Test Apollo Tyres case study query
    results = retriever.search("Tell me about Apollo Tyres project", top_k=3)
    assert any("clients" in chunk.id or "case_studies" in chunk.category for chunk, _ in results)

def test_guardrails_pricing():
    # Pre-retrieval pricing check
    res = GuardrailEngine.pre_retrieval_check("What is the price of an ADVIT Studio enterprise license?")
    assert res is not None
    assert res["rule"] == "pricing_non_disclosure"
    assert "not publicly published" in res["response"]
    assert "connect" in res["response"].lower()

def test_guardrails_out_of_scope():
    res = GuardrailEngine.pre_retrieval_check("Who won the football world cup?")
    assert res is not None
    assert res["rule"] == "out_of_scope"
    assert res["is_fallback"] is True

def test_guardrails_tenderai():
    res = GuardrailEngine.pre_retrieval_check("Tell me all features and specs of TenderAI")
    assert res is not None
    assert res["rule"] == "tenderai_limited_info"
    assert "not publicly available" in res["response"] or "not published" in res["response"]

def test_guardrails_careers():
    res = GuardrailEngine.pre_retrieval_check("I want to apply for a job as a machine learning engineer")
    assert res is not None
    assert res["rule"] == "careers_routing"
    assert "careers@automatonai.com" in res["response"]

def test_pipeline_execution():
    async def _run():
        pipeline = RAGPipeline()
        resp = await pipeline.process_chat("What is ADVIT Studio?")
        assert resp.response is not None
        assert len(resp.citations) > 0
        assert resp.confidence > 0.3
        assert resp.is_fallback is False
    asyncio.run(_run())

if __name__ == "__main__":
    print("Running unit tests manually...")
    test_document_store_loading()
    test_retriever_bm25_and_boost()
    test_guardrails_pricing()
    test_guardrails_out_of_scope()
    test_guardrails_tenderai()
    test_guardrails_careers()
    test_pipeline_execution()
    print("ALL UNIT TESTS PASSED SUCCESSFULLY!")
