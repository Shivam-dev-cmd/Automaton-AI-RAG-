import asyncio
import time
from typing import List, Dict, Any
from backend.app.rag.pipeline import RAGPipeline

EVAL_TEST_CASES = [
    {
        "id": "TC-01",
        "name": "Pricing Non-Disclosure Guardrail",
        "query": "How much does ADVIT Studio cost per month for an enterprise subscription?",
        "check": lambda resp: (
            "not publicly published" in resp.response.lower()
            and any(c in resp.response.lower() for c in ["connect", "info@automatonai.com"])
            and not any(currency in resp.response for currency in ["$99", "$499", "₹", "Rs."])
        ),
        "description": "Verifies that unlisted pricing is strictly protected from hallucination and routes to official contact."
    },
    {
        "id": "TC-02",
        "name": "TenderAI 404 & Unverified Product Boundary",
        "query": "What are all the technical features and modules of TenderAI?",
        "check": lambda resp: (
            ("not publicly available" in resp.response.lower() or "not published" in resp.response.lower() or "footer" in resp.response.lower())
            and not any(hallucination in resp.response.lower() for hallucination in ["module 1", "module 2", "pricing tier"])
        ),
        "description": "Ensures no fabricated specs are returned for TenderAI (which only appeared in the footer and 404'd)."
    },
    {
        "id": "TC-03",
        "name": "ADVIT Studio Platform Capabilities",
        "query": "Does ADVIT Studio support edge deployment and on-premise air-gapped installation?",
        "check": lambda resp: (
            any(w in resp.response.lower() for w in ["on-premise", "air-gapped"])
            and any(w in resp.response.lower() for w in ["jetson", "edge"])
        ),
        "description": "Checks retrieval and faithfulness for core ADVIT Studio deployment capabilities."
    },
    {
        "id": "TC-04",
        "name": "DocuGPT Verification & Accuracy Metric Attribution",
        "query": "What is DocuGPT used for and what accuracy does it achieve?",
        "check": lambda resp: (
            "99.2%" in resp.response
            and any(w in resp.response.lower() for w in ["document", "rfp", "kyc", "banking"])
        ),
        "description": "Checks DocuGPT's 99.2% extraction accuracy claim and banking use case."
    },
    {
        "id": "TC-05",
        "name": "Client Case Studies Accuracy (Apollo Tyres)",
        "query": "What work did Automaton AI do for Apollo Tyres?",
        "check": lambda resp: (
            "apollo tyres" in resp.response.lower()
            and any(w in resp.response.lower() for w in ["defect", "production", "edge", "retraining"])
        ),
        "description": "Validates exact retrieval and answer formulation for Apollo Tyres manufacturing case study."
    },
    {
        "id": "TC-06",
        "name": "Data Zoo Dataset Formats",
        "query": "What export formats are available for datasets in Data Zoo?",
        "check": lambda resp: (
            any(f in resp.response.upper() for f in ["COCO", "YOLO", "VOC", "CSV"])
        ),
        "description": "Validates precision of technical dataset export formats."
    },
    {
        "id": "TC-07",
        "name": "Out-of-Scope Rejection",
        "query": "Can you give me a recipe for chocolate cake and cookies?",
        "check": lambda resp: (
            resp.is_fallback is True
            and "official automaton ai virtual assistant" in resp.response.lower()
        ),
        "description": "Ensures the chatbot politely refuses unrelated out-of-scope requests."
    },
    {
        "id": "TC-08",
        "name": "Careers & Job Application Routing",
        "query": "How do I apply for the GenAI Engineer opening at Automaton AI?",
        "check": lambda resp: (
            "careers@automatonai.com" in resp.response.lower()
            or "grow-with-us" in resp.response.lower()
        ),
        "description": "Checks recruitment and career routing to official portals."
    },
    {
        "id": "TC-09",
        "name": "Office Location & Inconsistency Handling",
        "query": "Where is Automaton AI headquartered?",
        "check": lambda resp: (
            "hinjewadi" in resp.response.lower()
            and "pune" in resp.response.lower()
        ),
        "description": "Checks correct HQ address and geographic grounding in Hinjewadi, Pune."
    },
    {
        "id": "TC-10",
        "name": "Model Gallery Verification",
        "query": "Do you have any models for agricultural disease or pomegranate?",
        "check": lambda resp: (
            "pomegranate" in resp.response.lower()
            and "disease" in resp.response.lower()
        ),
        "description": "Ensures exact match from the pre-built model gallery catalog."
    }
]

async def run_evaluations(pipeline: RAGPipeline = None) -> Dict[str, Any]:
    if pipeline is None:
        pipeline = RAGPipeline()

    start_time = time.time()
    results = []
    passed_count = 0

    for tc in EVAL_TEST_CASES:
        t0 = time.time()
        chat_resp = await pipeline.process_chat(tc["query"])
        latency_ms = round((time.time() - t0) * 1000, 2)
        
        passed = False
        try:
            passed = bool(tc["check"](chat_resp))
        except Exception as e:
            passed = False

        if passed:
            passed_count += 1

        results.append({
            "test_id": tc["id"],
            "name": tc["name"],
            "query": tc["query"],
            "passed": passed,
            "latency_ms": latency_ms,
            "confidence": chat_resp.confidence,
            "citations_count": len(chat_resp.citations),
            "guardrail_triggered": chat_resp.guardrail_triggered,
            "description": tc["description"],
            "response_snippet": chat_resp.response[:200] + "..." if len(chat_resp.response) > 200 else chat_resp.response
        })

    total_time = round(time.time() - start_time, 2)
    accuracy_pct = round((passed_count / len(EVAL_TEST_CASES)) * 100, 1)

    return {
        "status": "passed" if passed_count == len(EVAL_TEST_CASES) else "partial",
        "total_tests": len(EVAL_TEST_CASES),
        "passed": passed_count,
        "failed": len(EVAL_TEST_CASES) - passed_count,
        "accuracy_percentage": accuracy_pct,
        "total_eval_time_seconds": total_time,
        "details": results
    }

if __name__ == "__main__":
    async def main():
        print("Starting Systematic RAG Evaluation Suite...")
        summary = await run_evaluations()
        print(f"\n================ EVALUATION SUMMARY ================")
        print(f"Total Tests : {summary['total_tests']}")
        print(f"Passed      : {summary['passed']}")
        print(f"Failed      : {summary['failed']}")
        print(f"Accuracy    : {summary['accuracy_percentage']}%")
        print(f"Total Time  : {summary['total_eval_time_seconds']}s")
        print(f"===================================================\n")
        for res in summary["details"]:
            status_symbol = "[PASS]" if res["passed"] else "[FAIL]"
            print(f"{status_symbol} {res['test_id']} - {res['name']} ({res['latency_ms']}ms)")
            if not res["passed"]:
                print(f"   Snippet: {res['response_snippet']}")

    asyncio.run(main())
