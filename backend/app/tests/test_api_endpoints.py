from starlette.testclient import TestClient
from backend.app.main import app

def test_api_routes():
    with TestClient(app) as client:
        # 1. Test Root serves frontend index.html
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Automaton AI" in resp.text

        # 2. Test Health
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["chunks_count"] > 20

        # 3. Test Knowledge Summary
        resp = client.get("/api/knowledge/summary")
        assert resp.status_code == 200
        k_data = resp.json()
        assert "ADVIT Studio" in str(k_data["products"])
        assert len(k_data["datasets"]) > 0

        # 4. Test Chat Endpoint (Normal ADVIT query)
        resp = client.post("/api/chat", json={"message": "What is ADVIT Studio V2?"})
        assert resp.status_code == 200
        chat_data = resp.json()
        assert len(chat_data["citations"]) > 0
        assert "advit" in chat_data["response"].lower()

        # 5. Test Chat Endpoint (Pricing Guardrail)
        resp = client.post("/api/chat", json={"message": "What is the cost of ADVIT Studio?"})
        assert resp.status_code == 200
        pricing_data = resp.json()
        assert "not publicly published" in pricing_data["response"]
        assert pricing_data["guardrail_triggered"] == "pricing_non_disclosure"

        # 6. Test Search Endpoint
        resp = client.post("/api/search", json={"query": "pomegranate disease", "top_k": 3})
        assert resp.status_code == 200
        search_data = resp.json()
        assert len(search_data["results"]) > 0
        assert "pomegranate" in search_data["results"][0]["content"].lower()

        # 7. Test Eval Endpoint
        resp = client.post("/api/eval")
        assert resp.status_code == 200
        eval_data = resp.json()
        assert eval_data["accuracy_percentage"] == 100.0

    print("ALL FASTAPI ENDPOINT TESTS PASSED WITH 100% SUCCESS!")

if __name__ == "__main__":
    test_api_routes()
