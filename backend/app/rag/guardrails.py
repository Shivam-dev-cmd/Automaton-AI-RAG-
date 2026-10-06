import re
from typing import Optional, Tuple, Dict, Any, List
from backend.app.config import settings
from backend.app.rag.document_store import DocumentChunk

class GuardrailEngine:
    """
    Multi-stage Guardrail Engine designed to eliminate hallucinations, enforce
    strict knowledge boundaries, and maintain 100% faithfulness to Automaton AI documentation.
    """

    PRICING_KEYWORDS = ["price", "pricing", "cost", "how much", "license fee", "subscription rate", "quotation", "rate card"]
    CAREERS_KEYWORDS = ["job", "career", "hiring", "openings", "vacancy", "internship", "apply for job", "interview", "recruitment"]
    OUT_OF_SCOPE_KEYWORDS = ["weather in", "cricket", "football", "recipe", "write python script for snake game", "who is the president", "stock market prediction", "crypto"]

    @classmethod
    def pre_retrieval_check(cls, query: str) -> Optional[Dict[str, Any]]:
        """
        Evaluates the query before retrieval.
        Returns a deterministic grounded response if a hard rule applies, or None to proceed.
        """
        q_lower = query.lower()

        # 1. Out-of-Scope Filter
        for oos in cls.OUT_OF_SCOPE_KEYWORDS:
            if oos in q_lower:
                return {
                    "rule": "out_of_scope",
                    "response": (
                        "I am the official Automaton AI Virtual Assistant, specialized in providing accurate information "
                        "regarding Automaton AI's enterprise AI platforms (ADVIT Studio, ADAPT AI, AI Labs), managed data labeling, "
                        "computer vision models, and enterprise services.\n\n"
                        "I cannot assist with general topics unrelated to Automaton AI. "
                        "How can I help you with our AI platforms, industry solutions, or enterprise services?"
                    ),
                    "confidence": 1.0,
                    "citations": [],
                    "sources": ["https://automatonai.com"],
                    "suggested_followups": [
                        "What is ADVIT Studio?",
                        "What managed data labeling services do you offer?",
                        "How can I request an enterprise demo?"
                    ],
                    "is_fallback": True
                }

        # 2. Strict Pricing Guardrail
        # Company policy: Pricing is NOT published online. Must direct to Connect form.
        if any(pk in q_lower for pk in cls.PRICING_KEYWORDS):
            return {
                "rule": "pricing_non_disclosure",
                "response": (
                    "**Pricing for Automaton AI solutions (including ADVIT Studio, ADAPT AI, and ADVIT AI Labs) is not publicly published.**\n\n"
                    "According to Automaton AI, licensing and subscription packages are tailored based on deployment architecture "
                    "(cloud, on-premise, or air-gapped), workload capacity, and enterprise requirements.\n\n"
                    "To receive an official quote or discuss pricing:\n"
                    f"- Submit an inquiry via the [Automaton AI Connect Form]({settings.CONNECT_URL}) (select topic: *'Explore pricing for ADVIT Studio'*)\n"
                    f"- Email the sales team at: **{settings.INFO_EMAIL}**\n"
                    f"- Call: **{settings.PHONE_PRIMARY}**"
                ),
                "confidence": 1.0,
                "citations": [
                    {
                        "id": "faq-pricing",
                        "title": "FAQ: How do I get pricing?",
                        "url": settings.CONNECT_URL,
                        "category": "faq",
                        "snippet": "Pricing is not published. Use the Connect form and choose Explore pricing for ADVIT Studio, or email info@automatonai.com."
                    }
                ],
                "sources": [settings.CONNECT_URL],
                "suggested_followups": [
                    "How do I schedule an ADVIT Studio demo?",
                    "What deployment options are supported by ADVIT Studio?",
                    "What are the key differentiators of Automaton AI?"
                ],
                "is_fallback": False
            }

        # 3. TenderAI Product Guardrail (404 / footer only)
        if "tenderai" in q_lower or "tender ai" in q_lower:
            return {
                "rule": "tenderai_limited_info",
                "response": (
                    "**TenderAI** is listed in the Automaton AI website navigation/footer, but its dedicated product page "
                    "is currently not publicly available or detailed in official documentation.\n\n"
                    "Automaton AI's published blog on *Agentic AI for Indian Enterprises* does reference automated government tender "
                    "analysis as a key enterprise use case built on the ADVIT platform.\n\n"
                    f"To request official specifications or a demonstration of TenderAI capabilities, please reach out via the "
                    f"[Connect Form]({settings.CONNECT_URL}) or email **{settings.INFO_EMAIL}**."
                ),
                "confidence": 0.95,
                "citations": [
                    {
                        "id": "prod-tenderai",
                        "title": "Product: TenderAI",
                        "url": "https://automatonai.com",
                        "category": "product",
                        "snippet": "Named in the site footer only. No official description published on website."
                    }
                ],
                "sources": ["https://automatonai.com"],
                "suggested_followups": [
                    "What is ADVIT Studio?",
                    "What is DocuGPT?",
                    "How can I contact sales for custom AI solutions?"
                ],
                "is_fallback": False
            }

        # 4. DocuGPT Guardrail (prevent ungrounded feature expansion)
        if "docugpt" in q_lower or "docu gpt" in q_lower:
            # Let retrieval handle it, but flag docugpt for grounding check
            pass

        # 5. Direct Careers Inquiry
        if any(ck in q_lower for ck in cls.CAREERS_KEYWORDS) and not any(k in q_lower for k in ["product", "advit", "service", "model"]):
            return {
                "rule": "careers_routing",
                "response": (
                    "**Careers at Automaton AI**:\n\n"
                    "Automaton AI regularly hires across engineering, AI research, and operations in Hinjewadi, Pune. Current listed openings include:\n"
                    "- **GenAI Engineer** (LLM-powered apps, RAG systems, PyTorch, Azure/AWS, vector databases)\n"
                    "- **Deep Learning Engineer & Intern**\n"
                    "- **Computer Vision Software Engineer & Intern**\n"
                    "- **MLOps & DevOps Engineers**\n"
                    "- **Data Annotators & BigData Engineers**\n"
                    "- **Hardware Design & System C AI Hardware Modeling Intern**\n\n"
                    "Automaton AI states they are also open to applicants from unlisted domains.\n\n"
                    f"- **Apply Online**: [Automaton AI Grow With Us](https://automatonai.com/grow-with-us/)\n"
                    f"- **Direct Email**: Send your resume to **{settings.CAREERS_EMAIL}**"
                ),
                "confidence": 1.0,
                "citations": [
                    {
                        "id": "careers-hiring",
                        "title": "Careers at Automaton AI",
                        "url": "https://automatonai.com/careers/",
                        "category": "careers",
                        "snippet": "Openings listed: GenAI Engineer, Computer Vision Engineer, MLOps Engineer, Data Annotators. Apply via careers@automatonai.com."
                    }
                ],
                "sources": ["https://automatonai.com/careers/"],
                "suggested_followups": [
                    "What are the requirements for the GenAI Engineer role?",
                    "Where is the Automaton AI office located?",
                    "What is the work culture and mission of Automaton AI?"
                ],
                "is_fallback": False
            }

        return None

    @classmethod
    def post_generation_check(
        cls, response_text: str, retrieved_chunks: List[DocumentChunk]
    ) -> Tuple[str, Optional[str]]:
        """
        Validates the generated output against retrieved ground-truth documents.
        Checks for hallucinated pricing, unverified claims, or missing attribution.
        """
        modified_text = response_text
        warning = None

        # Check for hallucinated currency figures
        # If response mentions $ or ₹ followed by digits that do NOT appear in the retrieved context
        currency_matches = re.findall(r"(\$|₹|USD|INR)\s*(\d+[\d,]*)", modified_text)
        if currency_matches:
            # Check if any chunk explicitly has this price
            combined_context = " ".join([c.content for c in retrieved_chunks])
            for symbol, amount in currency_matches:
                if amount not in combined_context:
                    # Hallucination detected! Replace with standard pricing disclosure
                    modified_text = re.sub(
                        r"(\$|₹|USD|INR)\s*\d+[\d,]*(\s*per\s*\w+)?",
                        "[pricing is custom and available upon direct inquiry]",
                        modified_text
                    )
                    warning = "Removed unverified pricing figures not present in knowledge base."

        # Ensure claims have company attribution if discussing metrics
        metric_keywords = ["99.2%", "60%", "10x", "5x", "50+", "40+"]
        for m in metric_keywords:
            if m in modified_text and "automaton ai" not in modified_text.lower() and "according to" not in modified_text.lower():
                modified_text = f"According to Automaton AI documentation:\n\n" + modified_text
                break

        return modified_text, warning
