import json
import logging
from typing import List, Dict, Any, Optional
import httpx
from backend.app.config import settings
from backend.app.rag.document_store import DocumentChunk

logger = logging.getLogger("generator")

SYSTEM_PROMPT = """You are the official AI Virtual Assistant for Automaton AI Infosystem Pvt. Ltd. (Pune, India).

YOUR STRICT INSTRUCTIONS & GUARDRAILS:
1. FAITHFULNESS FIRST: Answer strictly and only using the provided 'Automaton AI Knowledge Base Context' below.
2. ZERO HALLUCINATION: Never invent, speculate, or extrapolate facts, product features, pricing, or partner details that are not in the context.
3. PRICING RULE: Automaton AI does NOT publish pricing online. Never provide numbers for pricing. Direct users to the Connect Form (https://automatonai.com/connect/) or info@automatonai.com.
4. CLAIM ATTRIBUTION: Always attribute marketing metrics, speedups, and customer claims to the company (e.g., 'Automaton AI states...', 'According to company documentation...').
5. STATUS & CAVEATS: If asked about TenderAI or DocuGPT, note their exact published status honestly without fabricating specs.
6. ADDRESS NOTE: When asked for the office location, mention Hinjewadi, Pune (Suratwala Mark Plazzo).
7. OUT-OF-SCOPE / UNKNOWN: If the answer is not present in the context, politely state: 'I do not have verified information on that in Automaton AI's official documentation' and provide info@automatonai.com or the Connect Form.
8. TONE: Professional, crisp, innovative, and accurate. Format with markdown headings and bullet points where helpful.
"""

class ResponseGenerator:
    """
    Asynchronous LLM generation engine supporting:
    - Built-in Deterministic Grounded Synthesizer (Zero-hallucination baseline)
    - Google Gemini REST API (gemini-1.5-flash)
    - OpenAI / OpenAI-compatible REST API (Groq, Ollama, OpenAI)
    """

    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        logger.info(f"Initialized ResponseGenerator with provider: '{self.provider}'")

    async def generate(
        self, query: str, context_chunks: List[DocumentChunk], history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Dispatches generation to the configured provider."""
        if not context_chunks:
            return (
                "I could not locate verified information regarding that inquiry in Automaton AI's official documentation.\n\n"
                f"For personalized assistance, you can submit an inquiry via the [Connect Form]({settings.CONNECT_URL}) "
                f"or email the team directly at **{settings.INFO_EMAIL}**."
            )

        if self.provider == "gemini" and settings.GEMINI_API_KEY:
            try:
                return await self._generate_gemini(query, context_chunks, history)
            except Exception as e:
                logger.error(f"Gemini API error, falling back to deterministic synthesizer: {e}")
                return self._generate_deterministic(query, context_chunks)

        elif self.provider == "openai" and settings.OPENAI_API_KEY:
            try:
                return await self._generate_openai(query, context_chunks, history)
            except Exception as e:
                logger.error(f"OpenAI API error, falling back to deterministic synthesizer: {e}")
                return self._generate_deterministic(query, context_chunks)

        else:
            return self._generate_deterministic(query, context_chunks)

    def _build_context_text(self, chunks: List[DocumentChunk]) -> str:
        parts = []
        for i, c in enumerate(chunks):
            caveat_line = f" [Note: {c.caveats}]" if c.caveats else ""
            parts.append(f"--- Context Section {i+1}: {c.title}{caveat_line} ---\n{c.content}\nSource: {c.url or 'N/A'}")
        return "\n\n".join(parts)

    def _generate_deterministic(self, query: str, context_chunks: List[DocumentChunk]) -> str:
        """
        Deterministic, 100% faithful synthesizer.
        Extracts verified facts from the most relevant retrieved chunks.
        Zero risk of hallucinated tokens or false claims.
        """
        primary_chunk = context_chunks[0]
        other_chunks = context_chunks[1:3]

        response_lines = []
        response_lines.append(f"### {primary_chunk.title}\n")

        # Include caveats if any
        if primary_chunk.caveats:
            response_lines.append(f"> **Documentation Note**: {primary_chunk.caveats}\n")

        # Format content cleanly
        content_lines = [line.strip() for line in primary_chunk.content.split("\n") if line.strip()]
        for line in content_lines:
            if line.startswith("-") or line.startswith("*"):
                response_lines.append(line)
            elif ":" in line:
                key, val = line.split(":", 1)
                response_lines.append(f"- **{key.strip()}**: {val.strip()}")
            else:
                response_lines.append(line)

        # If there are additional relevant chunks with unique information, add a brief reference
        if other_chunks:
            response_lines.append("\n**Related Information:**")
            for oc in other_chunks:
                first_line = oc.content.split("\n")[0]
                response_lines.append(f"- **{oc.title}**: {first_line}")

        # Add official action CTA
        if primary_chunk.url:
            response_lines.append(f"\n*Official Link:* [{primary_chunk.title}]({primary_chunk.url})")

        return "\n".join(response_lines)

    async def _generate_gemini(
        self, query: str, context_chunks: List[DocumentChunk], history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        context_str = self._build_context_text(context_chunks)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GEMINI_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"

        prompt_text = (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== Automaton AI Knowledge Base Context ===\n{context_str}\n\n"
            f"=== User Query ===\n{query}\n\n"
            f"Provide an accurate, faithful, well-formatted response:"
        )

        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {
                "temperature": 0.1,  # Low temperature to eliminate hallucination
                "maxOutputTokens": 800
            }
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                return candidates[0]["content"]["parts"][0]["text"].strip()
            return self._generate_deterministic(query, context_chunks)

    async def _generate_openai(
        self, query: str, context_chunks: List[DocumentChunk], history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        context_str = self._build_context_text(context_chunks)
        url = f"{settings.OPENAI_BASE_URL.rstrip('/')}/chat/completions"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nUser Question:\n{query}"
            }
        ]

        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": settings.OPENAI_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 800
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0]["message"]["content"].strip()
            return self._generate_deterministic(query, context_chunks)
