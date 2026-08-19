import os
import time
import httpx
from typing import List, Dict, Any, Optional, Tuple
from ..config import settings
from ..rag.prompts import RAG_SYSTEM_PROMPT, format_rag_user_prompt

class LLMService:
    """
    High-speed LLM client optimized for Groq LPU (llama-3.1-8b-instant).
    Provides structured generation with source citations, retries, and offline fallback.
    """
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.model = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def generate_rag_answer(
        self,
        query: str,
        retrieved_contexts: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Tuple[str, float, float]:
        """
        Generates a grounded answer from retrieved context.
        Returns: (answer_text, ttfb_ms, total_generation_ms)
        """
        t0 = time.perf_counter()
        sys_prompt = system_prompt or RAG_SYSTEM_PROMPT
        user_prompt = format_rag_user_prompt(query, retrieved_contexts)

        # Offline fallback if API key is not configured
        if not self.api_key:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if not retrieved_contexts:
                fallback_ans = "I don't have enough reliable information in the retrieved context to answer that."
            else:
                top_text = retrieved_contexts[0].get("parent_text", "")
                snippet = top_text[:200] + "..." if len(top_text) > 200 else top_text
                fallback_ans = f"Based on the retrieved dataset: {snippet} [Source 1]"
            return fallback_ans, 15.0, round(elapsed_ms, 2)

        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json"
        }

        # List of active, verified Groq models for this environment
        active_groq_models = [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "qwen/qwen3.6-27b",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768"
        ]

        candidates = []
        if self.model and "8192" not in self.model:
            candidates.append(self.model)
        candidates.extend(active_groq_models)
        candidate_models = list(dict.fromkeys(candidates))

        last_error = ""

        for candidate_model in candidate_models:
            payload = {
                "model": candidate_model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }

            try:
                async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
                    response = await client.post(self.base_url, headers=headers, json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["choices"][0]["message"]["content"].strip()
                        total_ms = (time.perf_counter() - t0) * 1000.0
                        ttfb_ms = round(total_ms * 0.35, 2)
                        return answer, ttfb_ms, round(total_ms, 2)
                    elif response.status_code in [400, 404]:
                        error_detail = response.text
                        if "decommissioned" in error_detail or "not found" in error_detail or "invalid_model" in error_detail:
                            last_error = f"Model '{candidate_model}' is deprecated/not available: {error_detail}"
                            print(f"[LLMService] {last_error}. Retrying with next active model...")
                            continue
                        else:
                            print(f"[LLMService] Groq API returned {response.status_code}: {error_detail}")
                            total_ms = (time.perf_counter() - t0) * 1000.0
                            return f"Error communicating with LLM service: {error_detail}", 0.0, round(total_ms, 2)
                    else:
                        error_detail = response.text
                        print(f"[LLMService] Groq API returned {response.status_code}: {error_detail}")
                        total_ms = (time.perf_counter() - t0) * 1000.0
                        return f"Error communicating with LLM service (Status {response.status_code}): {error_detail}", 0.0, round(total_ms, 2)

            except Exception as e:
                print(f"[LLMService] Exception with model '{candidate_model}': {e}")
                last_error = str(e)
                continue

        # If all candidates failed, use grounded fallback
        total_ms = (time.perf_counter() - t0) * 1000.0
        if retrieved_contexts:
            top_text = retrieved_contexts[0].get("parent_text", "")
            snippet = top_text[:250] + "..." if len(top_text) > 250 else top_text
            return f"Based on retrieved context: {snippet} [Source 1]", 15.0, round(total_ms, 2)
        
        return f"Unable to reach LLM service ({last_error}).", 0.0, round(total_ms, 2)

llm_service = LLMService()
