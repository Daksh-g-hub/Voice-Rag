from typing import List, Dict, Any

RAG_SYSTEM_PROMPT = """You are an intelligent, low-latency AI Assistant integrated into a Voice-Enabled RAG system for Hacker House Goa 2026.

OPERATIONAL GUIDELINES:
1. Grounded Context First: If the provided retrieved context contains facts relevant to the user's question, base your answer primarily on those facts and append source citations like [Source 1], [Source 2].
2. Intelligent Synthesis: If the question is general knowledge, conversational, or broader than the retrieved context, answer the user's question accurately, intelligently, and clearly without refusing.
3. Voice Conciseness: Keep your response direct, natural, and concise (under 2-4 sentences) so it sounds crisp and fluent in a voice application.
4. Professional & Helpful Tone: Always be polite, factual, and insightful.
"""

def format_rag_user_prompt(query: str, context_passages: List[Dict[str, Any]]) -> str:
    """
    Formats the user query and retrieved parent passages with source identifiers.
    """
    if context_passages:
        formatted_contexts = []
        for idx, ctx in enumerate(context_passages, 1):
            source_id = ctx.get("parent_id") or ctx.get("doc_id") or f"doc_{idx}"
            lang = ctx.get("language", "en")
            text = ctx.get("parent_text", ctx.get("text", "")).strip()
            formatted_contexts.append(f"[Source {idx}] (ID: {source_id}, Lang: {lang})\n{text}")

        context_str = "\n\n".join(formatted_contexts)

        return f"""--- RETRIEVED CONTEXT START ---
{context_str}
--- RETRIEVED CONTEXT END ---

User Question: {query}

Provide a helpful, grounded, and concise answer (include source citations if context is used):"""
    else:
        return f"""User Question: {query}

Provide an intelligent, accurate, and concise answer for voice output:"""
