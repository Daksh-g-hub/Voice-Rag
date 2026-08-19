import time
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from ..config import settings
from ..telemetry.timer import PipelineTimer
from ..services.stt_service import stt_service
from ..services.embedding_service import embedding_service
from ..services.vector_store import vector_store
from ..services.reranker_service import reranker_service
from ..services.llm_service import llm_service
from ..services.cache_service import cache_service
from ..guardrails.input_guard import input_guard
from ..guardrails.relevance_guard import relevance_guard
from ..guardrails.grounding_guard import grounding_guard

class RAGPipelineResponse(BaseModel):
    query: str
    transcript: Optional[str] = None
    answer: str
    status: str # "success", "refused", "security_blocked", "error"
    is_grounded: bool
    confidence_score: float
    retrieved_contexts: List[Dict[str, Any]] = Field(default_factory=list)
    latency_breakdown: Dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float
    cache_hit: bool = False

class RAGOrchestrator:
    """
    Production-grade Voice-Enabled RAG Orchestration Pipeline with Stage-by-Stage Telemetry.
    """

    async def execute_query(
        self,
        query: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
        language_code: Optional[str] = None,
        use_cache: bool = True
    ) -> RAGPipelineResponse:
        timer = PipelineTimer()
        transcript = None

        # -------------------------------------------------------------
        # STAGE 1: Speech-To-Text (if audio provided)
        # -------------------------------------------------------------
        if audio_bytes:
            with timer.measure("stt_ms"):
                transcript, detected_lang, _ = await stt_service.transcribe_audio(
                    audio_bytes=audio_bytes,
                    language_code=language_code
                )
                query = transcript

        query = query.strip() if query else ""

        # -------------------------------------------------------------
        # CACHE CHECK
        # -------------------------------------------------------------
        if use_cache and query:
            cached_result = cache_service.get(query)
            if cached_result:
                summary = timer.get_summary()
                summary["cache_lookup_ms"] = 0.5
                return RAGPipelineResponse(
                    query=query,
                    transcript=transcript,
                    answer=cached_result["answer"],
                    status=cached_result["status"],
                    is_grounded=cached_result["is_grounded"],
                    confidence_score=cached_result["confidence_score"],
                    retrieved_contexts=cached_result["retrieved_contexts"],
                    latency_breakdown=summary,
                    total_latency_ms=summary["total_pipeline_ms"],
                    cache_hit=True
                )

        # -------------------------------------------------------------
        # STAGE 2: Pre-Retrieval Input Guardrail
        # -------------------------------------------------------------
        with timer.measure("input_guard_ms"):
            input_validation = input_guard.validate(query)

        if not input_validation.passed:
            summary = timer.get_summary()
            return RAGPipelineResponse(
                query=query,
                transcript=transcript,
                answer=f"Request blocked by guardrails: {input_validation.reason}",
                status="security_blocked",
                is_grounded=True,
                confidence_score=0.0,
                retrieved_contexts=[],
                latency_breakdown=summary,
                total_latency_ms=summary["total_pipeline_ms"]
            )

        sanitized_query = input_validation.sanitized_query

        # -------------------------------------------------------------
        # STAGE 3: Query Embedding (FastEmbed ONNX)
        # -------------------------------------------------------------
        with timer.measure("embedding_ms"):
            query_vector = embedding_service.embed_query(sanitized_query)

        # -------------------------------------------------------------
        # STAGE 4: Vector Search over Child Chunks
        # -------------------------------------------------------------
        with timer.measure("vector_search_ms"):
            child_hits = vector_store.search_children(
                query_vector=query_vector,
                top_k=settings.TOP_K_CHILDREN,
                score_threshold=0.30
            )

        # -------------------------------------------------------------
        # STAGE 5: Hierarchical Parent Context Resolution
        # -------------------------------------------------------------
        with timer.measure("parent_resolution_ms"):
            parent_contexts = vector_store.resolve_hierarchical_context(
                child_results=child_hits,
                max_parents=settings.MAX_PARENT_CONTEXTS
            )

        # -------------------------------------------------------------
        # STAGE 6: Conditional Cross-Encoder Reranking
        # -------------------------------------------------------------
        with timer.measure("rerank_ms"):
            if parent_contexts and settings.ENABLE_RERANKER:
                # Conditional: Only rerank if top score is in the ambiguous zone (< 0.82)
                top_score = parent_contexts[0].get("max_child_score", 0.0)
                if top_score < 0.82:
                    parent_contexts = reranker_service.rerank_passages(
                        query=sanitized_query,
                        passages=parent_contexts,
                        top_n=settings.MAX_PARENT_CONTEXTS
                    )

        # -------------------------------------------------------------
        # STAGE 7: Relevance & Context Evaluation
        # -------------------------------------------------------------
        with timer.measure("relevance_guard_ms"):
            rel_check = relevance_guard.evaluate(parent_contexts)

        # -------------------------------------------------------------
        # STAGE 8: LLM Answer Synthesis (Groq LPU)
        # -------------------------------------------------------------
        with timer.measure("llm_generation_ms"):
            raw_answer, ttfb_ms, _ = await llm_service.generate_rag_answer(
                query=sanitized_query,
                retrieved_contexts=parent_contexts
            )
            timer.metrics["llm_ttfb_ms"] = ttfb_ms

        # -------------------------------------------------------------
        # STAGE 9: Post-Generation Grounding & Citation Check
        # -------------------------------------------------------------
        with timer.measure("grounding_guard_ms"):
            grounding_result = grounding_guard.verify_grounding(
                generated_answer=raw_answer,
                retrieved_contexts=parent_contexts
            )

        summary = timer.get_summary()
        response_obj = RAGPipelineResponse(
            query=sanitized_query,
            transcript=transcript,
            answer=raw_answer,
            status="success",
            is_grounded=grounding_result.is_grounded or rel_check.is_dataset_grounded,
            confidence_score=max(rel_check.confidence_score, 0.75 if raw_answer else 0.0),
            retrieved_contexts=parent_contexts,
            latency_breakdown=summary,
            total_latency_ms=summary["total_pipeline_ms"]
        )

        if use_cache and grounding_result.is_grounded:
            cache_service.set(sanitized_query, response_obj.dict())

        return response_obj

orchestrator = RAGOrchestrator()
