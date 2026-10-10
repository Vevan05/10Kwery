"""Web front end for 10-Kwery.

One process serves three surfaces:

    /            Gradio UI for humans
    /api/*       JSON API for programmatic use
    /health      readiness probe

Hugging Face Spaces runs ``python app.py`` and expects the app to listen on
port 7860. Gradio is mounted into a FastAPI app so both surfaces share one
worker and one set of in-process model weights - the free CPU Basic tier gives
16 GB of RAM but only 2 cores, so there is no benefit in loading the embedder
and reranker twice.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv()

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from config import RATE_LIMIT_RPM  # noqa: E402
from service import EXAMPLE_QUESTIONS, get_service  # noqa: E402

logger = logging.getLogger(__name__)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))

MAX_TRACKED_CLIENTS = 1000


class TokenBucketLimiter:
    """Per-client token bucket.

    Every visitor shares one free-tier Groq key, so one aggressive client must
    not be able to push the demo into a rate-limit wall for everyone else.
    """

    def __init__(self, requests_per_minute: float) -> None:
        self._rate = max(requests_per_minute, 0.0) / 60.0
        self._capacity = 1.0
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, float]] = {}

    def check(self, key: str) -> tuple[bool, float]:
        """Return ``(allowed, retry_after_seconds)``."""
        if self._rate <= 0:
            return True, 0.0

        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self._capacity, now))
            tokens = min(self._capacity, tokens + (now - last) * self._rate)

            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                return True, 0.0

            self._buckets[key] = (tokens, now)
            self._prune()
            return False, (1.0 - tokens) / self._rate

    def _prune(self) -> None:
        """Drop stale buckets so an unauthenticated demo cannot grow unbounded."""
        if len(self._buckets) <= MAX_TRACKED_CLIENTS:
            return

        expiry = time.monotonic() - 3600
        self._buckets = {
            k: v for k, v in self._buckets.items() if v[1] > expiry
        }
        if len(self._buckets) > MAX_TRACKED_CLIENTS:
            # Keep the most recently active entries if everything is fresh.
            self._buckets = dict(
                sorted(self._buckets.items(), key=lambda kv: kv[1][1])[
                    -MAX_TRACKED_CLIENTS:
                ]
            )


_limiter = TokenBucketLimiter(RATE_LIMIT_RPM)


def client_key(request: Request) -> str:
    """Best-effort caller identity. Spaces sits behind a proxy, so prefer XFF."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "anonymous"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_answer(payload: dict) -> str:
    if payload.get("refused"):
        return (
            "**I cannot answer that from the indexed filings.**\n\n"
            f"{payload.get('reason', '')}"
        )
    return payload.get("answer", "")


def _section_label(section: str | None) -> str:
    section = (section or "").strip().replace("\n", " ")
    return section[:70] if section else "unspecified section"


def render_sources(payload: dict) -> str:
    sources = payload.get("sources") or []
    if not sources:
        return "_No sources._"

    lines = []
    for s in sources:
        header = (
            f"**[{s['n']}] {s['ticker']} {s['form']} "
            f"filed {s['filing_date']}** — {_section_label(s.get('section'))}"
        )
        score = s.get("rerank_score")
        if score is not None:
            header += f" · relevance `{score:.2f}`"

        url = s.get("source_url")
        if url:
            header += f"\n\n<{url}>"

        lines.append(header)

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="10-Kwery") as demo:
        gr.Markdown(
            "# 10-Kwery\n"
            "Ask questions about SEC 10-K and 10-Q filings. Answers are grounded "
            "in retrieved excerpts and every claim carries a bracketed citation "
            "you can check against the sources below."
        )

        status = gr.Markdown("_Loading retrieval index..._")
        companies_box = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="Question",
                    placeholder=EXAMPLE_QUESTIONS[0],
                    lines=2,
                )
                with gr.Row():
                    ask_btn = gr.Button("Ask", variant="primary")
                    clear_btn = gr.Button("New chat")

                answer = gr.Markdown()
                sources = gr.Markdown()

            with gr.Column(scale=1):
                gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=question,
                    label="Example questions",
                )

        history_state = gr.State(value=[])

        def load_status():
            service = get_service()
            try:
                service.warmup()
            except Exception as e:
                logger.exception("Warm-up failed")
                return f"**Startup failed:** {e}", ""

            companies = ", ".join(service.companies()) or "none"
            return "**Ready.** Retrieval index loaded.", f"**Indexed:** {companies}"

        def on_ask(question: str, history: list, request: gr.Request):
            if not question or not question.strip():
                return history, "", "", "Please enter a question."

            allowed, retry_after = _limiter.check(client_key(request))
            if not allowed:
                return (
                    history,
                    "",
                    "",
                    "Too many requests — the free LLM tier is rate limited. "
                    f"Try again in about {retry_after:.0f}s.",
                )

            payload = get_service().ask(question.strip(), history=history)
            history = history + [
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": payload.get("answer") or "",
                },
            ]

            return history, render_answer(payload), render_sources(payload), ""

        def on_clear():
            return [], "", "", ""

        demo.load(load_status, outputs=[status, companies_box])

        ask_btn.click(
            on_ask,
            inputs=[question, history_state],
            outputs=[history_state, answer, sources, question],
        )
        question.submit(
            on_ask,
            inputs=[question, history_state],
            outputs=[history_state, answer, sources, question],
        )
        clear_btn.click(
            on_clear,
            outputs=[history_state, answer, sources, question],
        )

    return demo


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list)
    use_cache: bool = True
    verify_entailment: bool = False


def create_api_app() -> FastAPI:
    api = FastAPI(
        title="10-Kwery API",
        description="Citation-grounded Q&A over SEC 10-K and 10-Q filings.",
        version="1.0.0",
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/health")
    def health() -> dict:
        service = get_service()
        return {
            "ready": service.ready,
            "elapsed_s": round(time.monotonic() - service.started_at, 1),
            "load_error": service.load_error,
        }

    @api.get("/api/companies")
    def companies() -> dict:
        return {"companies": get_service().companies()}

    @api.post("/api/query")
    def query(body: QueryIn, request: Request) -> JSONResponse:
        allowed, retry_after = _limiter.check(client_key(request))
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "detail": (
                        "The free LLM tier is rate limited. "
                        f"Retry in about {retry_after:.0f}s."
                    ),
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        try:
            payload = get_service().ask(
                body.question,
                history=[t.model_dump() for t in body.history],
                use_cache=body.use_cache,
                verify_entailment=body.verify_entailment,
            )
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception:
            logger.exception("Query failed")
            return JSONResponse(
                status_code=500,
                content={"error": "internal_error"},
            )

        return JSONResponse(content=payload)

    return api


def build_app() -> FastAPI:
    """Compose the JSON API and the Gradio UI into one FastAPI application.

    The custom routes are registered before the mount: FastAPI matches in
    registration order, so ``/api/*`` and ``/health`` must not fall through to
    the Gradio root handler.
    """
    api = create_api_app()
    demo = build_demo()
    return gr.mount_gradio_app(api, demo, path="/")


app = build_app()


if __name__ == "__main__":
    logger.info("10-Kwery listening on %s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT)
