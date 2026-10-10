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

from config import MAX_CONCURRENT_REQUESTS, RATE_LIMIT_RPM  # noqa: E402
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


CUSTOM_CSS = """
/* One centred column instead of the default wide gutters. */
gradio-app { max-width: 1220px; margin: 0 auto; }

/* Header: title and status badge sit on one line against a hairline rule. */
#kw-header {
    align-items: flex-end;
    gap: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border-color-primary, #ffffff1f);
}
#kw-title h1 {
    font-size: 1.3rem;
    font-weight: 650;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
}
#kw-title p {
    font-size: 0.8rem;
    opacity: 0.65;
    margin: 2px 0 0 0;
}
#kw-status { text-align: right; margin: 0; font-size: 0.78rem; }
#kw-status p { margin: 0; }

/* Sidebar: a quiet card, only as tall as its contents. */
#kw-side {
    gap: 10px;
    padding: 12px 14px;
    border: 1px solid var(--border-color-primary, #ffffff1f);
    border-radius: 10px;
    background: var(--background-fill-secondary, #ffffff08);
}
#kw-side h3 {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.55;
    margin: 0 0 4px 0;
}
#kw-side p { font-size: 0.8rem; line-height: 1.45; margin: 0; }

/* Toolbar: buttons sit tight under the prompt, stats pinned to the right. */
#kw-toolbar { align-items: center; gap: 8px; margin-top: -4px; }
#kw-latency {
    margin-left: auto;
    text-align: right;
    font-size: 0.76rem;
    opacity: 0.7;
    min-width: 90px;
}
#kw-latency p { margin: 0; }

/* Answer: the one surface framed like a document. */
#kw-answer {
    border: 1px solid var(--border-color-primary, #ffffff1f);
    border-radius: 10px;
    padding: 14px 16px;
    background: var(--background-fill-secondary, #ffffff08);
    min-height: 0;
}
#kw-answer p:first-child { margin-top: 0; }
#kw-answer p:last-child { margin-bottom: 0; }

#kw-foot { font-size: 0.72rem; opacity: 0.45; margin: 4px 0 0 0; }
#kw-foot p { margin: 0; }
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="10-Kwery") as demo:
        with gr.Row(elem_id="kw-header"):
            with gr.Column(scale=1, elem_id="kw-title", min_width=0):
                gr.Markdown("# 10-Kwery")
                gr.Markdown(
                    "Citation-grounded Q&A over SEC 10-K and 10-Q filings"
                )

            with gr.Column(scale=1, min_width=0):
                status = gr.Markdown(
                    "Loading retrieval index…", elem_id="kw-status"
                )

        with gr.Row():
            with gr.Column(scale=11, min_width=0, elem_id="kw-main"):
                question = gr.Textbox(
                    label="Question",
                    placeholder=EXAMPLE_QUESTIONS[0],
                    lines=2,
                    elem_id="kw-question",
                )

                with gr.Row(elem_id="kw-toolbar"):
                    ask_btn = gr.Button("Ask", variant="primary", size="sm")
                    clear_btn = gr.Button("New chat", size="sm")
                    latency = gr.Markdown(elem_id="kw-latency")

                answer = gr.Markdown(elem_id="kw-answer")

                with gr.Accordion(
                    "Sources", open=False, elem_id="kw-sources"
                ) as sources_accordion:
                    sources = gr.Markdown("_Ask a question to see sources._")

            with gr.Column(scale=5, min_width=0, elem_id="kw-side"):
                with gr.Group():
                    gr.Markdown("### Indexed")
                    companies_box = gr.Markdown("_loading…_")

                with gr.Group():
                    gr.Markdown("### Example questions")
                    gr.Examples(
                        examples=EXAMPLE_QUESTIONS,
                        inputs=[question],
                        label="",
                        elem_id="kw-examples",
                    )

                gr.Markdown(
                    "Answers come only from the retrieved excerpts; questions "
                    "outside the corpus are refused.",
                    elem_id="kw-foot",
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
            return "**Ready.**", companies

        def on_ask(question: str, history: list, request: gr.Request):
            if not question or not question.strip():
                return history, "", "", "Please enter a question.", "-"

            allowed, retry_after = _limiter.check(client_key(request))
            if not allowed:
                return (
                    history,
                    "",
                    "_Ask a question to see sources._",
                    "Too many requests — the free LLM tier is rate limited. "
                    f"Try again in about {retry_after:.0f}s.",
                    "rate limited",
                )

            payload = get_service().ask(question.strip(), history=history)
            history = history + [
                # Tickers are what make a follow-up like "how did that change?"
                # retrievable: rewriting rewrites it against the routed company.
                {
                    "role": "user",
                    "content": question,
                    "tickers": payload.get("tickers") or [],
                },
                {
                    "role": "assistant",
                    "content": payload.get("answer") or "",
                },
            ]

            return (
                history,
                render_answer(payload),
                render_sources(payload),
                "",
                f"{payload.get('latency_s', 0):.1f}s",
            )

        def on_clear():
            return [], "", "_Ask a question to see sources._", "", "-"

        demo.load(load_status, outputs=[status, companies_box])

        click_outputs = [history_state, answer, sources, question, latency]

        ask_btn.click(
            on_ask,
            inputs=[question, history_state],
            outputs=click_outputs,
        )
        question.submit(
            on_ask,
            inputs=[question, history_state],
            outputs=click_outputs,
        )
        clear_btn.click(
            on_clear,
            outputs=click_outputs,
        )

    return demo

# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str
    tickers: list[str] = Field(default_factory=list)


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

    ``css`` goes to ``mount_gradio_app``, not to ``gr.Blocks``: Gradio 6 moved
    theme/css/js/head from the Blocks constructor to launch() or the mount, and
    silently ignores them if passed there instead.
    """
    api = create_api_app()
    demo = build_demo()
    demo.queue(default_concurrency_limit=MAX_CONCURRENT_REQUESTS)
    return gr.mount_gradio_app(
        api, demo, path="/", css=CUSTOM_CSS
    )


app = build_app()


if __name__ == "__main__":
    logger.info("10-Kwery listening on %s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT)
