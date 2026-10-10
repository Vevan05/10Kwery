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

import html
import logging
import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
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

# Queries run in a worker so the UI generator can keep yielding progress while
# the retrieval pipeline blocks. The service's own BoundedSemaphore still bounds
# real concurrency, so this only needs to cover the in-flight events.
_QUERY_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(4, MAX_CONCURRENT_REQUESTS + 2),
    thread_name_prefix="kw-query",
)

# Rolling latency sample, used for the estimate shown while a query runs.
# The first few queries have no sample, so they fall back to DEFAULT_ESTIMATE_S.
_recent: list[float] = []
_recent_lock = threading.Lock()
_LATENCY_WINDOW = 12
DEFAULT_ESTIMATE_S = 15.0


def _record_latency(seconds: float) -> None:
    with _recent_lock:
        _recent.append(seconds)
        del _recent[:-_LATENCY_WINDOW]


def estimated_latency() -> float:
    """Median of the last few queries - robust to one slow or cached run."""
    with _recent_lock:
        sample = sorted(_recent)
    if not sample:
        return DEFAULT_ESTIMATE_S
    return sample[len(sample) // 2]


def _wait(future: Future, timeout: float) -> tuple[bool, set]:
    """Block until a worker future finishes or ``timeout`` elapses."""
    import concurrent.futures

    done, _pending = concurrent.futures.wait([future], timeout=timeout)
    return bool(done), done


def client_key(request: Request) -> str:
    """Best-effort caller identity. Spaces sits behind a proxy, so prefer XFF."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "anonymous"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

EMPTY_ANSWER = (
    '<div class="kw-empty">'
    '<div class="kw-empty-icon">&#8984;</div>'
    "<div><b>Ask anything about the indexed filings.</b><br>"
    "Every answer is grounded in retrieved excerpts and cited.</div>"
    "</div>"
)
EMPTY_SOURCES = '<div class="kw-muted">Ask a question to see sources.</div>'


def render_answer(payload: dict) -> str:
    if payload.get("refused"):
        reason = html.escape(payload.get("reason", "") or "")
        return (
            '<div class="kw-refusal"><b>I cannot answer that from the '
            "indexed filings.</b>"
            f"<p>{reason}</p></div>"
        )
    return payload.get("answer", "")


def render_notice(message: str) -> str:
    return f'<div class="kw-notice">{html.escape(message)}</div>'


LOADING_SOURCES = (
    '<div class="kw-muted">Sources appear alongside the answer.</div>'
)


def render_loading(elapsed: float, estimate: float) -> str:
    """Spinner card shown inside the answer panel while a query runs."""
    return (
        '<div class="kw-loading">'
        '<span class="kw-spinner" aria-hidden="true"></span>'
        '<div class="kw-loading-text">'
        '<div class="kw-loading-title">Searching filings and writing an answer'
        "</div>"
        f'<div class="kw-loading-sub">usually about {estimate:.0f}s '
        f"&middot; {elapsed:.0f}s elapsed</div>"
        "</div></div>"
    )


def _section_label(section: str | None) -> str:
    section = (section or "").strip().replace("\n", " ")
    return section[:70] if section else "unspecified section"


def render_sources(payload: dict) -> str:
    sources = payload.get("sources") or []
    if not sources:
        return '<div class="kw-muted">No sources.</div>'

    cards = []
    for s in sources:
        n = html.escape(str(s.get("n", "")))
        ticker = html.escape(str(s.get("ticker", "")))
        form = html.escape(str(s.get("form", "")))
        filed = html.escape(str(s.get("filing_date", "")))
        section = html.escape(_section_label(s.get("section")))

        score = s.get("rerank_score")
        score_html = ""
        if score is not None:
            score_html = f'<span class="kw-score">relevance {score:.2f}</span>'

        url = s.get("source_url")
        link_html = ""
        if url:
            safe = html.escape(str(url), quote=True)
            link_html = (
                f'<a class="kw-link" href="{safe}" target="_blank" '
                f'rel="noopener noreferrer">View filing &#8599;</a>'
            )

        cards.append(
            '<div class="kw-src">'
            f'<div class="kw-src-n">{n}</div>'
            '<div class="kw-src-body">'
            '<div class="kw-src-top">'
            f'<span class="kw-tag">{ticker}</span>'
            f'<span class="kw-tag kw-tag-alt">{form}</span>'
            f'<span class="kw-src-date">filed {filed}</span>'
            f"{score_html}"
            "</div>"
            f'<div class="kw-src-section">{section}</div>'
            f"{link_html}"
            "</div></div>"
        )

    return '<div class="kw-src-list">' + "".join(cards) + "</div>"


def status_html(kind: str, text: str) -> str:
    return (
        f'<span class="kw-pill kw-pill-{kind}">'
        f'<span class="kw-dot"></span>{html.escape(text)}</span>'
    )


def latency_html(text: str) -> str:
    if not text:
        return ""
    return f'<span class="kw-lat">{html.escape(text)}</span>'


def companies_html(names: list[str]) -> str:
    if not names:
        return '<div class="kw-muted">none</div>'
    chips = "".join(f'<span class="kw-chip">{html.escape(n)}</span>' for n in names)
    return f'<div class="kw-chips">{chips}</div>'


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


THEME = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="zinc",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#fafafa",
    body_background_fill_dark="#0c0c0e",
    block_background_fill="transparent",
    block_background_fill_dark="transparent",
    block_border_width="0px",
    block_shadow="none",
    block_shadow_dark="none",
    block_padding="0px",
    input_background_fill="white",
    input_background_fill_dark="#131316",
    input_border_width="1px",
    input_shadow="none",
    input_shadow_focus="0 0 0 3px rgba(99, 102, 241, .15)",
    button_primary_background_fill="#18181b",
    button_primary_background_fill_hover="#27272a",
    button_primary_background_fill_dark="#fafafa",
    button_primary_background_fill_hover_dark="#e4e4e7",
    button_primary_text_color="#fafafa",
    button_primary_text_color_dark="#18181b",
    button_primary_shadow="none",
    button_secondary_background_fill="transparent",
    button_secondary_background_fill_dark="transparent",
    button_secondary_shadow="none",
)

CUSTOM_CSS = """
:root {
    --kw-accent: #6366f1;
    --kw-line: rgba(113, 113, 122, .22);
    --kw-soft: rgba(113, 113, 122, .07);
    --kw-muted: rgba(113, 113, 122, .95);
    --kw-ok: #10b981;
    --kw-warn: #f59e0b;
    --kw-err: #ef4444;
    --kw-gap: 24px;
    --kw-track: rgba(113, 113, 122, .22);
}

footer { display: none !important; }

/* ---------- Page: always fill the viewport, never leave a void ---------- */
.gradio-container {
    max-width: 1160px !important;
    min-height: 100vh;
    margin: 0 auto !important;
    padding: 32px 28px 28px 28px !important;
    display: flex;
    flex-direction: column;
    gap: var(--kw-gap);
}
.gradio-container > main,
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    gap: var(--kw-gap);
    min-height: 0;
}

/* ---------- Header ---------- */
#kw-header {
    flex: 0 0 auto;
    align-items: center;
    gap: 16px;
    padding: 0 0 var(--kw-gap) 0;
    border-bottom: 1px solid var(--kw-line);
}
#kw-title h1 {
    font-size: 1.35rem;
    font-weight: 650;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin: 0;
}
#kw-title p { font-size: .84rem; color: var(--kw-muted); margin: 4px 0 0 0; }
#kw-status { text-align: right; }

.kw-pill {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: .78rem; color: var(--kw-muted);
}
.kw-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--kw-warn); }
.kw-pill-loading .kw-dot { animation: kw-pulse 1.2s ease-in-out infinite; }
.kw-pill-ok .kw-dot { background: var(--kw-ok); }
.kw-pill-err .kw-dot { background: var(--kw-err); }
@keyframes kw-pulse { 50% { opacity: .25; } }

/* Gradio's "processing | 1.2s" overlay must never cover the small widgets. */
#kw-status .wrap, #kw-companies .wrap, #kw-latency .wrap { display: none !important; }
#kw-status, #kw-companies, #kw-latency { min-height: 0; }
#kw-status .pending, #kw-companies .pending, #kw-latency .pending { opacity: 1 !important; }

/* ---------- Body: two columns that stretch to the bottom edge ---------- */
#kw-body {
    flex: 1 1 auto;
    align-items: stretch;
    gap: calc(var(--kw-gap) * 1.5);
    min-height: calc(100vh - 190px);
}
#kw-main, #kw-side {
    display: flex;
    flex-direction: column;
    gap: var(--kw-gap);
}

/* ---------- Question + toolbar ---------- */
#kw-question { flex: 0 0 auto; }
#kw-question textarea {
    font-size: 1rem;
    line-height: 1.55;
    padding: 16px 18px;
    border-radius: 12px;
    border-color: var(--kw-line);
    resize: none;
}
#kw-question textarea:focus { border-color: var(--kw-accent); }
#kw-question label > span { display: none; }

#kw-toolbar { flex: 0 0 auto; align-items: center; gap: 10px; margin-top: -8px; }
#kw-toolbar button {
    flex: 0 0 auto; min-width: 92px;
    font-weight: 600; font-size: .86rem;
    border-radius: 10px;
}
#kw-toolbar button.secondary { border: 1px solid var(--kw-line); color: var(--kw-muted); }
#kw-toolbar button.secondary:hover { color: inherit; background: var(--kw-soft); }
#kw-latency { margin-left: auto; text-align: right; min-width: 70px; }
.kw-lat { font-family: var(--font-mono); font-size: .74rem; color: var(--kw-muted); }

/* ---------- Answer: the flexible surface that takes up the slack ---------- */
#kw-answer {
    flex: 1 1 auto;
    min-height: 240px;
    padding: 24px 26px;
    border: 1px solid var(--kw-line);
    border-radius: 14px;
    font-size: .98rem;
    line-height: 1.75;
    overflow: auto;
}
#kw-answer p:first-child { margin-top: 0; }
#kw-answer p:last-child { margin-bottom: 0; }
#kw-answer ul, #kw-answer ol { padding-left: 1.3em; }
#kw-answer table { font-size: .88rem; }

.kw-empty {
    min-height: 200px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 6px; text-align: center; color: var(--kw-muted);
    font-size: .9rem; line-height: 1.6;
}
.kw-empty b { color: inherit; font-weight: 600; font-size: .98rem; }
.kw-empty-icon {
    width: 40px; height: 40px; margin-bottom: 8px;
    display: grid; place-items: center;
    border-radius: 12px; font-size: 1.15rem;
    border: 1px solid var(--kw-line);
}
.kw-refusal b { color: var(--kw-warn); }
.kw-refusal p { margin: 8px 0 0 0; color: var(--kw-muted); }
.kw-notice {
    padding: 12px 16px; border-radius: 10px; font-size: .9rem;
    background: rgba(245, 158, 11, .08);
    border: 1px solid rgba(245, 158, 11, .3);
}

/* ---------- Loading ---------- */
.kw-loading {
    min-height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    color: var(--kw-muted);
}
.kw-spinner {
    flex: 0 0 auto;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    border: 2.5px solid var(--kw-track);
    border-top-color: var(--kw-accent);
    animation: kw-spin .7s linear infinite;
}
@keyframes kw-spin { to { transform: rotate(360deg); } }
.kw-loading-title {
    font-weight: 600;
    font-size: .95rem;
    color: inherit;
}
.kw-loading-sub {
    margin-top: 4px;
    font-size: .78rem;
    font-family: var(--font-mono);
}

/* ---------- Sources ---------- */
#kw-sources {
    flex: 0 0 auto;
    border: 1px solid var(--kw-line);
    border-radius: 14px;
    padding: 4px 8px;
}
#kw-sources button.label-wrap, #kw-sources .label-wrap { font-weight: 600; font-size: .86rem; }
.kw-src-list { display: flex; flex-direction: column; }
.kw-src {
    display: flex; gap: 14px; align-items: flex-start;
    padding: 14px 6px;
    border-top: 1px solid var(--kw-line);
}
.kw-src:first-child { border-top: none; }
.kw-src-n {
    flex: 0 0 22px; height: 22px; border-radius: 6px;
    display: grid; place-items: center;
    font-size: .72rem; font-weight: 700;
    background: var(--kw-soft); color: var(--kw-muted);
}
.kw-src-body { min-width: 0; flex: 1; }
.kw-src-top { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 10px; }
.kw-tag { font-size: .8rem; font-weight: 700; letter-spacing: .02em; }
.kw-tag-alt { font-weight: 500; color: var(--kw-muted); }
.kw-src-date { font-size: .76rem; color: var(--kw-muted); }
.kw-score { margin-left: auto; font-family: var(--font-mono); font-size: .7rem; color: var(--kw-muted); }
.kw-src-section { font-size: .86rem; margin-top: 4px; line-height: 1.45; }
.kw-link {
    display: inline-block; margin-top: 6px;
    font-size: .78rem; font-weight: 600;
    color: var(--kw-accent); text-decoration: none;
}
.kw-link:hover { text-decoration: underline; }
.kw-muted { font-size: .85rem; color: var(--kw-muted); padding: 10px 6px; }

/* ---------- Sidebar: no box, just a hairline and breathing room ---------- */
#kw-side {
    padding-left: calc(var(--kw-gap) * 1.5);
    border-left: 1px solid var(--kw-line);
    gap: 32px;
}
#kw-side .group, #kw-side .gr-group {
    background: transparent; border: none; box-shadow: none; padding: 0;
    display: flex; flex-direction: column; gap: 12px;
}
#kw-side h3 {
    font-size: .7rem; font-weight: 650;
    text-transform: uppercase; letter-spacing: .1em;
    color: var(--kw-muted);
    margin: 0;
}
.kw-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.kw-chip {
    font-size: .78rem; font-weight: 600;
    padding: 4px 11px; border-radius: 999px;
    border: 1px solid var(--kw-line);
}

/* Example questions as quiet, clickable lines. */
#kw-examples .label, #kw-examples .label-wrap { display: none; }
#kw-examples table { border: none; width: 100%; border-collapse: separate; border-spacing: 0 2px; }
#kw-examples tr { cursor: pointer; background: transparent; }
#kw-examples td {
    font-size: .84rem; line-height: 1.45;
    padding: 10px 12px; border: none; border-radius: 8px;
    color: var(--kw-muted);
    transition: background .12s ease, color .12s ease;
}
#kw-examples tr:hover td { background: var(--kw-soft); color: inherit; }

/* Pinned to the bottom of the sidebar so both columns end on the same line. */
#kw-foot {
    margin-top: auto;
    padding-top: 20px;
    font-size: .74rem; line-height: 1.55;
    color: var(--kw-muted);
    border-top: 1px solid var(--kw-line);
}
#kw-foot p { margin: 0; }

/* ---------- Small screens ---------- */
@media (max-width: 800px) {
    .gradio-container { padding: 20px 16px !important; }
    #kw-header { flex-direction: column; align-items: flex-start; }
    #kw-status { text-align: left; }
    #kw-side { padding-left: 0; border-left: none; border-top: 1px solid var(--kw-line); padding-top: var(--kw-gap); }
    #kw-body { min-height: 0; }
    #kw-answer { padding: 18px; }
}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="10-Kwery", fill_height=True) as demo:
        with gr.Row(elem_id="kw-header"):
            with gr.Column(scale=3, elem_id="kw-title", min_width=0):
                gr.Markdown("# 10-Kwery")
                gr.Markdown(
                    "Citation-grounded Q&A over SEC 10-K and 10-Q filings"
                )

            with gr.Column(scale=1, min_width=0):
                status = gr.HTML(
                    status_html("loading", "Loading retrieval index…"),
                    elem_id="kw-status",
                )

        with gr.Row(equal_height=True, elem_id="kw-body"):
            with gr.Column(scale=11, min_width=0, elem_id="kw-main"):
                question = gr.Textbox(
                    label="Question",
                    placeholder=EXAMPLE_QUESTIONS[0],
                    lines=3,
                    max_lines=8,
                    elem_id="kw-question",
                    autofocus=True,
                )

                with gr.Row(elem_id="kw-toolbar"):
                    ask_btn = gr.Button("Ask", variant="primary", size="md")
                    clear_btn = gr.Button("New chat", size="md")
                    latency = gr.HTML(elem_id="kw-latency")

                answer = gr.Markdown(EMPTY_ANSWER, elem_id="kw-answer")

                with gr.Accordion(
                    "Sources", open=False, elem_id="kw-sources"
                ):
                    sources = gr.HTML(EMPTY_SOURCES)

            with gr.Column(scale=4, min_width=0, elem_id="kw-side"):
                with gr.Group():
                    gr.Markdown("### Indexed")
                    companies_box = gr.HTML(
                        '<div class="kw-muted">loading…</div>',
                        elem_id="kw-companies",
                    )

                with gr.Group():
                    gr.Markdown("### Try asking")
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
                return status_html("err", f"Startup failed: {e}"), ""
            return (
                status_html("ok", "Ready"),
                companies_html(list(service.companies())),
            )

        def on_ask(question: str, history: list, request: gr.Request):
            """Answer one question, streaming a loading card while it runs.

            A generator, so the loading state is emitted before the pipeline
            starts and updated every second: a 15s retrieval would otherwise
            leave the panel frozen on the previous answer with no sign of life.
            Gradio streams each yield straight to the components.
            """
            # Notices go into the answer card; the user's text stays in the
            # textbox so a rate-limit or empty submit never wipes the prompt.
            if not question or not question.strip():
                yield (
                    history,
                    render_notice("Please enter a question."),
                    EMPTY_SOURCES,
                    gr.update(),
                    "",
                )
                return

            allowed, retry_after = _limiter.check(client_key(request))
            if not allowed:
                yield (
                    history,
                    render_notice(
                        "Too many requests — the free LLM tier is rate "
                        f"limited. Try again in about {retry_after:.0f}s."
                    ),
                    EMPTY_SOURCES,
                    gr.update(),
                    latency_html("rate limited"),
                )
                return

            started = time.monotonic()
            estimate = estimated_latency()
            future = _QUERY_EXECUTOR.submit(
                get_service().ask, question.strip(), history=history
            )

            # First frame goes out before the worker reports anything, so the
            # spinner appears as soon as the button is pressed.
            yield (
                history,
                render_loading(0.0, estimate),
                LOADING_SOURCES,
                gr.update(),
                latency_html("0s"),
            )

            while True:
                done, _pending = _wait(future, 1.0)
                if done:
                    break
                elapsed = time.monotonic() - started
                yield (
                    history,
                    render_loading(elapsed, estimate),
                    LOADING_SOURCES,
                    gr.update(),
                    latency_html(f"{elapsed:.0f}s"),
                )

            try:
                payload = future.result()
            except Exception as e:
                logger.exception("Query failed")
                yield (
                    history,
                    render_notice(f"Query failed: {e}"),
                    EMPTY_SOURCES,
                    gr.update(),
                    latency_html("failed"),
                )
                return

            _record_latency(payload.get("latency_s", 0.0) or 0.0)

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

            yield (
                history,
                render_answer(payload),
                render_sources(payload),
                "",
                latency_html(f"{payload.get('latency_s', 0):.1f}s"),
            )

        def on_clear():
            return [], EMPTY_ANSWER, EMPTY_SOURCES, "", ""

        demo.load(
            load_status,
            outputs=[status, companies_box],
            show_progress="hidden",
        )

        click_outputs = [history_state, answer, sources, question, latency]

        ask_btn.click(
            on_ask,
            inputs=[question, history_state],
            outputs=click_outputs,
            show_progress="hidden",
        )
        question.submit(
            on_ask,
            inputs=[question, history_state],
            outputs=click_outputs,
            show_progress="hidden",
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

    ``theme`` and ``css`` go to ``mount_gradio_app``, not to ``gr.Blocks``:
    Gradio 6 moved theme/css/js/head from the Blocks constructor to launch()
    or the mount, and silently ignores them if passed there instead.
    """
    api = create_api_app()
    demo = build_demo()
    demo.queue(default_concurrency_limit=MAX_CONCURRENT_REQUESTS)
    return gr.mount_gradio_app(
        api, demo, path="/", theme=THEME, css=CUSTOM_CSS
    )


app = build_app()


if __name__ == "__main__":
    logger.info("10-Kwery listening on %s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT)