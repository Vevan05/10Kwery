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
import math
import os
import re
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
    '<div class="kw-empty-icon">'
    '<svg viewBox="0 0 24 24" width="26" height="26" fill="none" '
    'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
    'stroke-linejoin="round"><circle cx="11" cy="11" r="7"/>'
    '<path d="m20 20-3.5-3.5"/><path d="M8.5 11h5M11 8.5v5" opacity=".6"/>'
    "</svg></div>"
    '<div class="kw-empty-title">Ask anything about the indexed filings</div>'
    '<div class="kw-empty-sub">Every answer is grounded in retrieved '
    "excerpts and cited back to the source filing.</div>"
    '<div class="kw-empty-badges">'
    '<span class="kw-badge">Cited</span>'
    '<span class="kw-badge">Grounded</span>'
    '<span class="kw-badge">SEC 10-K / 10-Q</span>'
    "</div></div>"
)
EMPTY_SOURCES = '<div class="kw-muted">Ask a question to see sources.</div>'

# Citations, money, percentages: the things a reader scans a financial answer
# for. One alternation so already-wrapped spans are never wrapped twice.
_HIGHLIGHT_RE = re.compile(
    r"(?P<cite>\[\d+(?:\s*,\s*\d+)*\])"
    r"|(?P<money>\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:trillion|billion|million|thousand|[TBMK])\b)?)"
    r"|(?P<pct>\b\d+(?:\.\d+)?\s?%)",
    flags=re.IGNORECASE,
)


def highlight_answer(text: str) -> str:
    """Wrap citations and key figures in styled spans (markdown allows inline HTML)."""

    def repl(m: re.Match) -> str:
        if m.group("cite"):
            nums = re.sub(r"[\[\]\s]", "", m.group("cite"))
            return f'<span class="kw-cite">{nums}</span>'
        if m.group("money"):
            return f'<mark class="kw-num">{m.group("money")}</mark>'
        return f'<mark class="kw-pct">{m.group("pct")}</mark>'

    return _HIGHLIGHT_RE.sub(repl, text)


def render_answer(payload: dict) -> str:
    if payload.get("refused"):
        reason = html.escape(payload.get("reason", "") or "")
        return (
            '<div class="kw-refusal">'
            '<div class="kw-refusal-icon">!</div>'
            "<div><b>I cannot answer that from the indexed filings.</b>"
            f"<p>{reason}</p></div></div>"
        )
    answer = payload.get("answer", "") or ""
    tickers = payload.get("tickers") or []
    header = ""
    if tickers:
        chips = "".join(
            f'<span class="kw-chip kw-chip-accent">{html.escape(str(t))}</span>'
            for t in tickers
        )
        header = (
            '<div class="kw-answer-head">'
            '<span class="kw-answer-label">Answer</span>'
            f"{chips}</div>\n\n"
        )
    return header + highlight_answer(answer)


def render_notice(message: str) -> str:
    return f'<div class="kw-notice">{html.escape(message)}</div>'


LOADING_SOURCES = (
    '<div class="kw-muted">Sources appear alongside the answer.</div>'
)

_STAGES = (
    (0.00, "Understanding your question"),
    (0.15, "Retrieving relevant filings"),
    (0.45, "Reranking excerpts"),
    (0.70, "Writing a cited answer"),
    (0.90, "Verifying citations"),
)


def _progress_fraction(elapsed: float, estimate: float) -> float:
    """Asymptotic progress: tracks the estimate, never claims to be finished.

    Reaches ~63% at the estimated latency and creeps toward 97%, so a slow
    query keeps visibly moving instead of sitting on a full bar.
    """
    estimate = max(estimate, 1.0)
    return 0.97 * (1.0 - math.exp(-elapsed / estimate))


def render_loading(elapsed: float, estimate: float) -> str:
    """Progress card shown inside the answer panel while a query runs."""
    frac = _progress_fraction(elapsed, estimate)
    pct = frac * 100
    stage = _STAGES[0][1]
    stage_idx = 0
    for i, (threshold, label) in enumerate(_STAGES):
        if frac >= threshold:
            stage, stage_idx = label, i

    steps = "".join(
        f'<span class="kw-step{" kw-step-done" if i < stage_idx else ""}'
        f'{" kw-step-on" if i == stage_idx else ""}"></span>'
        for i in range(len(_STAGES))
    )
    return (
        '<div class="kw-loading">'
        '<div class="kw-loading-head">'
        '<span class="kw-spinner" aria-hidden="true"></span>'
        '<div class="kw-loading-text">'
        f'<div class="kw-loading-title">{html.escape(stage)}</div>'
        f'<div class="kw-loading-sub">usually about {estimate:.0f}s '
        f"&middot; {elapsed:.0f}s elapsed</div>"
        "</div>"
        f'<div class="kw-loading-pct">{pct:.0f}%</div>'
        "</div>"
        '<div class="kw-bar" role="progressbar" aria-valuemin="0" '
        f'aria-valuemax="100" aria-valuenow="{pct:.0f}">'
        f'<div class="kw-bar-fill" style="width:{pct:.1f}%"></div>'
        "</div>"
        f'<div class="kw-steps">{steps}</div>'
        '<div class="kw-skeleton">'
        '<div class="kw-sk" style="width:92%"></div>'
        '<div class="kw-sk" style="width:78%"></div>'
        '<div class="kw-sk" style="width:85%"></div>'
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
            try:
                score_html = f'<span class="kw-score">relevance {float(score):.2f}</span>'
            except (TypeError, ValueError):
                score_html = ""

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
    primary_hue="violet",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#f6f6fb",
    body_background_fill_dark="#07070d",
    block_background_fill="transparent",
    block_background_fill_dark="transparent",
    block_border_width="0px",
    block_shadow="none",
    block_shadow_dark="none",
    block_padding="0px",
    input_background_fill="white",
    input_background_fill_dark="#0f0f1a",
    input_border_width="1px",
    input_border_color="rgba(100, 100, 140, .25)",
    input_border_color_dark="rgba(160, 160, 220, .16)",
    input_shadow="none",
    input_shadow_focus="0 0 0 4px rgba(139, 92, 246, .18)",
    button_primary_background_fill="linear-gradient(135deg, #8b5cf6 0%, #6366f1 55%, #06b6d4 120%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #9d78f8 0%, #7578f3 55%, #22c5e0 120%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #8b5cf6 0%, #6366f1 55%, #06b6d4 120%)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #9d78f8 0%, #7578f3 55%, #22c5e0 120%)",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    button_primary_shadow="0 8px 24px -8px rgba(124, 92, 246, .65)",
    button_primary_shadow_dark="0 8px 24px -8px rgba(124, 92, 246, .65)",
    button_secondary_background_fill="transparent",
    button_secondary_background_fill_dark="transparent",
    button_secondary_shadow="none",
)

CUSTOM_CSS = """
:root {
    --kw-bg: #f6f6fb;
    --kw-card: rgba(255, 255, 255, .85);
    --kw-card-solid: #ffffff;
    --kw-ink: #14142b;
    --kw-accent: #7c5cf6;
    --kw-accent-2: #06b6d4;
    --kw-grad: linear-gradient(135deg, #8b5cf6 0%, #6366f1 55%, #06b6d4 120%);
    --kw-line: rgba(90, 90, 140, .16);
    --kw-soft: rgba(124, 92, 246, .07);
    --kw-muted: rgba(70, 70, 110, .78);
    --kw-ok: #10b981;
    --kw-warn: #f59e0b;
    --kw-err: #ef4444;
    --kw-num-bg: rgba(16, 185, 129, .13);
    --kw-num-fg: #047857;
    --kw-pct-bg: rgba(245, 158, 11, .15);
    --kw-pct-fg: #b45309;
    --kw-cite-bg: rgba(124, 92, 246, .14);
    --kw-cite-fg: #6d28d9;
    --kw-glow: rgba(124, 92, 246, .16);
    --kw-shadow: 0 1px 0 rgba(255,255,255,.6) inset, 0 18px 40px -24px rgba(60, 40, 140, .35);
    --kw-gap: 22px;
    --kw-r: 20px;
}
.dark {
    --kw-bg: #07070d;
    --kw-card: rgba(20, 20, 34, .72);
    --kw-card-solid: #12121f;
    --kw-ink: #ececf8;
    --kw-line: rgba(160, 160, 230, .13);
    --kw-soft: rgba(139, 92, 246, .10);
    --kw-muted: rgba(180, 180, 215, .70);
    --kw-num-bg: rgba(52, 211, 153, .13);
    --kw-num-fg: #6ee7b7;
    --kw-pct-bg: rgba(251, 191, 36, .13);
    --kw-pct-fg: #fcd34d;
    --kw-cite-bg: rgba(167, 139, 250, .18);
    --kw-cite-fg: #c4b5fd;
    --kw-glow: rgba(139, 92, 246, .22);
    --kw-shadow: 0 1px 0 rgba(255,255,255,.04) inset, 0 24px 50px -28px rgba(0, 0, 0, .9);
}

footer { display: none !important; }

/* ---------- Page: ambient glow behind a calm, centred column ---------- */
html, body { overflow-x: hidden; }
body {
    background:
        radial-gradient(900px 480px at 12% -8%, var(--kw-glow), transparent 60%),
        radial-gradient(760px 420px at 100% 0%, rgba(6, 182, 212, .10), transparent 60%),
        var(--kw-bg) !important;
    background-attachment: fixed !important;
}
.gradio-container {
    max-width: 1180px !important;
    /* border-box is essential: with content-box the 100vh min-height plus the
       padding overflows the viewport by 62px, which is the top scrollbar. */
    box-sizing: border-box;
    min-height: 100vh;
    margin: 0 auto !important;
    padding: 30px 28px 24px 28px !important;
    display: flex;
    flex-direction: column;
    gap: var(--kw-gap);
    background: transparent !important;
    overflow-x: hidden;
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
/* The title block must never clip or scroll: Gradio's markdown wrapper can
   otherwise show a tiny scrollbar and hide the heading. */
#kw-header, #kw-title, #kw-title *,
#kw-title .prose, #kw-title .md {
    overflow: visible !important;
    max-height: none !important;
}
.kw-h1 {
    font-size: 1.75rem;
    font-weight: 750;
    letter-spacing: -0.04em;
    line-height: 1.25;
    margin: 0;
    padding: 0;
    color: var(--kw-ink);
}
.kw-h1-bar {
    display: block; width: 44px; height: 4px; margin-top: 8px;
    border-radius: 999px; background: var(--kw-grad);
}
#kw-title p { font-size: .88rem; color: var(--kw-muted); margin: 6px 0 0 0; }
#kw-status { text-align: right; }

.kw-pill {
    display: inline-flex; align-items: center; gap: 9px;
    font-size: .78rem; font-weight: 600; color: var(--kw-muted);
    padding: 7px 14px; border-radius: 999px;
    background: var(--kw-card); border: 1px solid var(--kw-line);
    backdrop-filter: blur(10px);
}
.kw-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--kw-warn); }
.kw-pill-loading .kw-dot { animation: kw-pulse 1.2s ease-in-out infinite; }
.kw-pill-ok .kw-dot { background: var(--kw-ok); box-shadow: 0 0 0 4px rgba(16,185,129,.18); }
.kw-pill-err .kw-dot { background: var(--kw-err); }
@keyframes kw-pulse { 50% { opacity: .25; } }

/* Gradio's "processing | 1.2s" overlay must never cover the small widgets. */
#kw-status .wrap, #kw-companies .wrap, #kw-latency .wrap { display: none !important; }
#kw-status, #kw-companies, #kw-latency { min-height: 0; }
#kw-status .pending, #kw-companies .pending, #kw-latency .pending { opacity: 1 !important; }

/* ---------- Body ---------- */
#kw-body {
    flex: 1 1 auto;
    align-items: stretch;
    gap: calc(var(--kw-gap) * 1.2);
    /* No min-height: forcing the row to ~viewport height on top of a container
       that is already 100vh is what pushed the page past one screen. */
    min-height: 0;
}
#kw-main, #kw-side {
    display: flex;
    flex-direction: column;
    gap: var(--kw-gap);
}

/* ---------- Question ---------- */
#kw-question { flex: 0 0 auto; }
#kw-question textarea {
    font-size: 1.02rem;
    line-height: 1.55;
    padding: 18px 20px;
    border-radius: var(--kw-r);
    border: 1px solid var(--kw-line);
    background: var(--kw-card-solid);
    color: var(--kw-ink);
    box-shadow: var(--kw-shadow);
    resize: none;
    transition: border-color .15s ease, box-shadow .15s ease;
}
#kw-question textarea::placeholder { color: var(--kw-muted); }
#kw-question textarea:focus {
    border-color: var(--kw-accent);
    box-shadow: 0 0 0 4px rgba(124, 92, 246, .18), var(--kw-shadow);
}
#kw-question label > span { display: none; }

/* Gradio wraps lone form inputs in a .form whose background is the border
   colour (a 1px divider trick). Stretched by the flex column it shows as a
   slate slab under the textbox, so flatten it and stop it growing. */
.gradio-container .form,
#kw-main .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    flex: 0 0 auto !important;
    min-height: 0 !important;
    height: auto !important;
    gap: 0 !important;
}
#kw-question, #kw-question label, #kw-question .input-container {
    flex: 0 0 auto; min-height: 0; height: auto; background: transparent !important;
}

/* ---------- Toolbar ---------- */
#kw-toolbar { flex: 0 0 auto; align-items: center; gap: 10px; margin-top: -6px; }
#kw-toolbar button {
    flex: 0 0 auto; min-width: 108px;
    font-weight: 650; font-size: .88rem;
    border-radius: 14px; padding: 11px 20px;
    transition: transform .12s ease, box-shadow .12s ease, background .12s ease;
}
#kw-toolbar button.primary { border: none; }
#kw-toolbar button.primary:hover { transform: translateY(-1px); }
#kw-toolbar button.primary:active { transform: translateY(0); }
#kw-toolbar button.secondary {
    border: 1px solid var(--kw-line); color: var(--kw-muted);
    background: var(--kw-card);
}
#kw-toolbar button.secondary:hover { color: var(--kw-ink); background: var(--kw-soft); }
#kw-latency { margin-left: auto; text-align: right; min-width: 70px; }
.kw-lat {
    font-family: var(--font-mono); font-size: .74rem; font-weight: 600;
    color: var(--kw-accent);
    padding: 5px 11px; border-radius: 999px; background: var(--kw-soft);
}

/* ---------- Answer: the flexible, elevated surface ---------- */
/* No card behind the answer: it sits directly on the page, so the chat reads as
   text rather than a grey slab. */
#kw-answer {
    flex: 1 1 auto;
    min-height: 0;
    padding: 14px 4px;
    background: transparent;
    border: none;
    border-radius: 0;
    box-shadow: none;
    backdrop-filter: none;
    color: var(--kw-ink);
    font-size: 1rem;
    line-height: 1.78;
    overflow: auto;
}
#kw-answer p:first-child { margin-top: 0; }
#kw-answer p:last-child { margin-bottom: 0; }
#kw-answer ul, #kw-answer ol { padding-left: 1.3em; }
#kw-answer li { margin: .5em 0; }
#kw-answer li::marker { color: var(--kw-accent); }
#kw-answer table { font-size: .88rem; border-radius: 12px; overflow: hidden; }
#kw-answer strong { color: var(--kw-ink); font-weight: 700; }

/* Highlights: figures, percentages and citation markers. */
mark.kw-num, mark.kw-pct {
    padding: 1px 7px; border-radius: 8px;
    font-weight: 650; font-variant-numeric: tabular-nums;
    -webkit-box-decoration-break: clone; box-decoration-break: clone;
}
mark.kw-num { background: var(--kw-num-bg); color: var(--kw-num-fg); }
mark.kw-pct { background: var(--kw-pct-bg); color: var(--kw-pct-fg); }
.kw-cite {
    display: inline-block; vertical-align: baseline;
    margin: 0 2px; padding: 0 7px; min-width: 20px;
    border-radius: 999px; text-align: center;
    font-size: .72rem; font-weight: 700; line-height: 1.55;
    background: var(--kw-cite-bg); color: var(--kw-cite-fg);
}

.kw-answer-head {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    margin-bottom: 14px; padding-bottom: 14px;
    border-bottom: 1px dashed var(--kw-line);
}
.kw-answer-label {
    font-size: .68rem; font-weight: 750; letter-spacing: .12em;
    text-transform: uppercase; color: var(--kw-accent); margin-right: 4px;
}

/* ---------- Empty state ---------- */
.kw-empty {
    min-height: 0; height: 100%;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 6px; text-align: center; color: var(--kw-muted);
    font-size: .9rem; line-height: 1.55;
}
.kw-empty-icon {
    width: 38px; height: 38px; margin-bottom: 2px;
    display: grid; place-items: center;
    border-radius: 14px; color: white;
    background: var(--kw-grad);
    box-shadow: 0 10px 22px -10px rgba(124, 92, 246, .6);
}
.kw-empty-title { font-weight: 700; font-size: .98rem; color: var(--kw-ink); letter-spacing: -0.01em; }
.kw-empty-sub { max-width: 380px; font-size: .84rem; }
.kw-empty-badges { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-top: 4px; }
.kw-badge {
    font-size: .72rem; font-weight: 650; padding: 5px 12px; border-radius: 999px;
    background: var(--kw-soft); color: var(--kw-accent); border: 1px solid var(--kw-line);
}

/* ---------- Refusal + notice ---------- */
.kw-refusal {
    display: flex; gap: 14px; align-items: flex-start;
    padding: 18px 20px; border-radius: 16px;
    background: rgba(245, 158, 11, .09); border: 1px solid rgba(245, 158, 11, .32);
}
.kw-refusal-icon {
    flex: 0 0 30px; height: 30px; border-radius: 10px; display: grid; place-items: center;
    background: var(--kw-warn); color: #1c1200; font-weight: 800;
}
.kw-refusal b { color: var(--kw-warn); }
.kw-refusal p { margin: 6px 0 0 0; color: var(--kw-muted); }
.kw-notice {
    padding: 14px 18px; border-radius: 14px; font-size: .9rem;
    background: rgba(245, 158, 11, .09);
    border: 1px solid rgba(245, 158, 11, .32);
}

/* ---------- Loading: progress bar, stages, skeleton ---------- */
.kw-loading {
    min-height: 0; height: 100%;
    display: flex; flex-direction: column; justify-content: center; gap: 16px;
    color: var(--kw-muted);
}
.kw-loading-head { display: flex; align-items: center; gap: 14px; }
.kw-spinner {
    flex: 0 0 auto; width: 26px; height: 26px; border-radius: 50%;
    border: 3px solid var(--kw-line);
    border-top-color: var(--kw-accent);
    animation: kw-spin .7s linear infinite;
}
@keyframes kw-spin { to { transform: rotate(360deg); } }
.kw-loading-text { flex: 1; min-width: 0; }
.kw-loading-title { font-weight: 650; font-size: .98rem; color: var(--kw-ink); }
.kw-loading-sub { margin-top: 3px; font-size: .76rem; font-family: var(--font-mono); }
.kw-loading-pct {
    font-family: var(--font-mono); font-size: 1.1rem; font-weight: 700;
    color: var(--kw-accent); font-variant-numeric: tabular-nums;
}
.kw-bar {
    position: relative; height: 10px; border-radius: 999px;
    background: var(--kw-soft); border: 1px solid var(--kw-line); overflow: hidden;
}
.kw-bar-fill {
    position: relative; height: 100%; border-radius: 999px;
    background: var(--kw-grad);
    box-shadow: 0 0 18px rgba(124, 92, 246, .55);
    transition: width .9s cubic-bezier(.22, .8, .3, 1);
    overflow: hidden;
}
.kw-bar-fill::after {
    content: ""; position: absolute; inset: 0;
    background: linear-gradient(100deg, transparent 0%, rgba(255,255,255,.45) 50%, transparent 100%);
    transform: translateX(-100%);
    animation: kw-sheen 1.3s ease-in-out infinite;
}
@keyframes kw-sheen { to { transform: translateX(100%); } }
.kw-steps { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; }
.kw-step { height: 4px; border-radius: 999px; background: var(--kw-line); transition: background .3s ease; }
.kw-step-done { background: var(--kw-accent); }
.kw-step-on { background: var(--kw-accent-2); animation: kw-pulse 1s ease-in-out infinite; }
.kw-skeleton { display: flex; flex-direction: column; gap: 10px; margin-top: 6px; }
.kw-sk {
    height: 12px; border-radius: 8px;
    background: linear-gradient(90deg, var(--kw-soft) 25%, var(--kw-line) 50%, var(--kw-soft) 75%);
    background-size: 200% 100%;
    animation: kw-shimmer 1.4s linear infinite;
}
@keyframes kw-shimmer { to { background-position: -200% 0; } }

/* ---------- Sources ---------- */
/* No grey panel behind the accordion: just a plain toggle row, with each
   source drawn as its own card once opened. */
#kw-sources, #kw-sources > div, #kw-sources .block {
    flex: 0 0 auto;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    backdrop-filter: none !important;
    padding: 0 !important;
}
#kw-sources button.label-wrap, #kw-sources .label-wrap {
    font-weight: 650; font-size: .9rem; color: var(--kw-muted);
    padding: 8px 4px; border: none; background: transparent !important;
}
#kw-sources button.label-wrap:hover, #kw-sources .label-wrap:hover { color: var(--kw-ink); }
.kw-src-list { display: flex; flex-direction: column; gap: 10px; padding: 4px 0 10px 0; max-height: 26vh; overflow-y: auto; }
.kw-src {
    display: flex; gap: 14px; align-items: flex-start;
    padding: 14px 16px;
    border: 1px solid var(--kw-line); border-radius: 16px;
    background: var(--kw-card-solid);
    transition: transform .12s ease, border-color .12s ease;
}
.kw-src:hover { transform: translateY(-1px); border-color: var(--kw-accent); }
.kw-src-n {
    flex: 0 0 26px; height: 26px; border-radius: 9px;
    display: grid; place-items: center;
    font-size: .76rem; font-weight: 750;
    background: var(--kw-cite-bg); color: var(--kw-cite-fg);
}
.kw-src-body { min-width: 0; flex: 1; }
.kw-src-top { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; }
.kw-tag {
    font-size: .74rem; font-weight: 750; letter-spacing: .03em;
    padding: 2px 9px; border-radius: 8px;
    background: var(--kw-grad); color: white;
}
.kw-tag-alt { background: var(--kw-soft); color: var(--kw-accent); font-weight: 650; }
.kw-src-date { font-size: .76rem; color: var(--kw-muted); }
.kw-score {
    margin-left: auto; font-family: var(--font-mono); font-size: .7rem; font-weight: 600;
    padding: 3px 9px; border-radius: 999px;
    background: var(--kw-num-bg); color: var(--kw-num-fg);
}
.kw-src-section { font-size: .88rem; margin-top: 6px; line-height: 1.45; color: var(--kw-ink); }
.kw-link {
    display: inline-block; margin-top: 8px;
    font-size: .78rem; font-weight: 650;
    color: var(--kw-accent); text-decoration: none;
}
.kw-link:hover { text-decoration: underline; }
.kw-muted { font-size: .85rem; color: var(--kw-muted); padding: 10px 6px; }

/* ---------- Sidebar: one card per Group ---------- */
#kw-side { padding-left: 0; gap: 14px; }
/* The card is the Group's outermost element (found by elem_id), so exactly
   one border exists. Every wrapper div Gradio nests inside is flattened;
   the old [class*="group"] selector matched all of them and stacked 3 edges. */
#kw-card-idx, #kw-card-try {
    background: var(--kw-card) !important;
    border: 1px solid var(--kw-line) !important;
    border-radius: 20px !important;
    box-shadow: var(--kw-shadow) !important;
    padding: 16px 18px 18px 18px !important;
    display: flex; flex-direction: column; gap: 12px;
    backdrop-filter: blur(14px);
}
#kw-card-idx div:not([class*="kw-"]),
#kw-card-try div:not([class*="kw-"]) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
    border-radius: 0 !important;
    padding: 0;
    gap: 0;
}
#kw-side h3 {
    font-size: .74rem; font-weight: 750;
    text-transform: uppercase; letter-spacing: .13em;
    color: var(--kw-accent);
    margin: 0;
}
.kw-chips { display: flex; flex-wrap: wrap; gap: 9px; }
.kw-chip {
    font-size: .8rem; font-weight: 650;
    padding: 5px 13px; border-radius: 999px;
    border: 1px solid var(--kw-line);
    background: var(--kw-soft); color: var(--kw-ink);
    transition: transform .12s ease, border-color .12s ease;
}
.kw-chip:hover { transform: translateY(-1px); border-color: var(--kw-accent); }
.kw-chip-accent { background: var(--kw-grad); color: white; border-color: transparent; }

/* Example questions: each row is its own rounded card with an even gap.
   The row (tr / gallery item) carries the border and radius, not the cell, so
   it works whether Gradio renders a table or a list of buttons. Selectors are
   prefixed with #kw-card-try to out-rank the card's flatten-everything reset. */
#kw-card-try .label, #kw-card-try .label-wrap { display: none; }
#kw-card-try table, #kw-card-try tbody,
#kw-card-try .gallery, #kw-card-try .gallery-container {
    display: flex !important; flex-direction: column; gap: 7px;
    width: 100%; border: none !important; background: transparent !important;
    border-collapse: separate; border-spacing: 0;
}
#kw-card-try thead { display: none; }
#kw-card-try tr,
#kw-card-try .gallery-item.gallery-item {
    display: block; width: 100%; box-sizing: border-box;
    cursor: pointer; text-align: left;
    border: 1px solid var(--kw-line) !important;
    border-radius: 14px !important;
    background: var(--kw-card-solid) !important;
    overflow: hidden;
    transition: background .12s ease, border-color .12s ease, transform .12s ease;
}
#kw-card-try td {
    display: block; text-align: left !important;
    padding: 8px 13px !important;
    border: none !important; border-radius: 0 !important;
    background: transparent !important;
    font-size: .84rem; line-height: 1.4;
    color: var(--kw-muted);
    transition: color .12s ease;
}
#kw-card-try .gallery-item.gallery-item {
    padding: 8px 13px; font-size: .84rem; line-height: 1.4; color: var(--kw-muted);
}
#kw-card-try tr:hover,
#kw-card-try .gallery-item.gallery-item:hover {
    background: var(--kw-soft) !important;
    border-color: var(--kw-accent) !important;
    transform: translateX(2px);
}
#kw-card-try tr:hover td,
#kw-card-try .gallery-item.gallery-item:hover { color: var(--kw-ink); }

/* Pinned to the bottom of the sidebar so both columns end on the same line. */
#kw-foot {
    margin-top: auto;
    padding: 14px 16px;
    font-size: .76rem; line-height: 1.55;
    color: var(--kw-muted);
    border: 1px dashed var(--kw-line); border-radius: 16px;
}
#kw-foot p { margin: 0; }

/* ---------- Scrollbars ---------- */
* { scrollbar-width: thin; scrollbar-color: var(--kw-line) transparent; }

/* ---------- Fit the viewport: nothing scrolls except inside panels ---------- */
@media (min-width: 801px) {
    html, body { height: 100%; overflow: hidden !important; }
    .gradio-container {
        height: 100vh; max-height: 100vh;
        min-height: 0 !important;
        padding: 20px 28px 16px 28px !important;
        gap: 14px;
        overflow: hidden;
    }
    #kw-header { padding-bottom: 14px; }
    .kw-h1 { font-size: 1.5rem; }
    #kw-body { flex: 1 1 0; min-height: 0; flex-wrap: nowrap !important; }
    #kw-main, #kw-side { min-height: 0; max-height: 100%; }
    #kw-main { gap: 14px; }
    #kw-answer { flex: 1 1 0; min-height: 0; overflow-y: auto; }
    #kw-side { overflow-y: auto; padding-right: 2px; }
    #kw-foot { margin-top: 0; padding: 10px 14px; }
}

/* ---------- Small screens ---------- */
@media (max-width: 800px) {
    .gradio-container { padding: 20px 16px !important; }
    #kw-header { flex-direction: column; align-items: flex-start; }
    #kw-status { text-align: left; }
    #kw-body { min-height: 0; }
    #kw-answer { padding: 20px; border-radius: 20px; }
}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="10-Kwery", fill_height=True) as demo:
        with gr.Row(elem_id="kw-header"):
            with gr.Column(scale=3, elem_id="kw-title", min_width=0):
                # gr.HTML, not gr.Markdown: Gradio's markdown wrapper can clip or
                # shadow-host the heading, which is what made it disappear.
                gr.HTML(
                    '<h1 class="kw-h1">10-Kwery<span class="kw-h1-bar"></span></h1>',
                    elem_id="kw-title-text",
                )
                gr.Markdown(
                    "Citation-grounded Q&A over SEC 10-K and 10-Q filings"
                )

            with gr.Column(scale=1, min_width=0):
                status = gr.HTML(
                    status_html("loading", "Loading retrieval index…"),
                    elem_id="kw-status",
                )

        with gr.Row(equal_height=True, elem_id="kw-body"):
            with gr.Column(scale=9, min_width=0, elem_id="kw-main"):
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

            with gr.Column(scale=6, min_width=0, elem_id="kw-side"):
                with gr.Group(elem_id="kw-card-idx"):
                    gr.Markdown("### Indexed")
                    companies_box = gr.HTML(
                        '<div class="kw-muted">loading…</div>',
                        elem_id="kw-companies",
                    )

                with gr.Group(elem_id="kw-card-try"):
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
            """Answer one question, streaming a progress card while it runs.

            A generator, so the loading state is emitted before the pipeline
            starts and updated several times a second: a 15s retrieval would
            otherwise leave the panel frozen on the previous answer with no
            sign of life. Gradio streams each yield straight to the components.
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
            # progress bar appears as soon as the button is pressed.
            yield (
                history,
                render_loading(0.0, estimate),
                LOADING_SOURCES,
                gr.update(),
                latency_html("0.0s"),
            )

            while True:
                # Short tick so the bar glides instead of jumping once a second.
                done, _pending = _wait(future, 0.5)
                if done:
                    break
                elapsed = time.monotonic() - started
                yield (
                    history,
                    render_loading(elapsed, estimate),
                    LOADING_SOURCES,
                    gr.update(),
                    latency_html(f"{elapsed:.1f}s"),
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