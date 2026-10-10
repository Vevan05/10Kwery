---
title: 10Kwery
emoji: 🔍
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
python_version: "3.11"
pinned: false
---

# 10-Kwery

A retrieval-augmented question answering demo over SEC 10-K and 10-Q filings for
seven large-cap companies (AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA).

Ask a question in natural language. The system:

1. encodes the query with **BAAI/bge-small-en-v1.5**
2. retrieves from four channels - dense and BM25 over all chunks, plus dense and
   BM25 over tables only
3. fuses the rankings with Reciprocal Rank Fusion
4. reranks with **cross-encoder/ms-marco-MiniLM-L-6-v2**
5. balances table and prose slots in the final five excerpts
6. generates a citation-grounded answer with **openai/gpt-oss-120b** via Groq

Every answer cites the excerpts it came from, and the source list links straight
to the filing on SEC EDGAR so any claim can be verified.

## API

`POST /api/query` takes `{"question": "...", "history": [...]}` and returns the
answer plus a structured source list:

```bash
curl -X POST https://<user>-10kwery.hf.space/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What drove Tesla automotive revenue growth in 2024?"}'
```

Other endpoints: `GET /health` (readiness) and `GET /api/companies`
(indexed tickers).

## Notes

- Runs on the free CPU Basic tier: 2 vCPU, 16 GB RAM. Cold starts after the
  Space sleeps re-download the 277 MB retrieval index, so the first request
  after a nap takes a minute or two.
- Answers are grounded strictly in the retrieved excerpts. Questions about
  companies outside the indexed corpus are refused rather than answered from
  outside knowledge.
- The LLM runs on the Groq free tier, so requests are rate limited per client.
