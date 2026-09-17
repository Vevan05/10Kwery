import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

load_dotenv()

USER_AGENT: str = os.getenv("SEC_USER_AGENT") or ""
if not USER_AGENT:
    raise RuntimeError(
        "SEC_USER_AGENT not set. Add it to your .env file, e.g.\n"
        '  SEC_USER_AGENT="Your Name your.email@example.com"'
    )

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
FORMS_WANTED = {"10-K", "10-Q"}
YEARS_BACK = 3
OUTPUT_DIR = Path("data/filings")


class EdgarClient:
    def __init__(self, user_agent : str, min_interval: float = 0.11):
        self.min_interval = min_interval
        self._last_call = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        retries = Retry(
                    total = 5,
                    backoff_factor = 0.5,
                    status_forcelist = [429, 500, 502, 503, 504]
        )

        self.session.mount("https://", HTTPAdapter(max_retries = retries))

    def _request(self, url: str) -> requests.Response:

        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        resp = self.session.get(url, timeout=15)
        self._last_call = time.time()
        resp.raise_for_status()
        return resp


    def _get_json(self, url: str) -> dict:
        return self._request(url).json()

    def _get_text(self, url: str) -> str:
        return self._request(url).text


    def get_ticker_cik_map(self) -> dict:
        data = self._get_json("https://www.sec.gov/files/company_tickers.json")
        return {
            row["ticker"].upper(): str(row["cik_str"]).zfill(10)
            for row in data.values()
        }

    def get_submissions(self, cik: str) -> dict:
        return self._get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")

    def get_document(self, cik_no_zeros: str, accession_no_dashes: str, primary_doc: str) -> str:
        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_no_zeros}/{accession_no_dashes}/{primary_doc}"
        )
        return self._get_text(url)


def ingest_company(client: EdgarClient, ticker: str, cik: str, cutoff: datetime) -> int:
    submissions = client.get_submissions(cik)
    recent = submissions["filings"]["recent"]

    rows = list(
        zip(
            recent["form"],
            recent["filingDate"],
            recent["accessionNumber"],
            recent["primaryDocument"]
        )
    )

    company_dir = OUTPUT_DIR / ticker
    company_dir.mkdir(parents=True, exist_ok=True)
    cik_no_zeros = str(int(cik))

    downloaded = 0
    for form, filing_date, accession, primary_doc in rows:
        if form not in FORMS_WANTED:
            continue

        if datetime.strptime(filing_date, "%Y-%m-%d") < cutoff:
            continue

        if not primary_doc:
            continue

        accession_no_dashes = accession.replace("-", "")
        out_path = company_dir/ f"{form}_{filing_date}_{accession}.htm"

        if out_path.exists():
            continue

        html = client.get_document(cik_no_zeros, accession_no_dashes, primary_doc)
        out_path.write_text(html, encoding = "utf-8")

        meta = {
            "ticker": ticker,
            "cik": cik,
            "company_name": submissions.get("name"),
            "form": form,
            "filing_date": filing_date,
            "accession_number": accession,
            "source_url": (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_no_zeros}/{accession_no_dashes}/{primary_doc}"
            )
        }

        out_path.with_suffix(".json").write_text(json.dumps(meta, indent = 2))
        downloaded += 1

    return downloaded


def main():
    client = EdgarClient(USER_AGENT)
    cutoff = datetime.now() - timedelta(days = 365 * YEARS_BACK)

    print("Resolving tickers into CIKs")
    ticker_to_cik = client.get_ticker_cik_map()

    total_downloaded = 0

    for ticker in tqdm(TICKERS, desc="Companies"):
        cik = ticker_to_cik.get(ticker)
        if not cik:
            print(f"WARNING: No CIK found for {ticker}... skipping")
            continue

        count = ingest_company(client, ticker, cik, cutoff)
        total_downloaded += count
        tqdm.write(f"{ticker}: {count} new filings downloaded")

    print(f"\nDone. {total_downloaded} new filings saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
