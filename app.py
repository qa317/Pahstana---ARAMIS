# app.py
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as dtparser

# ----------------------------
# Defaults (these DO return items)
# ----------------------------

DEFAULT_DB = "packaging_trends.db"

DEFAULT_KEYWORDS = [
    "PPWR",
    "packaging waste",
    "sustainable packaging",
    "recycled content",
    "reusable packaging",
    "EPR packaging",
    "deposit return scheme",
    "PFAS packaging",
    "packaging labeling",
]

# Reliable RSS sources (Google News RSS search feeds)
DEFAULT_RSS_FEEDS = [
    "https://news.google.com/rss/search?q=PPWR%20packaging%20EU&hl=en&gl=EU&ceid=EU:en",
    "https://news.google.com/rss/search?q=packaging%20waste%20regulation%20EU&hl=en&gl=EU&ceid=EU:en",
    "https://news.google.com/rss/search?q=sustainable%20packaging%20Europe&hl=en&gl=EU&ceid=EU:en",
    "https://news.google.com/rss/search?q=reusable%20packaging%20Europe&hl=en&gl=EU&ceid=EU:en",
]

DEFAULT_OFFICIAL_URLS = [
    "https://environment.ec.europa.eu/topics/waste-and-recycling/packaging-waste_en",
    "https://eur-lex.europa.eu/eli/reg/2025/40/oj/eng",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PackagingTrendsMonitor/1.0)",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "*/*",
}

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"


# ----------------------------
# DB
# ----------------------------

def db_connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL,
        url TEXT NOT NULL UNIQUE,
        domain TEXT,
        title TEXT,
        published_at TEXT,
        collected_at TEXT NOT NULL,
        text TEXT
    );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_docs_source ON documents(source);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_docs_published ON documents(published_at);")
    con.commit()


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def insert_doc(
    con: sqlite3.Connection,
    source: str,
    url: str,
    title: str,
    published_at: Optional[datetime],
    text: str,
) -> bool:
    url = (url or "").strip()
    if not url:
        return False
    try:
        con.execute(
            """
            INSERT OR IGNORE INTO documents
              (source, url, domain, title, published_at, collected_at, text)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                source,
                url,
                domain_of(url),
                (title or "").strip(),
                published_at.astimezone(timezone.utc).isoformat() if published_at else None,
                datetime.now(timezone.utc).isoformat(),
                (text or "").strip(),
            ),
        )
        con.commit()
        # Check if inserted / exists
        cur = con.execute("SELECT 1 FROM documents WHERE url=? LIMIT 1;", (url,))
        return cur.fetchone() is not None
    except Exception:
        return False


def load_docs(con: sqlite3.Connection, days: int = 90, limit: int = 5000) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    cur = con.execute(
        """
        SELECT source, url, domain, title, published_at, collected_at, text
        FROM documents
        WHERE (published_at IS NULL) OR (published_at >= ?)
        ORDER BY COALESCE(published_at, collected_at) DESC
        LIMIT ?;
        """,
        (since.isoformat(), limit),
    )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["source", "url", "domain", "title", "published_at", "collected_at", "text"])
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["collected_at"] = pd.to_datetime(df["collected_at"], errors="coerce", utc=True)
    return df


# ----------------------------
# Extraction (fast + decent)
# ----------------------------

def fetch(url: str, timeout: int = 20) -> requests.Response:
    return requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)


def html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article")
    target = main if main else soup

    parts: List[str] = []
    for p in target.find_all(["h1", "h2", "h3", "p", "li"]):
        t = p.get_text(" ", strip=True)
        if t and len(t) >= 40:
            parts.append(t)

    text = " ".join(parts) if parts else target.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_date_any(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = dtparser.parse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


# ----------------------------
# GDELT
# ----------------------------

def build_gdelt_query(keywords: List[str]) -> str:
    # Phrases get quoted; single words not quoted. Joined with OR.
    terms = []
    for k in keywords:
        k = k.strip()
        if not k:
            continue
        if " " in k:
            terms.append(f'"{k}"')
        else:
            terms.append(k)
    # If user provides many, keep it reasonable:
    if len(terms) > 15:
        terms = terms[:15]
    return " OR ".join(terms) if terms else "packaging waste EU"


def gdelt_request(query: str, days: int, max_records: int) -> Tuple[List[Dict], Dict]:
    # GDELT doc API behaves best within ~90 days
    days = max(1, min(int(days), 90))

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "sort": "HybridRel",
        "maxrecords": int(max_records),
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }

    debug: Dict = {"endpoint": GDELT_ENDPOINT, "params": params}

    try:
        r = requests.get(GDELT_ENDPOINT, params=params, headers=HEADERS, timeout=25)
        debug["status_code"] = r.status_code
        debug["final_url"] = r.url
        debug["content_type"] = r.headers.get("Content-Type", "")
        debug["preview"] = (r.text or "")[:250].replace("\n", " ")

        if r.status_code != 200:
            return [], debug

        # If not JSON, return []
        ct = (r.headers.get("Content-Type") or "").lower()
        if "json" not in ct and not (r.text or "").lstrip().startswith("{"):
            return [], debug

        data = r.json()
        articles = data.get("articles", []) or []
        return articles, debug
    except Exception as e:
        debug["error"] = repr(e)
        return [], debug


def collect_gdelt(
    con: sqlite3.Connection,
    keywords: List[str],
    days: int,
    max_records: int,
    fetch_full_text: bool,
    max_full_text: int,
) -> Tuple[int, int, int, Dict]:
    """
    Returns: inserted, seen, extracted_ok, debug
    """
    query = build_gdelt_query(keywords)
    articles, debug = gdelt_request(query=query, days=days, max_records=max_records)

    # Fallback if 0 articles: run a simpler query
    if len(articles) == 0:
        fallback_query = "packaging waste EU OR PPWR OR sustainable packaging Europe"
        articles2, debug2 = gdelt_request(query=fallback_query, days=days, max_records=max_records)
        debug["fallback_query"] = fallback_query
        debug["fallback_debug"] = debug2
        articles = articles2

    seen = len(articles)
    inserted = 0
    extracted_ok = 0

    # If you want speed, don’t fetch full text for all results
    full_text_budget = max(0, int(max_full_text))

    for idx, a in enumerate(articles, start=1):
        url = (a.get("url") or "").strip()
        title = (a.get("title") or "").strip()
        seendate = (a.get("seendate") or "").strip()

        published_at = None
        # seendate sometimes is "YYYYMMDDHHMMSS"
        if re.fullmatch(r"\d{14}", seendate):
            try:
                published_at = datetime.strptime(seendate, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            except Exception:
                published_at = None

        text = ""
        if fetch_full_text and idx <= full_text_budget and url:
            try:
                page = fetch(url, timeout=20)
                if page.status_code == 200:
                    text = html_to_text(page.text)
            except Exception:
                text = ""

        # If no full text, store at least the title so you still “have data”
        if not text:
            text = title

        if text:
            extracted_ok += 1

        if insert_doc(con, "GDELT", url, title, published_at, text):
            inserted += 1

    return inserted, seen, extracted_ok, debug


# ----------------------------
# RSS
# ----------------------------

def collect_rss(
    con: sqlite3.Connection,
    feed_urls: List[str],
    fetch_full_text: bool,
    max_per_feed: int,
) -> Tuple[int, int, int]:
    """
    Returns: inserted, entries_seen, extracted_ok
    """
    inserted = 0
    entries_seen = 0
    extracted_ok = 0

    for feed_url in [u.strip() for u in feed_urls if u.strip()]:
        d = feedparser.parse(feed_url)
        entries = d.entries or []
        entries_seen += len(entries)

        for e in entries[: int(max_per_feed)]:
            url = (getattr(e, "link", "") or "").strip()
            title = (getattr(e, "title", "") or "").strip()

            published_at = None
            if getattr(e, "published", None):
                published_at = parse_date_any(getattr(e, "published", ""))
            elif getattr(e, "updated", None):
                published_at = parse_date_any(getattr(e, "updated", ""))

            # Start with summary/description (fast + reliable)
            summary = (getattr(e, "summary", "") or getattr(e, "description", "") or "").strip()
            summary_text = re.sub(r"\s+", " ", BeautifulSoup(summary, "html.parser").get_text(" ", strip=True))

            text = summary_text

            # Optionally fetch article page text (slower, may be blocked)
            if fetch_full_text and url:
                try:
                    page = fetch(url, timeout=20)
                    if page.status_code == 200:
                        full = html_to_text(page.text)
                        if len(full) > len(text):
                            text = full
                except Exception:
                    pass

            if text:
                extracted_ok += 1

            if insert_doc(con, "RSS", url, title, published_at, text):
                inserted += 1

    return inserted, entries_seen, extracted_ok


# ----------------------------
# Official URLs
# ----------------------------

def collect_official(
    con: sqlite3.Connection,
    urls: List[str],
) -> Tuple[int, int]:
    inserted = 0
    extracted_ok = 0
    for url in [u.strip() for u in urls if u.strip()]:
        try:
            r = fetch(url, timeout=25)
            if r.status_code != 200:
                continue
            text = html_to_text(r.text)
            if not text:
                continue
            extracted_ok += 1
            if insert_doc(con, "Official", url, url, None, text):
                inserted += 1
        except Exception:
            continue
    return inserted, extracted_ok


# ----------------------------
# Dashboard analysis (simple + useful)
# ----------------------------

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between
both but by can did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no
nor not of off on once only or other our ours ourselves out over own same she should so some such
than that the their theirs them themselves then there these they this those through to too under
until up very was we were what when where which while who whom why with you your yours yourself yourselves
""".split())


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    toks = re.findall(r"[a-z]{3,}", text)
    return [t for t in toks if t not in STOPWORDS]


def top_terms(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    from collections import Counter
    c = Counter()
    for t in df["text"].fillna("").astype(str):
        c.update(tokenize(t))
    return pd.DataFrame(c.most_common(n), columns=["term", "count"])


def simple_signals(df: pd.DataFrame) -> pd.DataFrame:
    buckets = {
        "Compliance / enforcement": ["compliance", "enforcement", "penalty", "fine", "obligation", "deadline"],
        "Cost / supply": ["cost", "price", "supply", "shortage", "logistics"],
        "Reuse / refill": ["reuse", "reusable", "refill", "return", "deposit"],
        "Recycling / recycled content": ["recycle", "recyclable", "recycled", "content"],
        "Chemicals (PFAS etc.)": ["pfas", "bpa", "chemical"],
        "Labeling / greenwashing": ["label", "labels", "claim", "greenwashing"],
    }

    counts = {k: 0 for k in buckets}
    for _, row in df.iterrows():
        text = f"{row.get('title','')} {row.get('text','')}".lower()
        for b, keys in buckets.items():
            if any(k in text for k in keys):
                counts[b] += 1

    out = pd.DataFrame(sorted(counts.items(), key=lambda x: x[1], reverse=True), columns=["signal", "count"])
    return out[out["count"] > 0]


# ----------------------------
# Streamlit UI (3 tabs)
# ----------------------------

st.set_page_config(page_title="Packaging Trends Monitor", layout="wide")
st.title("📦 Packaging Trends Monitor (Simple)")
st.caption("Collect public packaging trend info (GDELT + RSS + official pages), store in SQLite, explore it, and analyze signals.")

with st.sidebar:
    st.header("Settings")

    db_path = st.text_input("SQLite DB path", DEFAULT_DB, help="Created automatically if it doesn't exist.")
    con = db_connect(db_path)
    init_db(con)

    st.subheader("Collect settings")
    days_collect = st.slider("Lookback days (GDELT)", 1, 90, 14)
    max_records = st.slider("Max GDELT records", 10, 300, 120, step=10)

    keywords_text = st.text_area("Keywords (one per line)", "\n".join(DEFAULT_KEYWORDS), height=160)
    keywords = [k.strip() for k in keywords_text.splitlines() if k.strip()]

    rss_text = st.text_area("RSS feeds (one per line)", "\n".join(DEFAULT_RSS_FEEDS), height=160)
    rss_feeds = [u.strip() for u in rss_text.splitlines() if u.strip()]

    official_text = st.text_area("Official URLs (one per line)", "\n".join(DEFAULT_OFFICIAL_URLS), height=120)
    official_urls = [u.strip() for u in official_text.splitlines() if u.strip()]

    st.divider()
    st.subheader("Speed options")
    gdelt_full_text = st.checkbox("GDELT: fetch full article text (slower)", value=False)
    gdelt_full_text_n = st.slider("GDELT full-text limit", 0, 60, 20, help="Only first N GDELT articles will be downloaded for text.")
    rss_full_text = st.checkbox("RSS: fetch full article text (slower)", value=False)
    rss_max_per_feed = st.slider("RSS max items per feed", 5, 60, 25)


tab_collect, tab_explore, tab_analyze = st.tabs(["Collect data", "Explore data", "Analyze (Dashboard)"])


with tab_collect:
    st.subheader("Collect data")

    col1, col2, col3 = st.columns(3)
    use_gdelt = col1.checkbox("Collect from GDELT (news)", value=True)
    use_rss = col2.checkbox("Collect from RSS feeds", value=True)
    use_official = col3.checkbox("Collect from official pages", value=True)

    if st.button("🚀 Run collection", type="primary"):
        debug_holder = None

        total_inserted = 0

        if use_gdelt:
            with st.spinner("Collecting from GDELT…"):
                ins, seen, extracted_ok, debug = collect_gdelt(
                    con, keywords, days_collect, max_records,
                    fetch_full_text=gdelt_full_text,
                    max_full_text=gdelt_full_text_n,
                )
                total_inserted += ins
                st.success(f"✅ GDELT seen {seen} | extracted {extracted_ok} | inserted {ins}")
                with st.expander("GDELT debug (click if you still see 0)"):
                    st.json(debug)

        if use_rss:
            with st.spinner("Collecting from RSS…"):
                ins, entries_seen, extracted_ok = collect_rss(
                    con, rss_feeds,
                    fetch_full_text=rss_full_text,
                    max_per_feed=rss_max_per_feed,
                )
                total_inserted += ins
                st.success(f"✅ RSS entries seen {entries_seen} | extracted {extracted_ok} | inserted {ins}")

        if use_official:
            with st.spinner("Collecting from official pages…"):
                ins, extracted_ok = collect_official(con, official_urls)
                total_inserted += ins
                st.success(f"✅ Official extracted {extracted_ok} | inserted {ins}")

        if total_inserted == 0:
            st.warning(
                "No new documents inserted. Open **GDELT debug** above to see status/preview, "
                "and try keeping only 2–3 keywords (e.g., 'PPWR', 'packaging waste', 'sustainable packaging')."
            )

    st.info("Tip: For fastest results, keep RSS full-text OFF. You’ll still get summaries, which is enough for analysis.")


with tab_explore:
    st.subheader("Explore data")
    days_view = st.slider("Show data from last N days", 7, 365, 90)

    df = load_docs(con, days=days_view)
    if df.empty:
        st.info("No data yet. Go to **Collect data** first.")
    else:
        sources = ["All"] + sorted(df["source"].dropna().unique().tolist())
        pick_source = st.selectbox("Filter by source", sources, index=0)
        q = st.text_input("Search in title/text", value="")

        view = df.copy()
        if pick_source != "All":
            view = view[view["source"] == pick_source]
        if q.strip():
            qq = q.strip().lower()
            view = view[
                view["title"].fillna("").str.lower().str.contains(qq)
                | view["text"].fillna("").str.lower().str.contains(qq)
            ]

        show = view.copy()
        show["published_at"] = show["published_at"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
        show["collected_at"] = show["collected_at"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
        st.dataframe(
            show[["source", "published_at", "domain", "title", "url"]],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download CSV",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="packaging_trends_export.csv",
            mime="text/csv",
        )


with tab_analyze:
    st.subheader("Analyze (Dashboard)")
    days_dash = st.slider("Analyze last N days", 7, 365, 90)

    df = load_docs(con, days=days_dash)
    if df.empty:
        st.info("No data yet. Collect first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents", f"{len(df):,}")
        c2.metric("Sources", f"{df['source'].nunique()}")
        c3.metric("Domains", f"{df['domain'].nunique()}")
        c4.metric("With text", f"{int((df['text'].fillna('').str.len() > 0).sum())}")

        st.write("### Documents over time")
        tmp = df.copy()
        tmp["date"] = (tmp["published_at"].fillna(tmp["collected_at"])).dt.date
        daily = tmp.groupby("date").size().rename("count").to_frame()
        st.line_chart(daily)

        st.write("### By source")
        by_source = df["source"].value_counts().rename_axis("source").reset_index(name="count")
        st.bar_chart(by_source.set_index("source")["count"])

        st.write("### Top terms (what people talk about most)")
        terms = top_terms(df, n=25)
        st.dataframe(terms, use_container_width=True)

        st.write("### Signals (concerns / expectations buckets)")
        sig = simple_signals(df)
        if sig.empty:
            st.info("No signals detected yet (try collecting more items).")
        else:
            st.dataframe(sig, use_container_width=True)
            st.bar_chart(sig.set_index("signal")["count"])

        st.write("### Recent items (quick view)")
        recent = df.head(20).copy()
        recent["published_at"] = recent["published_at"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
        st.dataframe(recent[["source", "published_at", "title", "url"]], use_container_width=True, hide_index=True)
