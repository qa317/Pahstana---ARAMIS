# app.py (UPDATED for Streamlit Cloud)
# ✅ Adds Reddit collection (via Reddit RSS — no API keys needed)
# ✅ Adds Sentiment Analysis (VADER via vaderSentiment — lightweight, Streamlit Cloud friendly)
# ✅ Improves analysis: sentiment over time, sentiment by source type, key insights summary
# ✅ Keeps: GDELT + RSS + GNews + Official pages + Emerging terms + Signal buckets
#
# requirements.txt (recommended for Streamlit Cloud):
# streamlit
# requests
# feedparser
# pandas
# beautifulsoup4
# python-dateutil
# vaderSentiment
# gnews
#
# Run locally:
#   pip install -r requirements.txt
#   streamlit run app.py

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlsplit, urlunsplit, parse_qsl, urlencode, quote_plus

import pandas as pd
import requests
import streamlit as st
import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as dtparser

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional: GNews library
try:
    from gnews import GNews  # type: ignore
except Exception:
    GNews = None

# ----------------------------
# Defaults
# ----------------------------

DEFAULT_DB = "packaging_trends.db"
GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

DEFAULT_KEYWORDS = [
    "packaging waste",
    "PPWR",
    "packaging and packaging waste regulation",
    "sustainable packaging",
    "recyclable packaging",
    "recycled content packaging",
    "reusable packaging",
    "EPR packaging",
    "deposit return scheme",
    "PFAS packaging",
    "packaging labeling",
    "greenwashing packaging",
]

# Google News RSS queries + diversified direct feeds
DEFAULT_RSS_FEEDS = [
    # --- Google News RSS (broad coverage) ---
    "https://news.google.com/rss/search?q=packaging%20waste%20EU&hl=en&gl=FR&ceid=FR:en",
    "https://news.google.com/rss/search?q=PPWR%20packaging%20waste%20regulation&hl=en&gl=FR&ceid=FR:en",
    "https://news.google.com/rss/search?q=sustainable%20packaging%20Europe&hl=en&gl=FR&ceid=FR:en",
    "https://news.google.com/rss/search?q=reusable%20packaging%20Europe&hl=en&gl=FR&ceid=FR:en",
    "https://news.google.com/rss/search?q=EPR%20packaging%20Europe&hl=en&gl=FR&ceid=FR:en",
    "https://news.google.com/rss/search?q=PFAS%20packaging%20Europe&hl=en&gl=FR&ceid=FR:en",
    # --- Direct / sector feeds (diversification) ---
    "https://packagingeurope.com/feed/",
    "https://zerowasteeurope.eu/feed/",
    "https://www.ellenmacarthurfoundation.org/feeds/latest",
    "https://ec.europa.eu/commission/presscorner/api/rss?language=en",
]

# Expanded institutional watchlist
DEFAULT_OFFICIAL_URLS = [
    "https://environment.ec.europa.eu/topics/waste-and-recycling/packaging-waste_en",
    "https://eur-lex.europa.eu/eli/reg/2025/40/oj/eng",
    "https://www.eea.europa.eu/themes/waste",
    "https://echa.europa.eu/hot-topics/perfluoroalkyl-chemicals-pfas",
    "https://www.europarl.europa.eu/topics/en/article/20190110STO23109/waste-and-recycling",
]

# Reddit RSS defaults (no API keys needed)
DEFAULT_REDDIT_SUBREDDITS = [
    "sustainability",
    "environment",
    "ZeroWaste",
    "packaging",
    "circularEconomy",
    "europe",
]
DEFAULT_REDDIT_QUERY = "packaging OR PPWR OR reusable packaging OR recyclable packaging OR PFAS packaging OR EPR packaging"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; GreenPackAnalytics/1.0; +https://example.com)",
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.7",
    "Accept": "*/*",
}

# ----------------------------
# URL normalization / Dedupe
# ----------------------------

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "ref", "src", "spm", "ito"
}

def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    try:
        parts = urlsplit(url)
        query = [
            (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() not in TRACKING_PARAMS
        ]
        clean = parts._replace(query=urlencode(query, doseq=True), fragment="")
        return urlunsplit(clean)
    except Exception:
        return url

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def norm_title(title: str) -> str:
    t = (title or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t

def title_domain_hash(title: str, domain: str) -> str:
    key = f"{norm_title(title)}|{(domain or '').lower()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

# ----------------------------
# Source classification (credibility buckets)
# ----------------------------

INSTITUTIONAL_DOMAINS = {
    "eur-lex.europa.eu",
    "environment.ec.europa.eu",
    "ec.europa.eu",
    "echa.europa.eu",
    "www.europarl.europa.eu",
    "www.eea.europa.eu",
    "european-union.europa.eu",
}
MAINSTREAM_MEDIA_HINTS = {
    "reuters.com", "bbc.co.uk", "lemonde.fr", "ft.com", "theguardian.com",
    "politico.eu", "euractiv.com", "bloomberg.com"
}
TRADE_MEDIA_HINTS = {
    "packagingeurope.com", "packworld.com", "packagingdigest.com",
    "sustainablebrands.com"
}
NGO_HINTS = {
    "zerowasteeurope.eu", "ellenmacarthurfoundation.org", "wwf.org", "greenpeace.org"
}

def classify_source_type(source: str, url: str) -> str:
    dom = domain_of(url)
    if source == "Official" or dom in INSTITUTIONAL_DOMAINS or dom.endswith(".europa.eu"):
        return "Institutional"
    if source == "Reddit":
        return "Online discourse"
    if any(h in dom for h in MAINSTREAM_MEDIA_HINTS):
        return "Mainstream media"
    if any(h in dom for h in TRADE_MEDIA_HINTS):
        return "Trade media"
    if any(h in dom for h in NGO_HINTS):
        return "NGO / civil society"
    return "Other"

# ----------------------------
# Sentiment (VADER)
# ----------------------------

SENTI = SentimentIntensityAnalyzer()

def sentiment_compound(text: str) -> float:
    text = (text or "").strip()
    if not text:
        return 0.0
    return float(SENTI.polarity_scores(text)["compound"])

def sentiment_label(comp: float) -> str:
    if comp >= 0.05:
        return "Positive"
    if comp <= -0.05:
        return "Negative"
    return "Neutral"

# ----------------------------
# DB
# ----------------------------

def db_connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def _try_exec(con: sqlite3.Connection, sql: str) -> None:
    try:
        con.execute(sql)
        con.commit()
    except Exception:
        pass

def init_db(con: sqlite3.Connection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL,
        source_type TEXT,
        url TEXT NOT NULL,
        url_norm TEXT NOT NULL UNIQUE,
        domain TEXT,
        title TEXT,
        title_dom_hash TEXT,
        published_at TEXT,
        collected_at TEXT NOT NULL,
        text TEXT,
        sentiment_compound REAL,
        sentiment_label TEXT
    );
    """)
    _try_exec(con, "CREATE UNIQUE INDEX IF NOT EXISTS uq_docs_url_norm ON documents(url_norm);")
    _try_exec(con, "CREATE INDEX IF NOT EXISTS idx_docs_source ON documents(source);")
    _try_exec(con, "CREATE INDEX IF NOT EXISTS idx_docs_source_type ON documents(source_type);")
    _try_exec(con, "CREATE INDEX IF NOT EXISTS idx_docs_published ON documents(published_at);")
    _try_exec(con, "CREATE INDEX IF NOT EXISTS idx_docs_domain ON documents(domain);")
    _try_exec(con, "CREATE INDEX IF NOT EXISTS idx_docs_titlehash ON documents(title_dom_hash);")
    _try_exec(con, "CREATE INDEX IF NOT EXISTS idx_docs_sentiment ON documents(sentiment_compound);")

    # Migration for older DBs:
    _try_exec(con, "ALTER TABLE documents ADD COLUMN source_type TEXT;")
    _try_exec(con, "ALTER TABLE documents ADD COLUMN url_norm TEXT;")
    _try_exec(con, "ALTER TABLE documents ADD COLUMN title_dom_hash TEXT;")
    _try_exec(con, "ALTER TABLE documents ADD COLUMN sentiment_compound REAL;")
    _try_exec(con, "ALTER TABLE documents ADD COLUMN sentiment_label TEXT;")
    con.commit()

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

    url_norm = normalize_url(url)
    dom = domain_of(url_norm)
    s_type = classify_source_type(source, url_norm)
    t_hash = title_domain_hash(title, dom)

    # Sentiment on combined text (headline + body/summary)
    combo = f"{title}. {text}".strip()
    comp = sentiment_compound(combo)
    label = sentiment_label(comp)

    # Secondary dedupe: same title+domain
    try:
        cur = con.execute("SELECT 1 FROM documents WHERE title_dom_hash=? LIMIT 1;", (t_hash,))
        if cur.fetchone() is not None:
            return True
    except Exception:
        pass

    try:
        con.execute(
            """
            INSERT OR IGNORE INTO documents
              (source, source_type, url, url_norm, domain, title, title_dom_hash, published_at, collected_at, text,
               sentiment_compound, sentiment_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                source,
                s_type,
                url,
                url_norm,
                dom,
                (title or "").strip(),
                t_hash,
                published_at.astimezone(timezone.utc).isoformat() if published_at else None,
                datetime.now(timezone.utc).isoformat(),
                (text or "").strip(),
                comp,
                label,
            ),
        )
        con.commit()
        cur = con.execute("SELECT 1 FROM documents WHERE url_norm=? LIMIT 1;", (url_norm,))
        return cur.fetchone() is not None
    except Exception:
        return False

def load_docs(con: sqlite3.Connection, days: int = 180, limit: int = 5000) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    cur = con.execute(
        """
        SELECT source, source_type, url, url_norm, domain, title, published_at, collected_at, text,
               sentiment_compound, sentiment_label
        FROM documents
        WHERE (published_at IS NULL) OR (published_at >= ?)
        ORDER BY COALESCE(published_at, collected_at) DESC
        LIMIT ?;
        """,
        (since.isoformat(), limit),
    )
    rows = cur.fetchall()
    df = pd.DataFrame(
        rows,
        columns=[
            "source", "source_type", "url", "url_norm", "domain", "title",
            "published_at", "collected_at", "text", "sentiment_compound", "sentiment_label"
        ],
    )
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["collected_at"] = pd.to_datetime(df["collected_at"], errors="coerce", utc=True)
    df["sentiment_compound"] = pd.to_numeric(df["sentiment_compound"], errors="coerce").fillna(0.0)
    df["sentiment_label"] = df["sentiment_label"].fillna("Neutral")
    return df

# ----------------------------
# Extraction
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
    return re.sub(r"\s+", " ", text).strip()

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
    terms = []
    for k in keywords:
        k = k.strip()
        if not k:
            continue
        terms.append(f'"{k}"' if " " in k else k)
    if len(terms) > 12:
        terms = terms[:12]
    return " OR ".join(terms) if terms else "packaging waste EU"

def gdelt_call(params: Dict) -> Tuple[List[Dict], Dict]:
    debug: Dict = {"endpoint": GDELT_ENDPOINT, "params": params}
    r = requests.get(GDELT_ENDPOINT, params=params, headers=HEADERS, timeout=25)
    debug["status_code"] = r.status_code
    debug["final_url"] = r.url
    debug["content_type"] = r.headers.get("Content-Type", "")
    debug["preview"] = (r.text or "")[:250].replace("\n", " ")

    if r.status_code != 200:
        return [], debug

    ct = (r.headers.get("Content-Type") or "").lower()
    if "json" not in ct and not (r.text or "").lstrip().startswith("{"):
        return [], debug

    data = r.json()
    return (data.get("articles", []) or []), debug

def gdelt_request(query: str, days: int, max_records: int) -> Tuple[List[Dict], Dict]:
    days = max(1, min(int(days), 90))
    end = datetime.now(timezone.utc)

    start = end - timedelta(days=days)
    p1 = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "sort": "HybridRel",
        "maxrecords": int(max_records),
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    a1, d1 = gdelt_call(p1)
    if len(a1) > 0:
        return a1, d1

    if days < 90:
        start90 = end - timedelta(days=90)
        p2 = dict(p1)
        p2["startdatetime"] = start90.strftime("%Y%m%d%H%M%S")
        p2["enddatetime"] = end.strftime("%Y%m%d%H%M%S")
        a2, d2 = gdelt_call(p2)
        d1["retry_90_days"] = d2
        if len(a2) > 0:
            return a2, d1

    p3 = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "sort": "HybridRel",
        "maxrecords": int(max_records),
    }
    a3, d3 = gdelt_call(p3)
    d1["retry_no_dates"] = d3
    return a3, d1

def collect_gdelt(
    con: sqlite3.Connection,
    keywords: List[str],
    days: int,
    max_records: int,
    fetch_full_text: bool,
    max_full_text: int,
) -> Tuple[int, int, int, Dict]:
    query = build_gdelt_query(keywords)
    articles, debug = gdelt_request(query=query, days=days, max_records=max_records)

    if len(articles) == 0:
        fallback_query = "packaging waste EU OR PPWR OR sustainable packaging Europe OR EPR packaging"
        a2, d2 = gdelt_request(query=fallback_query, days=days, max_records=max_records)
        debug["fallback_query"] = fallback_query
        debug["fallback_debug"] = d2
        articles = a2

    seen = len(articles)
    inserted = 0
    extracted_ok = 0
    budget = max(0, int(max_full_text))

    for idx, a in enumerate(articles, start=1):
        url = (a.get("url") or "").strip()
        title = (a.get("title") or "").strip()
        seendate = (a.get("seendate") or "").strip()

        published_at = None
        if re.fullmatch(r"\d{14}", seendate):
            try:
                published_at = datetime.strptime(seendate, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            except Exception:
                published_at = None

        text = title
        if fetch_full_text and idx <= budget and url:
            try:
                page = fetch(url, timeout=20)
                if page.status_code == 200:
                    full = html_to_text(page.text)
                    if full:
                        text = full
            except Exception:
                pass

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
) -> Tuple[int, int, int, int]:
    inserted = 0
    feeds_ok = 0
    entries_seen = 0
    extracted_ok = 0

    for feed_url in [u.strip() for u in feed_urls if u.strip()]:
        try:
            d = feedparser.parse(feed_url)
            if not getattr(d, "entries", None):
                continue
            feeds_ok += 1
            entries = d.entries or []
            entries_seen += len(entries)
        except Exception:
            continue

        for e in entries[: int(max_per_feed)]:
            url = (getattr(e, "link", "") or "").strip()
            title = (getattr(e, "title", "") or "").strip()

            published_at = None
            if getattr(e, "published", None):
                published_at = parse_date_any(getattr(e, "published", ""))
            elif getattr(e, "updated", None):
                published_at = parse_date_any(getattr(e, "updated", ""))

            summary_html = (getattr(e, "summary", "") or getattr(e, "description", "") or "").strip()
            summary_text = re.sub(
                r"\s+",
                " ",
                BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True),
            )

            text = summary_text or title

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

    return inserted, feeds_ok, entries_seen, extracted_ok

# ----------------------------
# Reddit (via RSS)
# ----------------------------

def build_reddit_rss(subreddit: str, query: str, sort: str = "new") -> str:
    # Example:
    # https://www.reddit.com/r/sustainability/search.rss?q=packaging&restrict_sr=1&sort=new
    return (
        f"https://www.reddit.com/r/{subreddit}/search.rss?"
        f"q={quote_plus(query)}&restrict_sr=1&sort={sort}&t=all"
    )

def collect_reddit_rss(
    con: sqlite3.Connection,
    subreddits: List[str],
    query: str,
    max_per_sub: int,
) -> Tuple[int, int, int]:
    inserted = 0
    entries_seen = 0
    extracted_ok = 0

    for sr in [s.strip() for s in subreddits if s.strip()]:
        rss_url = build_reddit_rss(sr, query=query, sort="new")
        try:
            d = feedparser.parse(rss_url)
            entries = d.entries or []
            entries_seen += len(entries)
        except Exception:
            continue

        for e in entries[: int(max_per_sub)]:
            url = (getattr(e, "link", "") or "").strip()
            title = (getattr(e, "title", "") or "").strip()

            published_at = None
            if getattr(e, "published", None):
                published_at = parse_date_any(getattr(e, "published", ""))

            # RSS summary often contains HTML; keep it as "discourse snippet"
            summary_html = (getattr(e, "summary", "") or "").strip()
            summary_text = re.sub(
                r"\s+",
                " ",
                BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True),
            )

            # Keep it short, but informative (avoid huge blobs)
            text = summary_text[:2000] if summary_text else title

            if text:
                extracted_ok += 1
            if insert_doc(con, "Reddit", url, title, published_at, text):
                inserted += 1

    return inserted, entries_seen, extracted_ok

# ----------------------------
# GNews library (optional)
# ----------------------------

def collect_gnews(
    con: sqlite3.Connection,
    keywords: List[str],
    lookback_days: int,
    country: str,
    language: str,
    max_results: int,
) -> Tuple[int, int]:
    if GNews is None:
        return 0, 0

    today_dt = datetime.now()
    start_dt = today_dt - timedelta(days=int(lookback_days))

    gn = GNews(
        language=language,
        country=country,
        start_date=start_dt,
        end_date=today_dt,
        max_results=int(max_results),
    )

    topics = keywords[:6] if keywords else ["packaging waste", "PPWR", "sustainable packaging"]
    inserted = 0
    seen = 0

    for topic in topics:
        try:
            news = gn.get_news(topic)
        except Exception:
            continue

        for a in news:
            seen += 1
            title = (a.get("title") or "").strip()
            url = (a.get("url") or "").strip()
            desc = (a.get("description") or "").strip()
            published = a.get("published date") or a.get("published_date") or ""
            published_at = parse_date_any(str(published)) if published else None

            text = (desc or title)
            if insert_doc(con, "GNews", url, title, published_at, text):
                inserted += 1

    return inserted, seen

# ----------------------------
# Official pages
# ----------------------------

def collect_official(con: sqlite3.Connection, urls: List[str]) -> Tuple[int, int]:
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
# Analysis helpers
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
        "Cost / supply": ["cost", "price", "supply", "shortage", "logistics", "inflation"],
        "Reuse / refill": ["reuse", "reusable", "refill", "return", "deposit"],
        "Recycling / recycled content": ["recycle", "recyclable", "recycled", "content"],
        "Chemicals (PFAS etc.)": ["pfas", "bpa", "chemical"],
        "Labeling / greenwashing": ["label", "labels", "claim", "greenwashing", "misleading"],
        "Targets / bans": ["ban", "banned", "target", "quota", "mandatory"],
    }
    counts = {k: 0 for k in buckets}
    for _, row in df.iterrows():
        text = f"{row.get('title','')} {row.get('text','')}".lower()
        for b, keys in buckets.items():
            if any(k in text for k in keys):
                counts[b] += 1
    out = pd.DataFrame(sorted(counts.items(), key=lambda x: x[1], reverse=True), columns=["signal", "count"])
    return out[out["count"] > 0]

def emerging_terms(df: pd.DataFrame, window_days: int = 30, top_n: int = 15, min_recent: int = 3) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    ts = df["published_at"].fillna(df["collected_at"])
    now = ts.max()
    if pd.isna(now):
        return pd.DataFrame()

    recent_start = now - pd.Timedelta(days=window_days)
    prev_start = now - pd.Timedelta(days=2 * window_days)

    recent = df[ts >= recent_start]
    prev = df[(ts < recent_start) & (ts >= prev_start)]

    from collections import Counter
    cr, cp = Counter(), Counter()
    for t in recent["text"].fillna("").astype(str):
        cr.update(tokenize(t))
    for t in prev["text"].fillna("").astype(str):
        cp.update(tokenize(t))

    rows = []
    for term, rc in cr.items():
        pc = cp.get(term, 0)
        if rc >= min_recent:
            rows.append((term, rc, pc, rc - pc))
    out = pd.DataFrame(rows, columns=["term", "recent", "previous", "delta"]).sort_values("delta", ascending=False)
    return out.head(top_n)

def build_key_insights(df: pd.DataFrame) -> List[str]:
    """Create simple, defensible 'insight bullets' from the dashboard metrics."""
    if df.empty:
        return ["No data collected yet. Run collection first."]

    bullets: List[str] = []

    # 1) Source type balance
    by_type = df["source_type"].fillna("Other").value_counts()
    top_type = by_type.index[0]
    top_type_share = (by_type.iloc[0] / len(df)) * 100
    bullets.append(f"Coverage is dominated by **{top_type}** sources (~{top_type_share:.0f}% of documents).")

    # 2) Signals
    sig = simple_signals(df)
    if not sig.empty:
        top_sig = sig.iloc[0]
        bullets.append(f"Top recurring theme: **{top_sig['signal']}** (mentions: {int(top_sig['count'])}).")

    # 3) Emerging terms
    emerg = emerging_terms(df, window_days=30, top_n=5, min_recent=3)
    if not emerg.empty:
        rising = ", ".join([f"{r.term} (+{int(r.delta)})" for r in emerg.itertuples(index=False)])
        bullets.append(f"Emerging terms (last 30d vs previous 30d): {rising}.")

    # 4) Sentiment summary
    avg_sent = df["sentiment_compound"].mean()
    bullets.append(f"Overall tone is **{sentiment_label(avg_sent)}** (avg VADER compound: {avg_sent:.2f}).")

    # 5) Sentiment differences by source type
    if df["source_type"].nunique() >= 2:
        g = df.groupby("source_type")["sentiment_compound"].mean().sort_values()
        low = g.index[0]
        high = g.index[-1]
        bullets.append(
            f"Tone differs by source type: lowest average in **{low}** ({g.iloc[0]:.2f}), highest in **{high}** ({g.iloc[-1]:.2f})."
        )

    # 6) Practical implication framing
    bullets.append(
        "Interpretation note: sentiment reflects **language in sources** (headlines/snippets), not objective policy outcomes; use it to detect **concerns and expectations**, not to forecast."
    )
    return bullets

# ============================
# Streamlit UI
# ============================

st.set_page_config(page_title="GreenPack Analytics – Scanning", layout="wide")
st.title("📦 GreenPack Analytics – Strategic Scanning Monitor")
st.caption("Collect signals from media, institutions, and online discourse; detect themes, weak signals, and stakeholder sentiment.")

with st.sidebar:
    db_path = DEFAULT_DB
    con = db_connect(db_path)
    init_db(con)

    st.subheader("Keywords")
    keywords_text = st.text_area("One per line", "\n".join(DEFAULT_KEYWORDS), height=160)
    keywords = [k.strip() for k in keywords_text.splitlines() if k.strip()]

    st.subheader("GDELT settings")
    gdelt_days = st.slider("GDELT date window (auto-retries anyway)", 7, 90, 30)
    gdelt_max = st.slider("GDELT max records", 10, 300, 120, step=10)
    gdelt_full_text = st.checkbox("GDELT: fetch full article text (slower)", value=False)
    gdelt_full_text_n = st.slider("GDELT full-text limit", 0, 60, 15)

    st.subheader("RSS feeds (one per line)")
    rss_text = st.text_area("RSS URLs", "\n".join(DEFAULT_RSS_FEEDS), height=180)
    rss_feeds = [u.strip() for u in rss_text.splitlines() if u.strip()]
    rss_full_text = st.checkbox("RSS: fetch full article text (slower)", value=False)
    rss_max_per_feed = st.slider("RSS max items per feed", 5, 60, 25)

    st.subheader("Reddit (RSS search, no API keys)")
    use_reddit = st.checkbox("Enable Reddit source", value=True)
    reddit_query = st.text_input("Reddit query", DEFAULT_REDDIT_QUERY)
    reddit_subs_text = st.text_area("Subreddits (one per line)", "\n".join(DEFAULT_REDDIT_SUBREDDITS), height=120)
    reddit_subs = [s.strip() for s in reddit_subs_text.splitlines() if s.strip()]
    reddit_max_per_sub = st.slider("Max posts per subreddit", 5, 50, 20)

    st.subheader("GNews (optional extra)")
    use_gnews = st.checkbox("Enable GNews source", value=True)
    lookback_days = st.slider("GNews lookback days", 7, 365, 180)
    st.caption("Tip: 'EU' may not work everywhere; FR/DE/GB are safer.")
    gnews_country = st.text_input("GNews country (2 letters)", "FR")
    gnews_language = st.text_input("GNews language (2 letters)", "en")
    gnews_max = st.slider("GNews max results per topic", 10, 100, 50, step=10)

    st.subheader("Official Watchlist (one per line)")
    official_text = st.text_area("Official URLs", "\n".join(DEFAULT_OFFICIAL_URLS), height=160)
    official_urls = [u.strip() for u in official_text.splitlines() if u.strip()]

tab_collect, tab_explore, tab_analyze = st.tabs(["Collect data", "Explore data", "Analyze (Insights)"])

with tab_collect:
    st.subheader("Collect data")

    c1, c2, c3, c4, c5 = st.columns(5)
    do_gdelt = c1.checkbox("GDELT", value=True)
    do_rss = c2.checkbox("RSS", value=True)
    do_reddit = c3.checkbox("Reddit", value=True)
    do_gnews = c4.checkbox("GNews", value=True)
    do_official = c5.checkbox("Official", value=True)

    if GNews is None and do_gnews:
        st.warning("GNews library not available. Add `gnews` to requirements.txt.")

    if st.button("🚀 Run collection", type="primary"):
        total_inserted = 0

        if do_gdelt:
            with st.spinner("Collecting from GDELT…"):
                ins, seen, extracted_ok, _debug = collect_gdelt(
                    con,
                    keywords=keywords,
                    days=gdelt_days,
                    max_records=gdelt_max,
                    fetch_full_text=gdelt_full_text,
                    max_full_text=gdelt_full_text_n,
                )
                total_inserted += ins
                st.success(f"✅ GDELT seen {seen} | extracted {extracted_ok} | inserted {ins}")

        if do_rss:
            with st.spinner("Collecting from RSS…"):
                ins, feeds_ok, entries_seen, extracted_ok = collect_rss(
                    con,
                    feed_urls=rss_feeds,
                    fetch_full_text=rss_full_text,
                    max_per_feed=rss_max_per_feed,
                )
                total_inserted += ins
                st.success(f"✅ RSS feeds OK {feeds_ok} | entries seen {entries_seen} | extracted {extracted_ok} | inserted {ins}")

        if do_reddit and use_reddit:
            with st.spinner("Collecting from Reddit RSS…"):
                ins, seen, extracted_ok = collect_reddit_rss(
                    con,
                    subreddits=reddit_subs,
                    query=reddit_query,
                    max_per_sub=reddit_max_per_sub,
                )
                total_inserted += ins
                st.success(f"✅ Reddit entries seen {seen} | extracted {extracted_ok} | inserted {ins}")

        if do_gnews and use_gnews and GNews is not None:
            with st.spinner("Collecting from GNews…"):
                ins, seen = collect_gnews(
                    con,
                    keywords=keywords,
                    lookback_days=lookback_days,
                    country=(gnews_country.strip() or "FR"),
                    language=(gnews_language.strip() or "en"),
                    max_results=gnews_max,
                )
                total_inserted += ins
                st.success(f"✅ GNews seen {seen} | inserted {ins}")

        if do_official:
            with st.spinner("Collecting from official pages…"):
                ins, extracted_ok = collect_official(con, official_urls)
                total_inserted += ins
                st.success(f"✅ Official extracted {extracted_ok} | inserted {ins}")

        if total_inserted == 0:
            st.warning(
                "No new documents inserted. Try:\n"
                "- Broader keywords (e.g., 'packaging waste')\n"
                "- Increase RSS items per feed / Reddit max posts\n"
                "- Replace dead RSS feeds\n"
                "- Keep full-text OFF for speed"
            )
        else:
            st.success(f"Done — inserted {total_inserted} documents.")

    st.info("Note: sentiment is computed on (title + text snippet). It indicates tone/concerns, not objective policy outcomes.")

with tab_explore:
    st.subheader("Explore data")
    days_view = st.slider("Show data from last N days", 7, 365, 180)

    df = load_docs(con, days=days_view)
    if df.empty:
        st.info("No data yet. Go to **Collect data** first.")
    else:
        sources = ["All"] + sorted(df["source"].dropna().unique().tolist())
        types = ["All"] + sorted(df["source_type"].dropna().unique().tolist())
        sentiments = ["All", "Positive", "Neutral", "Negative"]

        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        pick_source = c1.selectbox("Source", sources, index=0)
        pick_type = c2.selectbox("Source type", types, index=0)
        pick_sent = c3.selectbox("Sentiment", sentiments, index=0)
        q = c4.text_input("Search in title/text", value="")

        view = df.copy()
        if pick_source != "All":
            view = view[view["source"] == pick_source]
        if pick_type != "All":
            view = view[view["source_type"] == pick_type]
        if pick_sent != "All":
            view = view[view["sentiment_label"] == pick_sent]
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
            show[["source_type", "source", "published_at", "domain", "sentiment_label", "sentiment_compound", "title", "url"]],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download CSV",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="greenpack_scanning_export.csv",
            mime="text/csv",
        )

with tab_analyze:
    st.subheader("Analyze (Insights)")
    days_dash = st.slider("Analyze last N days", 7, 365, 180)

    df = load_docs(con, days=days_dash)
    if df.empty:
        st.info("No data yet. Collect first.")
    else:
        ts = df["published_at"].fillna(df["collected_at"])

        # --- Top summary metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Documents", f"{len(df):,}")
        m2.metric("Sources", f"{df['source'].nunique()}")
        m3.metric("Source types", f"{df['source_type'].nunique()}")
        m4.metric("Domains", f"{df['domain'].nunique()}")
        m5.metric("Avg sentiment", f"{df['sentiment_compound'].mean():.2f}")

        # --- Key insights (the "reader-friendly" part)
        st.write("## Key insights (auto-generated)")
        insights = build_key_insights(df)
        for b in insights:
            st.markdown(f"- {b}")

        st.write("---")

        # --- Volume over time
        st.write("### Attention over time (documents/day)")
        tmp = df.copy()
        tmp["date"] = ts.dt.date
        daily = tmp.groupby("date").size().rename("count").to_frame()
        st.line_chart(daily)

        # --- Sentiment over time
        st.write("### Sentiment over time (average/day)")
        s_daily = tmp.groupby("date")["sentiment_compound"].mean().rename("avg_sentiment").to_frame()
        st.line_chart(s_daily)

        # --- Source mix
        st.write("### Source mix")
        by_source = df["source"].value_counts().rename_axis("source").reset_index(name="count")
        st.bar_chart(by_source.set_index("source")["count"])

        st.write("### Credibility / stakeholder sphere (source type)")
        by_type = df["source_type"].fillna("Other").value_counts().rename_axis("source_type").reset_index(name="count")
        st.bar_chart(by_type.set_index("source_type")["count"])

        # --- Sentiment by stakeholder sphere
        st.write("### Sentiment by source type")
        by_type_sent = df.groupby("source_type")["sentiment_compound"].mean().sort_values()
        st.bar_chart(by_type_sent)

        # --- Themes
        st.write("### Top terms (dominant themes)")
        terms = top_terms(df, n=25)
        st.dataframe(terms, use_container_width=True)

        st.write("### Emerging terms (weak signals) — last 30 days vs previous 30 days")
        emerg = emerging_terms(df, window_days=30, top_n=15, min_recent=3)
        if emerg.empty:
            st.info("Not enough data yet for emerging terms. Collect more items or increase the analysis window.")
        else:
            st.dataframe(emerg, use_container_width=True)

        # --- Signal buckets
        st.write("### Signals (concerns / expectations buckets)")
        sig = simple_signals(df)
        if sig.empty:
            st.info("No signals detected yet (try collecting more items).")
        else:
            st.dataframe(sig, use_container_width=True)
            st.bar_chart(sig.set_index("signal")["count"])

        # --- Most negative / positive items (useful for interpretation)
        st.write("### Most negative & most positive items (for qualitative interpretation)")
        cols = ["source_type", "source", "published_at", "sentiment_compound", "title", "url"]

        most_neg = df.sort_values("sentiment_compound").head(8).copy()
        most_neg["published_at"] = most_neg["published_at"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
        st.write("**Most negative**")
        st.dataframe(most_neg[cols], use_container_width=True, hide_index=True)

        most_pos = df.sort_values("sentiment_compound", ascending=False).head(8).copy()
        most_pos["published_at"] = most_pos["published_at"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
        st.write("**Most positive**")
        st.dataframe(most_pos[cols], use_container_width=True, hide_index=True)

        st.caption(
            "Method note: sentiment is computed with VADER on (title + snippet/body). It is used to detect stakeholder "
            "concerns/expectations and shifts in discourse tone, not to predict legislative outcomes."
        )
