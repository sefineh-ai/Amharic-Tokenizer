#!/usr/bin/env python3
"""Breadth-first crawler that collects Amharic sentences from websites.

Not part of the installed package. Requires ``requests`` and ``beautifulsoup4``::

    pip install requests beautifulsoup4
    python tools/crawl_corpus.py https://example.com --max-pages 500 --output raw_amharic.txt
    amh-tokenizer clean raw_amharic.txt cleaned_amharic.txt

Stays on the start URLs' domains and waits ``--delay`` seconds between requests.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Set
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("crawl_corpus")

ETHIOPIC = re.compile(r"[ሀ-፿]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?።፧፨])\s+")
WHITESPACE = re.compile(r"\s+")
SKIPPED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".pdf", ".mp4", ".zip", ".exe",
)  # fmt: skip
SKIPPED_PATH_KEYWORDS = (
    "login", "signup", "register", "privacy", "contact", "terms", "policy", "account", "cookie",
)  # fmt: skip
NON_CONTENT_TAGS = (
    "script", "style", "noscript", "iframe", "header", "footer", "svg", "img", "nav", "form",
)  # fmt: skip
TRACKING_PARAMS = ("fbclid", "ref")
USER_AGENT = "Mozilla/5.0 (compatible; amharic-tokenizer-corpus-crawler)"
MIN_SENTENCE_LENGTH = 3


def normalize_url(base_url: str, href: str) -> str:
    """Resolve ``href`` against ``base_url`` and drop fragments and tracking parameters."""
    parts = urlparse(urljoin(base_url, href))
    query = {
        k: v
        for k, v in parse_qs(parts.query).items()
        if not (k.lower().startswith("utm") or k.lower() in TRACKING_PARAMS)
    }
    return urlunparse(parts._replace(fragment="", query=urlencode(query, doseq=True)))


def extract_links(html: str, base_url: str) -> Set[str]:
    """Same-domain links worth crawling."""
    domain = urlparse(base_url).netloc
    links: Set[str] = set()
    for anchor in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not href or href.startswith("#"):
            continue
        url = normalize_url(base_url, href)
        lowered = url.lower()
        if urlparse(url).netloc != domain or lowered.endswith(SKIPPED_EXTENSIONS):
            continue
        if any(word in lowered for word in SKIPPED_PATH_KEYWORDS):
            continue
        links.add(url)
    return links


def extract_sentences(html: str) -> List[str]:
    """Visible sentences of the page that contain Ethiopic characters."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(NON_CONTENT_TAGS):
        tag.decompose()
    sentences = []
    for fragment in soup.stripped_strings:
        text = WHITESPACE.sub(" ", fragment).strip()
        for sentence in SENTENCE_SPLIT.split(text):
            sentence = sentence.strip()
            if len(sentence) >= MIN_SENTENCE_LENGTH and ETHIOPIC.search(sentence):
                sentences.append(sentence)
    return sentences


class Crawler:
    def __init__(self, start_urls: Iterable[str], output: Path, max_pages: int, delay: float):
        self.queue: Deque[str] = deque(start_urls)
        self.output = output
        self.max_pages = max_pages
        self.delay = delay
        self.visited: Set[str] = set()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def fetch(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(quote(url, safe=":/?=&%"), timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("failed to fetch %s: %s", url, exc)
            return None
        resp.encoding = "utf-8"
        return resp.text

    def run(self) -> int:
        """Crawl until the queue is empty or ``max_pages`` is reached; return sentences written."""
        written = 0
        with self.output.open("w", encoding="utf-8") as out:
            while self.queue and len(self.visited) < self.max_pages:
                url = self.queue.popleft()
                if url in self.visited:
                    continue
                self.visited.add(url)
                logger.info("[%d] %s", len(self.visited), url)
                html = self.fetch(url)
                if html is None:
                    continue
                for sentence in extract_sentences(html):
                    out.write(sentence.replace("\n", " ") + "\n")
                    written += 1
                self.queue.extend(
                    link for link in extract_links(html, url) if link not in self.visited
                )
                time.sleep(self.delay)
        logger.info(
            "crawled %d pages, wrote %d sentences to %s", len(self.visited), written, self.output
        )
        return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("start_urls", nargs="+")
    parser.add_argument("--output", type=Path, default=Path("raw_amharic.txt"))
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    Crawler(args.start_urls, args.output, args.max_pages, args.delay).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
