"""Web crawler and translator for extracting Amharic text."""

import re
import time
import urllib.parse
from typing import List, Set, Optional
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode
from collections import deque
import requests
from requests.exceptions import RequestException
from deep_translator.exceptions import NotValidPayload, LanguageNotSupportedException
from bs4 import BeautifulSoup


class AmharicCrawler:
    """
    A web crawler for extracting Amharic text from websites.
    Supports normalization, filtering, and saving extracted sentences.
    """

    AMHARIC_REGEX = re.compile(r"[\u1200-\u137f]")  # Ethiopic block
    SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?።፧፨])\s+")

    UNWANTED_EXTENSIONS = (
        ".jpg", ".jpeg", ".png", ".gif",
        ".svg", ".pdf", ".mp4", ".zip", ".exe", ".webp", ".ico"
    )
    SKIP_KEYWORDS = (
        "login", "signup", "register", "privacy",
        "contact", "terms", "policy", "account", "cookie"
    )

    def __init__(
        self,
        start_urls: List[str],
        max_pages: int = 500,
        delay: float = 1.0,
        output_txt: str = "raw_amharic.txt",
    ) -> None:
        self.start_urls: List[str] = start_urls
        self.max_pages: int = max_pages
        self.delay: float = delay
        self.output_txt: str = output_txt
        self.visited: Set[str] = set()
        self.queue: deque = deque(start_urls)

        # Reset output file
        with open(self.output_txt, "w", encoding="utf-8").close():
            pass

    @staticmethod
    def is_amharic_text(text: str) -> bool:
        """Return True if the text contains any Amharic characters."""
        return bool(AmharicCrawler.AMHARIC_REGEX.search(text))

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize whitespace and strip text."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def get_page(url: str) -> Optional[str]:
        """Normalize whitespace and strip text."""
        try:
            url_encoded = urllib.parse.quote(url, safe=":/?=&")
            resp = requests.get(
                url_encoded, timeout=10,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp.text
        except RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            return None

    @staticmethod
    def normalize_url(base_url: str, href: str) -> Optional[str]:
        """Join and clean URLs, removing fragments and tracking params."""
        parsed = urljoin(base_url, href)
        parts = urlparse(parsed)
        clean_query = {
            k: v
            for k, v in parse_qs(parts.query).items()
            if not (k.lower().startswith("utm") or k.lower() in ["fbclid", "ref"])
        }

        clean_parts = parts._replace(
            fragment="", query=urlencode(clean_query, doseq=True))
        normalized = urlunparse(clean_parts)
        return normalized

    @classmethod
    def extract_links(cls, html: str, base_url: str) -> Set[str]:
        """Extract only relevant internal links for crawling."""
        soup = BeautifulSoup(html, "html.parser")
        base_domain = urlparse(base_url).netloc

        links: Set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("#"):
                continue

            # Normalize
            normalized = cls.normalize_url(base_url, href)
            if not normalized:
                continue

            parsed = urlparse(normalized)

            # Filter by domain
            if parsed.netloc != base_domain:
                continue

            # Skip file extensions & unwanted paths
            if normalized.lower().endswith(cls.UNWANTED_EXTENSIONS):
                continue
            if any(word in normalized.lower() for word in cls.SKIP_KEYWORDS):
                continue

            links.add(normalized)

        return links

    # -------------------- Content Extraction --------------------
    @classmethod
    def extract_and_translate_sentences(cls, html: str) -> List[str]:
        """
        Extract visible text, split into sentences, translate to Amharic, 
        and return list of lines.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove non-textual or redundant sections
        for tag in soup([
            "script", "style", "noscript", "iframe", "header", "footer",
            "svg", "img", "nav", "form"
        ]):
            tag.decompose()

        sentences_am: List[str] = []
        # translator = GoogleTranslator(source="auto", target="am")

        for t in soup.stripped_strings:
            if len(t) < 3:
                continue

            t_clean = cls.clean_text(t)
            if not t_clean:
                continue

            # Split into sentences
            split_sentences = re.split(cls.SENTENCE_SPLIT_REGEX, t_clean)
            for sent in split_sentences:
                sent = sent.strip()
                if len(sent) < 3:
                    continue

                try:
                    if cls.is_amharic_text(sent):
                        sentences_am.append(sent)
                    # else:
                        # translated = translator.translate(sent)
                        # sentences_am.append(translated)
                        # time.sleep(0.15)
                except (NotValidPayload, LanguageNotSupportedException) as e:
                    print(f"  - Translation failed for '{sent[:40]}...': {e}")

        return sentences_am

    # -------------------- File Output --------------------
    def append_sentences_to_file(self, sentences: List[str]) -> None:
        """Append translated sentences to output file (one per line)."""
        with open(self.output_txt, "a", encoding="utf-8") as f:
            for sent in sentences:
                f.write(sent.replace("\n", " ").strip() + "\n")

    # -------------------- Main Crawler --------------------
    def crawl(self) -> None:
        """Main crawl loop."""
        while self.queue and len(self.visited) < self.max_pages:
            url: str = self.queue.popleft()
            if url in self.visited:
                continue
            self.visited.add(url)

            print(f"[{len(self.visited)}] Crawling → {url}")
            html: Optional[str] = self.get_page(url)
            if html is None:
                continue

            sentences: List[str] = self.extract_and_translate_sentences(html)
            if sentences:
                self.append_sentences_to_file(sentences)
                print(f"  ✓ Saved {len(sentences)} sentences.")

            links: Set[str] = self.extract_links(html, url)
            for link in links:
                if link not in self.visited:
                    self.queue.append(link)

            time.sleep(self.delay)

        print(
            f"\n✅ Crawled {len(self.visited)} pages. Output: {self.output_txt}")


if __name__ == "__main__":
    START_URLS = ["<URL>"]
    crawler = AmharicCrawler(start_urls=START_URLS, max_pages=500, delay=1.0)
    crawler.crawl()
