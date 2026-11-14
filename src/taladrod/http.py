"""Polite HTTP client: random delay before every request, exponential-backoff retries, and
clean detection of the site's AWS WAF challenge (HTTP 202 + ``x-amzn-waf-action: challenge``).

We never try to solve or bypass the challenge. If it persists after all retries we raise
``BlockedError`` so the scraper can stop, keep its checkpoint, and be resumed later.
"""
from __future__ import annotations

import logging
import random
import time

import requests
from tenacity import RetryError, Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Settings

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class RetryableError(Exception):
    pass


class ChallengeError(RetryableError):
    pass


class BlockedError(RuntimeError):
    """The site keeps answering with a bot challenge; stop and resume later."""


class PoliteClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.s = settings
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": settings.user_agent,
            "Accept-Language": "th-TH,th;q=0.9,en;q=0.8",
        })

    # -- internals ---------------------------------------------------------
    def _pause(self) -> None:
        lo, hi = self.s.delay_range
        if hi > 0:
            time.sleep(random.uniform(lo, hi))

    def _once(self, url: str, **kw) -> requests.Response:
        self._pause()
        try:
            r = self.session.get(url, timeout=self.s.timeout, **kw)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise RetryableError(f"{type(e).__name__}: {e}") from e
        if r.headers.get("x-amzn-waf-action") == "challenge" or (r.status_code == 202 and not r.content):
            log.warning("WAF challenge on %s, backing off %.0fs", url, self.s.challenge_wait)
            time.sleep(self.s.challenge_wait)
            raise ChallengeError(url)
        if r.status_code in RETRY_STATUS:
            raise RetryableError(f"HTTP {r.status_code} for {url}")
        r.raise_for_status()  # 4xx (e.g. 404) is final: no retry
        return r

    def _get(self, url: str, **kw) -> requests.Response:
        retrying = Retrying(
            stop=stop_after_attempt(self.s.retries),
            wait=wait_exponential(multiplier=self.s.backoff_base, min=self.s.backoff_base, max=60),
            retry=retry_if_exception_type(RetryableError),
            sleep=lambda secs: time.sleep(secs),
            reraise=False,
        )
        try:
            for attempt in retrying:
                with attempt:
                    return self._once(url, **kw)
        except RetryError as e:
            last = e.last_attempt.exception()
            if isinstance(last, ChallengeError):
                raise BlockedError(f"still challenged after {self.s.retries} tries: {url}") from last
            raise last  # type: ignore[misc]
        raise AssertionError("unreachable")

    # -- public ------------------------------------------------------------
    def get_html(self, url: str) -> str:
        r = self._get(url)
        if not r.encoding or r.encoding.lower() == "iso-8859-1":
            # server omitted charset: let chardet decide (legacy Thai pages may be windows-874)
            r.encoding = r.apparent_encoding
        return r.text

    def get_bytes(self, url: str) -> tuple[bytes, str]:
        r = self._get(url, headers={"Referer": "https://www.taladrod.com/"})
        return r.content, r.headers.get("content-type", "")
