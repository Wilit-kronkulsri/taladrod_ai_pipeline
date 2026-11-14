"""HTML parsers for taladrod.com. Selectors and page structure are documented in docs/recon.md.

The listing page embeds every result as ``var SchDataJSON={...}`` and the site paginates on the
client side (10 rows per scroll), so we parse that JSON and paginate it ourselves in display order.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .models import AdRecord, ImageRecord

log = logging.getLogger(__name__)

DETAIL_PATH = "../icar/cardet.aspx?cid={cid}"
_NUM = re.compile(r"\d[\d,]*")


class ParseError(ValueError):
    """The page does not have the structure we expect (e.g. a WAF challenge or layout change)."""


@dataclass(frozen=True)
class ListingStub:
    """One row of the search results, enough to fetch and fall back on."""
    ad_id: str
    url: str
    year: Optional[int]
    price_thb: Optional[int]
    model: Optional[str]
    name_mmt: str
    headline: str


# ---------------------------------------------------------------- field normalizers
def _first_number(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = _NUM.search(text)
    if not m:
        return None
    digits = m.group(0).replace(",", "")
    return int(digits) if digits else None


def parse_price(text: Optional[str]) -> Optional[int]:
    return _first_number(text)


def parse_mileage(text: Optional[str]) -> Optional[int]:
    return _first_number(text)


def parse_year(text: Optional[str]) -> Optional[int]:
    """Return a Gregorian year. Buddhist-era years (> 2400) are converted by subtracting 543."""
    if not text:
        return None
    m = re.search(r"\b(\d{4})\b", text)
    if not m:
        return None
    year = int(m.group(1))
    return year - 543 if year > 2400 else year


# ---------------------------------------------------------------- listing page
def _extract_sch_json(html: str) -> dict:
    marker = "var SchDataJSON="
    start = html.find(marker)
    if start < 0:
        raise ParseError("SchDataJSON not found on listing page")
    try:
        data, _ = json.JSONDecoder().raw_decode(html, start + len(marker))
    except json.JSONDecodeError as e:
        raise ParseError(f"SchDataJSON is not valid JSON: {e}") from e
    return data


def parse_list_page(html: str, base_url: str) -> list[ListingStub]:
    """All ads on the result page, in the order the site displays them."""
    data = _extract_sch_json(html)
    stubs: list[ListingStub] = []
    seen: set[str] = set()
    for car in data.get("cars", []):
        cid = str(car.get("cid", "")).strip()
        if car.get("type") != "car" or not cid or cid in seen:
            continue  # "upg" rows are promo blocks, not ads
        seen.add(cid)
        stubs.append(ListingStub(
            ad_id=cid,
            url=urljoin(base_url, DETAIL_PATH.format(cid=cid)),
            year=parse_year(car.get("yr4")),
            price_thb=parse_price(car.get("prc")),
            model=(car.get("model") or None),
            name_mmt=car.get("namemmt", ""),
            headline=car.get("title", ""),
        ))
    return stubs


def iter_pages(stubs: list[ListingStub], page_size: int = 10) -> Iterator[list[ListingStub]]:
    """Yield the result list in pages, mirroring the site's 10-per-scroll pagination."""
    for i in range(0, len(stubs), page_size):
        yield stubs[i:i + page_size]


# ---------------------------------------------------------------- detail page
_IMG_PREFIX = re.compile(r"(https?://[^\"'\s]+/c/cidx/[\d/]+?/\d+)_1\.jpg")


def _split_title(text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """'BMW X1, 2.0 sDrive20d M SPORT' -> ('BMW', 'X1', '2.0 sDrive20d M SPORT')."""
    head, _, trim = text.partition(",")
    parts = head.split(maxsplit=1)
    brand = parts[0] if parts else None
    model = parts[1].strip() if len(parts) > 1 else None
    return brand, model or None, trim.strip() or None


def parse_detail_page(html: str, url: str, stub: Optional[ListingStub] = None) -> AdRecord:
    """Parse one ad. Missing fields become ``None`` (logged); ``stub`` fills gaps from the listing."""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    ad_id = None
    m = re.search(r"รหัสอ้างอิง\s*(\d+)", text)
    if m:
        ad_id = m.group(1)
    if not ad_id:
        m = re.search(r"cid=(\d+)", url)
        ad_id = m.group(1) if m else (stub.ad_id if stub else None)
    if not ad_id:
        raise ParseError(f"cannot find ad id for {url}")

    brand = model = trim = None
    year = None
    title_div = soup.select_one("div.CD_MMT_Title")
    if title_div:
        spans = title_div.find_all("span")
        if spans:
            year = parse_year(spans[0].get_text())
        if len(spans) > 1:
            brand, model, trim = _split_title(spans[1].get_text(" ", strip=True))
    else:
        log.warning("ad %s: title block missing, using listing data", ad_id)

    price = None
    price_el = soup.select_one("#xPrice")
    if price_el:
        price = parse_price(price_el.get_text())

    mileage = None
    m = re.search(r"เลขไมล์\s*([\d,]+)\s*กม", text)
    if m:
        mileage = parse_mileage(m.group(1))
    else:
        log.info("ad %s: no mileage on page", ad_id)

    if stub:
        year = year or stub.year
        price = price if price is not None else stub.price_thb
        model = model or stub.model
        brand = brand or (stub.name_mmt.split()[0] if stub.name_mmt else None)

    images: list[ImageRecord] = []
    og = soup.select_one('meta[property="og:image"]')
    prefix_m = _IMG_PREFIX.search(og["content"]) if og and og.get("content") else None
    if prefix_m is None:
        first = soup.find("img", src=re.compile(r"/c/cidx/.+_1\.jpg"))
        prefix_m = _IMG_PREFIX.search(first["src"]) if first else None
    if prefix_m:
        own = re.compile(re.escape(prefix_m.group(1)) + r"_(\d+)\.jpg")
        found: dict[int, str] = {}
        for img in soup.find_all("img", src=True):
            mm = own.match(img["src"])
            if mm:
                found.setdefault(int(mm.group(1)), img["src"])
        for n in sorted(found):
            images.append(ImageRecord(index=len(images) + 1, local_path=None, source_url=found[n]))
    else:
        log.warning("ad %s: no images found", ad_id)

    title_bits = [str(year) if year else None, brand, model, trim]
    full_title = " ".join(b for b in title_bits if b) or (stub.name_mmt if stub else ad_id)

    return AdRecord(
        ad_id=ad_id, brand=brand, model=model, year=year, full_title=full_title,
        price_thb=price, mileage_km=mileage, ad_source_url=url,
        seller_headline=(stub.headline or None) if stub else None,
        scraped_at=datetime.now(timezone.utc), images=images,
    )
