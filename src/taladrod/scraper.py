"""Phase 1: collect ads + images. Output: ``raw_ads.jsonl`` (one ad per line, appended as we go)
and image files in one flat folder named ``{ad_id}_{index:02d}{ext}``.

Selection strategy: see ``config.SELECTION_STRATEGY``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional, Protocol

from .config import Settings
from .http import BlockedError, PoliteClient
from .models import AdRecord
from .parse import ParseError, iter_pages, parse_detail_page, parse_list_page

log = logging.getLogger(__name__)

Progress = Callable[[int, int, str], None]

EXT_BY_TYPE = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


class Client(Protocol):
    def get_html(self, url: str) -> str: ...
    def get_bytes(self, url: str) -> tuple[bytes, str]: ...


def load_checkpoint(path: Path) -> dict[str, AdRecord]:
    ads: dict[str, AdRecord] = {}
    if not path.exists():
        return ads
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            ad = AdRecord.model_validate_json(line)
        except ValueError as e:  # e.g. a line cut short by a kill / power loss mid-write
            log.warning("checkpoint line %d unreadable, skipped (ad will be fetched again): %s", n, e)
            continue
        ads[ad.ad_id] = ad
    return ads


def _append(path: Path, ad: AdRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(ad.model_dump_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


def download_images(ad: AdRecord, client: Client, images_dir: Path) -> AdRecord:
    """Download every image of ``ad``. A failed image keeps ``local_path=None`` and an ``error``."""
    images_dir.mkdir(parents=True, exist_ok=True)
    for img in ad.images:
        try:
            content, ctype = client.get_bytes(img.source_url)
            ext = EXT_BY_TYPE.get(ctype.split(";")[0].strip().lower(), ".jpg")
            path = images_dir / f"{ad.ad_id}_{img.index:02d}{ext}"
            path.write_bytes(content)
            img.local_path = path.as_posix()
        except BlockedError:
            raise  # site is challenging us: stop the run, do not save a half-downloaded ad
        except Exception as e:  # one bad image must not kill the ad
            log.warning("image %s failed: %s", img.source_url, e)
            img.local_path, img.error = None, f"download failed: {e}"
    return ad


def run_phase1(settings: Settings, client: Optional[Client] = None,
               on_progress: Optional[Progress] = None) -> list[AdRecord]:
    client = client or PoliteClient(settings)
    done = load_checkpoint(settings.raw_path)
    target = settings.max_ads
    notify = on_progress or (lambda d, t, m: None)

    if len(done) >= target:
        notify(len(done), target, "checkpoint already complete")
        return list(done.values())

    try:
        stubs = parse_list_page(client.get_html(settings.start_url), settings.start_url)
    except Exception as e:  # blocked, layout changed, 403, persistent 5xx...
        log.error("cannot read listing page: %s", e)
        notify(len(done), target, f"stopped: {e}")
        return list(done.values())
    notify(len(done), target, f"listing has {len(stubs)} ads")

    failures = 0
    for page_no, page in enumerate(iter_pages(stubs, settings.page_size), start=1):
        for stub in page:
            if len(done) >= target:
                break
            if stub.ad_id in done:
                continue
            try:
                ad = parse_detail_page(client.get_html(stub.url), stub.url, stub=stub)
                ad.ad_id = stub.ad_id  # listing id is the canonical one
                download_images(ad, client, settings.images_dir)
            except BlockedError as e:
                log.error("blocked by site, stopping (resume later): %s", e)
                notify(len(done), target, "stopped: site is challenging requests, resume later")
                return list(done.values())
            except Exception as e:
                failures += 1
                log.warning("skip ad %s (%d in a row): %s", stub.ad_id, failures, e)
                if failures >= settings.max_consecutive_failures:
                    log.error("%d ads failed in a row, stopping (resume later)", failures)
                    notify(len(done), target, f"stopped: {failures} ads failed in a row, resume later")
                    return list(done.values())
                continue
            failures = 0
            _append(settings.raw_path, ad)
            done[ad.ad_id] = ad
            notify(len(done), target, f"page {page_no}: {ad.full_title} ({len(ad.images)} images)")
        if len(done) >= target:
            break

    return list(done.values())
