"""Pipeline orchestration shared by the CLI and the Streamlit dashboard.

Phase 1 (``scraper.run_phase1``) talks to the website and writes ``raw_ads.jsonl`` + images.
Phase 2 (``run_phase2``) never touches the website: categorize -> IQA + view type -> duplicates -> cover ->
validated ``taladrod_data.json``. So the AI part can be re-tuned and re-run for free.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .categorizer import categorize
from .config import MAX_ADS_CAP, Settings
from .enrich import clip_category, mark_duplicates, pick_cover
from .iqa import BODY_PROMPTS, MUSIQ_METRIC, VIEW_PROMPTS, ImageScorer
from .models import Dataset
from .scraper import load_checkpoint, run_phase1

log = logging.getLogger(__name__)

Progress = Callable[[int, int, str], None]


class NoCheckpointError(RuntimeError):
    """Phase 2 was asked to rebuild existing results but there is no checkpoint to rebuild from."""


def run_phase2(settings: Settings, iqa=None, on_progress: Optional[Progress] = None) -> Dataset:
    notify = on_progress or (lambda d, t, m: None)
    if not settings.raw_path.exists() and settings.final_path.exists():
        # e.g. a results folder copied without its checkpoint: scoring "nothing" would replace
        # the finished JSON with an empty one
        raise NoCheckpointError(
            f"{settings.raw_path} not found, so {settings.final_path} was left untouched. "
            "Run the scrape step first, or put raw_ads.jsonl back next to the images.")
    ads = list(load_checkpoint(settings.raw_path).values())
    # The checkpoint can hold more than MAX_ADS (two scrapers at once, repeated resumes); the
    # deliverable never does: keep the first ads collected. max_ads=0 means "scrape nothing",
    # not "discard what is already collected".
    cap = settings.max_ads or MAX_ADS_CAP
    if len(ads) > cap:
        log.warning("checkpoint has %d ads, using the first %d", len(ads), cap)
        notify(cap, cap, f"checkpoint has {len(ads)} ads, using the first {cap}")
        ads = ads[:cap]

    for ad in ads:
        ad.category, ad.category_source = categorize(ad)

    jobs = [(img, Path(img.local_path)) for ad in ads for img in ad.images if img.local_path and not img.error]
    body_probs: dict[int, dict[str, float]] = {}  # id(image record) -> P(body type)
    for ad in ads:  # images that never downloaded still get a score (0.0) so every image is scored
        for img in ad.images:
            if not img.local_path or img.error:
                img.quality_score, img.quality_breakdown = 0.0, {"musiq": 0.0}
                img.error = img.error or "no local file"

    if jobs:
        iqa = iqa or ImageScorer(settings)
        results = iqa.score_paths([p for _, p in jobs],
                                  on_progress=lambda d, t: notify(d, t, f"scored {d}/{t} images"))
        for (img, _), r in zip(jobs, results):
            img.quality_score, img.quality_breakdown, img.raw_metrics = r.score, r.breakdown, r.raw_metrics or None
            img.view_type, img.view_confidence = r.view_type, r.view_confidence
            body_probs[id(img)] = r.body_probs
            if r.error:
                img.error = r.error

    mark_duplicates(ads)
    for ad in ads:
        ad.cover_image = pick_cover(ad)
        ad.category_clip, ad.category_clip_confidence, ad.category_mismatch = clip_category(
            ad.category, [body_probs.get(id(img), {}) for img in ad.images])

    ds = Dataset(meta={
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": settings.start_url,
        "ad_count": len(ads),
        "image_count": sum(len(a.images) for a in ads),
        "selection_strategy": settings.selection_strategy,
        "iqa_method": f"MUSIQ (pyiqa '{MUSIQ_METRIC}', KonIQ-10k weights, runs locally); "
                      "quality_score = MUSIQ (0-100) / 10",
        "view_method": "CLIP zero-shot (local open_clip), mean embedding of the prompts per class",
        "view_prompts": VIEW_PROMPTS,
        "clip_model": f"{settings.clip_model}/{settings.clip_pretrained}",
        "category_clip_prompts": BODY_PROMPTS,
        "categories": ["เก๋ง", "กระบะ", "SUV/PPV", "รถตู้/MPV", "อื่นๆ"],
    }, ads=ads)
    settings.final_path.parent.mkdir(parents=True, exist_ok=True)
    settings.final_path.write_text(
        json.dumps(ds.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    notify(len(ads), len(ads), f"wrote {settings.final_path}")
    return ds


def run_all(settings: Settings, on_progress: Optional[Progress] = None, iqa=None) -> Dataset:
    run_phase1(settings, on_progress=on_progress)
    return run_phase2(settings, iqa=iqa, on_progress=on_progress)


def load_dataset(path: Path) -> Optional[Dataset]:
    return Dataset.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None
