"""Extra features beyond the brief: duplicate-photo detection, automatic cover-image pick and a
CLIP body-type cross-check of the dictionary category."""
from __future__ import annotations

import logging
from typing import Optional

import imagehash
from PIL import Image

from .models import AdRecord

log = logging.getLogger(__name__)

# Below this, CLIP's body-type guess is too unsure to raise a flag. In the trial run on 100 ads
# every wrong "pickup -> SUV" guess sat at 0.50-0.53.
MIN_MISMATCH_CONFIDENCE = 0.6


def mark_duplicates(ads: list[AdRecord], max_distance: int = 4) -> None:
    """Perceptual-hash every downloaded image. An image within ``max_distance`` bits of one seen
    earlier (same ad or another ad: dealers often reuse photos) gets ``is_duplicate_of`` set to
    the first image's ``local_path``."""
    seen: list[tuple[imagehash.ImageHash, str]] = []
    for ad in ads:
        for img in ad.images:
            if not img.local_path or img.error:
                continue
            try:
                with Image.open(img.local_path) as im:
                    h = imagehash.phash(im)
            except Exception as e:
                log.warning("phash failed for %s: %s", img.local_path, e)
                continue
            img.phash = str(h)
            img.is_duplicate_of = next((path for prev, path in seen if h - prev <= max_distance), None)
            if img.is_duplicate_of is None:
                seen.append((h, img.local_path))


def pick_cover(ad: AdRecord) -> Optional[str]:
    """Best-scoring image that downloaded fine and is not a duplicate. Exterior shots win over
    everything else: an ad's cover should show the car, not its dashboard."""
    usable = [i for i in ad.images
              if i.local_path and not i.error and i.is_duplicate_of is None and i.quality_score is not None]
    usable = [i for i in usable if i.view_type == "exterior"] or usable
    return max(usable, key=lambda i: (i.quality_score, -i.index)).local_path if usable else None


def clip_category(category: Optional[str], body_probs: list[dict[str, float]],
                  ) -> tuple[Optional[str], Optional[float], Optional[bool]]:
    """Body type CLIP sees in an ad's photos: ``(guess, confidence, mismatch)``.

    ``body_probs`` holds one dict per image; only clear exterior shots carry one (``iqa`` leaves
    the rest empty), and they are averaged. ``mismatch`` is True when a confident guess
    (>= ``MIN_MISMATCH_CONFIDENCE``) disagrees with the dictionary ``category``; "อื่นๆ"/None is
    "unknown", not a disagreement. No exterior shot -> ``(None, None, None)``."""
    rows = [p for p in body_probs if p]
    if not rows:
        return None, None, None
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    guess = max(mean, key=mean.get)
    conf = round(mean[guess], 3)
    return guess, conf, category not in (None, "อื่นๆ") and category != guess and conf >= MIN_MISMATCH_CONFIDENCE


def avg_quality(ad: AdRecord) -> Optional[float]:
    """Mean score of images that were actually scored (failed downloads are not 'bad photos')."""
    scores = [i.quality_score for i in ad.images if not i.error and i.quality_score is not None]
    return round(sum(scores) / len(scores), 1) if scores else None
