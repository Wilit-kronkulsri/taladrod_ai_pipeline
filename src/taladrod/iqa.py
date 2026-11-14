"""Phase 2b: local image analysis. Two pretrained models, both run on this machine (CPU is
enough) and fully offline after the first weight download. No external API is ever called.

1. Quality score: MUSIQ (Ke et al., ICCV 2021) through ``pyiqa``
--------------------------------------------------------------
MUSIQ is a no-reference IQA transformer trained on KonIQ-10k, i.e. on quality ratings given by
people, so one number already reflects sharpness, exposure and framing the way a viewer judges
them. ``quality_score = MUSIQ (0-100) / 10``, not rescaled.

Why MUSIQ: we scored the same 1,320 scraped photos with four approaches and compared the 16
lowest / 8 median / 16 highest images of each by eye (see README "How the IQA method was chosen"):

* CLIP-IQA with antonym prompt pairs (the first version of this module): rejected. Sharp full-car
  side shots scored 3.2, seat close-ups got "composition" 10.0, and its sharpness factor had a
  rank correlation of only 0.19 with Laplacian variance.
* Hybrid, OpenCV (re-blur sharpness + exposure) plus CLIP framing: the lowest images made sense,
  but engine bays and collages reached 10.0 and the median was 9.3: no separation at the top.
* NIMA: rejected. It is an aesthetics model; clean exterior side shots ranked last.
* MUSIQ: lowest = dark instrument screens, sunroof glass, tax stickers, tiny details;
  highest = sharp, well-lit shots of the whole car.

Trade-off: MUSIQ gives one overall score, not separate sharpness / lighting / composition
numbers. OpenCV Laplacian variance and mean brightness are still stored as ``raw_metrics`` so a
reviewer has something measurable to compare against.

2. What the photo shows: CLIP zero-shot (open_clip ``ViT-B-32`` / ``laion2b_s34b_b79k``)
-----------------------------------------------------------------------------------
CLIP turned out to be unreliable at judging *quality* but accurate at recognising *content*, so
that is what it does here:

* ``view_type``: exterior / interior / engine / detail / other (documents, stickers, keys).
  It explains a low score (a tax sticker is not a bad photo of a car, it is not a photo of the
  car) and lets the cover picker prefer exterior shots.
* body type (sedan / pickup / SUV / van) from **exterior shots only**, used by
  ``enrich.clip_category`` to flag ads whose photos disagree with the dictionary category.
  Using every image biased the guess to "SUV" (cabin shots look like nothing in particular);
  exterior-only agreed with the dictionary on 87 of 96 ads.

Each class is the mean embedding of a few paraphrased prompts (a prompt ensemble).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence

import cv2
import numpy as np
from PIL import Image

from .config import Settings

log = logging.getLogger(__name__)

VIEW_PROMPTS: dict[str, list[str]] = {
    "exterior": ["a photo of the whole exterior of a car",
                 "a photo of a car parked outside, seen from the side, front or rear"],
    "interior": ["a photo of a car interior with seats and dashboard",
                 "a photo of a car steering wheel, dashboard or seats from inside"],
    "engine": ["a photo of a car engine bay with the hood open"],
    "detail": ["a close-up photo of a car wheel, tire, headlight or badge",
               "a close-up photo of one small part of a car"],
    "other": ["a photo of a document, sticker, registration book or car keys",
              "a photo of a paper form or a tax sticker on a windshield"],
}
# keys are the Thai categories used in models.Category ("อื่นๆ" has no prompt on purpose)
BODY_PROMPTS: dict[str, list[str]] = {
    "เก๋ง": ["a photo of a sedan car", "a photo of a small hatchback car",
             "a photo of a coupe or convertible car"],
    "กระบะ": ["a photo of a pickup truck", "a photo of a pickup truck with an open cargo bed"],
    "SUV/PPV": ["a photo of an SUV", "a photo of a crossover SUV"],
    "รถตู้/MPV": ["a photo of a van", "a photo of a minivan"],
}
CLIP_LOGIT_SCALE = 100.0        # CLIP's trained temperature (exp(4.6052))
MIN_VIEW_CONFIDENCE = 0.6       # an image votes on body type only if it is clearly an exterior shot
MUSIQ_METRIC = "musiq"          # pyiqa name; KonIQ-10k weights, score range ~0-100


@dataclass
class ScoreResult:
    score: float                                   # 0-10
    breakdown: dict[str, float]                    # {"musiq": raw 0-100 score}
    raw_metrics: dict[str, float] = field(default_factory=dict)
    view_type: Optional[str] = None
    view_confidence: Optional[float] = None
    body_probs: dict[str, float] = field(default_factory=dict)  # only for clear exterior shots
    error: Optional[str] = None

    @classmethod
    def failed(cls, error: str) -> "ScoreResult":
        return cls(score=0.0, breakdown={"musiq": 0.0}, error=error)


class Encoder(Protocol):
    logit_scale: float
    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


Quality = Callable[[Path], float]  # image path -> MUSIQ score, 0-100


class OpenClipEncoder:
    """Thin wrapper around open_clip returning L2-normalised numpy embeddings."""
    logit_scale = CLIP_LOGIT_SCALE

    def __init__(self, model_name: str, pretrained: str, device: str = "cpu"):
        import open_clip
        import torch

        self.torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def _norm(self, x):
        return (x / x.norm(dim=-1, keepdim=True)).cpu().numpy()

    def encode_images(self, images):
        with self.torch.no_grad():
            batch = self.torch.stack([self.preprocess(im) for im in images]).to(self.device)
            return self._norm(self.model.encode_image(batch).float())

    def encode_texts(self, texts):
        with self.torch.no_grad():
            return self._norm(self.model.encode_text(self.tokenizer(list(texts)).to(self.device)).float())


class MusiqScorer:
    """MUSIQ through pyiqa. The import is lazy so the rest of the package (scraper, tests with
    fakes) works without pyiqa installed."""

    def __init__(self, device: str = "cpu"):
        import pyiqa
        import torch

        self.torch = torch
        self.metric = pyiqa.create_metric(MUSIQ_METRIC, device=device)

    def __call__(self, path: Path) -> float:
        with self.torch.no_grad():
            return float(self.metric(str(path)))


def raw_metrics(bgr: np.ndarray) -> dict[str, float]:
    """Classical CV reference values (not part of the score)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "laplacian_var": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "mean_brightness": round(float(gray.mean()), 1),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=-1, keepdims=True)


class ImageScorer:
    def __init__(self, settings: Settings, device: str = "cpu",
                 encoder: Optional[Encoder] = None, quality: Optional[Quality] = None):
        self.settings = settings
        self.device = device
        self._encoder = encoder
        self._quality = quality
        self._view_text = None  # (5, d) one centroid per view type
        self._body_text = None  # (4, d) one centroid per body type

    def _centroids(self, groups: dict[str, list[str]]) -> np.ndarray:
        rows = []
        for prompts in groups.values():
            c = self._encoder.encode_texts(prompts).mean(axis=0)
            rows.append(c / np.linalg.norm(c))
        return np.stack(rows)

    def load(self) -> "ImageScorer":
        if self._encoder is None:
            log.info("loading CLIP %s/%s", self.settings.clip_model, self.settings.clip_pretrained)
            self._encoder = OpenClipEncoder(self.settings.clip_model, self.settings.clip_pretrained, self.device)
        if self._quality is None:
            log.info("loading MUSIQ")
            self._quality = MusiqScorer(self.device)
        if self._view_text is None:
            self._view_text = self._centroids(VIEW_PROMPTS)
            self._body_text = self._centroids(BODY_PROMPTS)
        return self

    def _score_one(self, path: Path, metrics: dict[str, float], view_p: np.ndarray, body_p: np.ndarray) -> ScoreResult:
        try:
            raw = min(max(float(self._quality(path)), 0.0), 100.0)
        except Exception as e:  # one image the model chokes on must not kill the batch
            return ScoreResult.failed(f"quality model failed: {type(e).__name__}: {e}")
        views = list(VIEW_PROMPTS)
        k = int(view_p.argmax())
        conf = float(view_p[k])
        is_exterior = views[k] == "exterior" and conf >= MIN_VIEW_CONFIDENCE
        return ScoreResult(
            score=round(raw / 10, 1), breakdown={"musiq": round(raw, 1)}, raw_metrics=metrics,
            view_type=views[k], view_confidence=round(conf, 3),
            body_probs=dict(zip(BODY_PROMPTS, map(float, body_p))) if is_exterior else {},
        )

    def score_paths(self, paths: Sequence[Path], batch_size: int = 16, on_progress=None) -> list[ScoreResult]:
        self.load()
        results: list[Optional[ScoreResult]] = [None] * len(paths)
        for start in range(0, len(paths), batch_size):
            chunk = list(enumerate(paths))[start:start + batch_size]
            ok_idx, images, metrics = [], [], []
            for i, p in chunk:
                try:
                    bgr = cv2.imread(str(p))
                    if bgr is None:
                        raise ValueError("unreadable or missing image file")
                    images.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                    metrics.append(raw_metrics(bgr))
                    ok_idx.append(i)
                except Exception as e:
                    results[i] = ScoreResult.failed(f"{type(e).__name__}: {e}")
            if images:
                emb = self._encoder.encode_images(images)  # one CLIP pass feeds both classifiers
                view_p = _softmax(self._encoder.logit_scale * emb @ self._view_text.T)
                body_p = _softmax(self._encoder.logit_scale * emb @ self._body_text.T)
                for n, i in enumerate(ok_idx):
                    results[i] = self._score_one(Path(paths[i]), metrics[n], view_p[n], body_p[n])
            if on_progress:
                on_progress(min(start + batch_size, len(paths)), len(paths))
        return results  # type: ignore[return-value]
