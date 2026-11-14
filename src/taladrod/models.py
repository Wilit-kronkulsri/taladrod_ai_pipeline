"""Pydantic schema for the JSON deliverable (``output/taladrod_data.json``)."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

Category = Literal["เก๋ง", "กระบะ", "SUV/PPV", "รถตู้/MPV", "อื่นๆ"]
CategorySource = Literal["site", "dictionary", "fallback"]
ViewType = Literal["exterior", "interior", "engine", "detail", "other"]


class ImageRecord(BaseModel):
    index: int = Field(ge=1)
    local_path: Optional[str]
    source_url: str
    quality_score: Optional[float] = Field(default=None, ge=0, le=10)
    quality_breakdown: Optional[dict[str, float]] = None  # {"musiq": raw 0-100 score}
    raw_metrics: Optional[dict[str, float]] = None
    view_type: Optional[ViewType] = None  # what the photo shows, from CLIP zero-shot
    view_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    phash: Optional[str] = None
    is_duplicate_of: Optional[str] = None
    error: Optional[str] = None


class AdRecord(BaseModel):
    ad_id: str
    brand: Optional[str]
    model: Optional[str]
    year: Optional[int]
    full_title: str
    category: Optional[Category] = None
    category_source: Optional[CategorySource] = None
    # CLIP's guess from the photos. A review flag only: it never overwrites ``category``.
    category_clip: Optional[Category] = None
    category_clip_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    category_mismatch: Optional[bool] = None
    price_thb: Optional[int]
    mileage_km: Optional[int]
    ad_source_url: str
    body_type_raw: Optional[str] = None
    seller_headline: Optional[str] = None
    scraped_at: datetime
    cover_image: Optional[str] = None
    images: list[ImageRecord] = Field(default_factory=list)


class Dataset(BaseModel):
    meta: dict
    ads: list[AdRecord]
