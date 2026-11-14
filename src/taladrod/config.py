"""Run settings. Every value can be overridden with an env var prefixed ``TALADROD_``
(e.g. ``TALADROD_MAX_ADS=20``) or by passing keyword arguments."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

MAX_ADS_CAP = 100  # the brief: at most 100 ads per run
START_URL = "https://www.taladrod.com/w40/isch/schc.aspx?fno:new"

SELECTION_STRATEGY = (
    "Sequential: take ads in the exact order the 'รถมาใหม่วันนี้' (fno:new) page shows them "
    "(newest first), walking the result list page by page (10 per page, the size the site's own "
    "infinite scroll uses) until MAX_ADS (<=100) ads are collected or the results run out. "
    "Ads already in raw_ads.jsonl count toward MAX_ADS and are not fetched again."
)


class Settings(BaseModel):
    start_url: str = START_URL
    max_ads: int = Field(default=MAX_ADS_CAP, ge=0, le=MAX_ADS_CAP)
    page_size: int = Field(default=10, ge=1)
    delay_range: tuple[float, float] = (1.5, 3.0)
    retries: int = Field(default=4, ge=1)
    backoff_base: float = 1.0           # seconds; waits 1, 2, 4, 8 ...
    challenge_wait: float = 30.0        # extra wait after an AWS WAF challenge
    max_consecutive_failures: int = Field(default=5, ge=1)  # e.g. a WAF 403 block on every ad
    timeout: float = 20.0
    output_dir: Path = Path("output")
    keep_archives: int = Field(default=3, ge=1)  # finished runs kept in output_archive/
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    selection_strategy: str = SELECTION_STRATEGY
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    )

    @property
    def images_dir(self) -> Path:
        return self.output_dir / "images"

    @property
    def archive_dir(self) -> Path:
        return self.output_dir.parent / f"{self.output_dir.name}_archive"

    @property
    def lock_path(self) -> Path:
        # next to output/, not inside it: output/ itself gets moved when a run is archived
        return self.output_dir.parent / f"{self.output_dir.name}.running"

    @property
    def raw_path(self) -> Path:
        return self.output_dir / "raw_ads.jsonl"

    @property
    def final_path(self) -> Path:
        return self.output_dir / "taladrod_data.json"

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        values: dict = {}
        for name in ("start_url", "max_ads", "output_dir", "clip_model", "clip_pretrained"):
            env = os.environ.get(f"TALADROD_{name.upper()}")
            if env is not None:
                values[name] = env
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)
