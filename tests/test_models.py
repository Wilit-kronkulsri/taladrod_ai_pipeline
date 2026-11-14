from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from taladrod.config import Settings
from taladrod.models import AdRecord, Dataset, ImageRecord


def _ad(**kw):
    base = dict(
        ad_id="1", brand="TOYOTA", model="HILUX REVO", year=2018, full_title="2018 TOYOTA HILUX REVO",
        price_thb=459000, mileage_km=None, ad_source_url="https://x/cardet.aspx?cid=1",
        scraped_at=datetime.now(timezone.utc), images=[],
    )
    base.update(kw)
    return AdRecord(**base)


def test_ad_record_loads_without_clip_fields():  # JSON written before the CLIP cross-check existed
    ad = _ad()
    assert ad.category_clip is None and ad.category_clip_confidence is None and ad.category_mismatch is None
    old = ad.model_dump(mode="json", exclude={"category_clip", "category_clip_confidence", "category_mismatch"})
    assert AdRecord.model_validate(old).category_mismatch is None


def test_quality_score_bounds():
    with pytest.raises(ValidationError):
        ImageRecord(index=1, local_path=None, source_url="u", quality_score=10.5)
    with pytest.raises(ValidationError):
        ImageRecord(index=1, local_path=None, source_url="u", quality_score=-0.1)
    assert ImageRecord(index=1, local_path=None, source_url="u", quality_score=10.0).quality_score == 10.0


def test_category_literal():
    with pytest.raises(ValidationError):
        _ad(category="รถบรรทุก")
    assert _ad(category="กระบะ", category_source="dictionary").category == "กระบะ"


def test_dataset_roundtrip_keeps_thai():
    ds = Dataset(meta={"x": 1}, ads=[_ad(full_title="รถบ้าน มือเดียว")])
    txt = ds.model_dump_json()
    assert Dataset.model_validate_json(txt).ads[0].full_title == "รถบ้าน มือเดียว"


def test_settings_paths_and_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("TALADROD_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("TALADROD_MAX_ADS", "7")
    s = Settings.from_env()
    assert s.max_ads == 7
    assert s.images_dir == tmp_path / "images"
    assert s.raw_path == tmp_path / "raw_ads.jsonl"
    assert s.final_path == tmp_path / "taladrod_data.json"
    with pytest.raises(ValidationError):
        Settings(max_ads=101)
