import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from taladrod.config import Settings
from taladrod.enrich import clip_category, mark_duplicates, pick_cover
from taladrod.iqa import ScoreResult
from taladrod.models import AdRecord, Dataset, ImageRecord
from taladrod.pipeline import NoCheckpointError, run_phase2


def img_file(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    a = cv2.resize(rng.integers(0, 255, (12, 16, 3), dtype=np.uint8), (320, 240), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), a)
    return path


def make_ad(ad_id, title, images):
    return AdRecord(ad_id=ad_id, brand="TOYOTA", model=title.split()[2], year=2018, full_title=title,
                    price_thb=1, mileage_km=None, ad_source_url=f"https://x/cardet.aspx?cid={ad_id}",
                    scraped_at=datetime.now(timezone.utc), images=images)


def test_mark_duplicates_across_ads(tmp_path):
    a = img_file(tmp_path / "1_01.jpg", 1)
    b = img_file(tmp_path / "1_02.jpg", 2)
    c = tmp_path / "2_01.jpg"
    shutil.copy(a, c)  # same photo reused in another ad
    ads = [
        make_ad("1", "2018 TOYOTA HILUX", [ImageRecord(index=1, local_path=str(a), source_url="a"),
                                           ImageRecord(index=2, local_path=str(b), source_url="b")]),
        make_ad("2", "2018 TOYOTA VIOS", [ImageRecord(index=1, local_path=str(c), source_url="c"),
                                          ImageRecord(index=2, local_path=None, source_url="d", error="x")]),
    ]
    mark_duplicates(ads)
    assert ads[0].images[0].phash and ads[0].images[0].is_duplicate_of is None
    assert ads[0].images[1].is_duplicate_of is None
    assert ads[1].images[0].is_duplicate_of == str(a)
    assert ads[1].images[1].phash is None


def test_pick_cover_skips_errors_and_duplicates():
    imgs = [
        ImageRecord(index=1, local_path="p1", source_url="1", quality_score=9.0, error="bad"),
        ImageRecord(index=2, local_path="p2", source_url="2", quality_score=8.0, is_duplicate_of="x"),
        ImageRecord(index=3, local_path="p3", source_url="3", quality_score=6.0),
        ImageRecord(index=4, local_path="p4", source_url="4", quality_score=7.5),
    ]
    assert pick_cover(make_ad("1", "2018 TOYOTA VIOS", imgs)) == "p4"
    assert pick_cover(make_ad("1", "2018 TOYOTA VIOS", [])) is None


P_PICKUP = {"เก๋ง": 0.1, "กระบะ": 0.7, "SUV/PPV": 0.1, "รถตู้/MPV": 0.1}
P_SEDAN = {"เก๋ง": 0.9, "กระบะ": 0.05, "SUV/PPV": 0.03, "รถตู้/MPV": 0.02}


def test_clip_category_averages_images():
    # mean: เก๋ง 0.5, กระบะ 0.375 -> เก๋ง
    assert clip_category("เก๋ง", [P_SEDAN, P_PICKUP]) == ("เก๋ง", 0.5, False)


def test_clip_category_flags_mismatch():
    assert clip_category("เก๋ง", [P_PICKUP]) == ("กระบะ", 0.7, True)


def test_clip_category_no_flag_for_other_or_missing():
    assert clip_category("อื่นๆ", [P_PICKUP]) == ("กระบะ", 0.7, False)
    assert clip_category(None, [P_PICKUP]) == ("กระบะ", 0.7, False)


def test_clip_category_low_confidence_is_not_flagged():
    weak = {"เก๋ง": 0.2, "กระบะ": 0.4, "SUV/PPV": 0.2, "รถตู้/MPV": 0.2}
    assert clip_category("เก๋ง", [weak]) == ("กระบะ", 0.4, False)


def test_pick_cover_prefers_exterior_shots():
    imgs = [ImageRecord(index=1, local_path="in", source_url="1", quality_score=9.0, view_type="interior"),
            ImageRecord(index=2, local_path="ex", source_url="2", quality_score=6.0, view_type="exterior")]
    assert pick_cover(make_ad("1", "2018 TOYOTA VIOS", imgs)) == "ex"
    assert pick_cover(make_ad("1", "2018 TOYOTA VIOS", imgs[:1])) == "in"  # no exterior: best of the rest


def test_clip_category_no_images():
    assert clip_category("เก๋ง", []) == (None, None, None)
    assert clip_category("เก๋ง", [{}, {}]) == (None, None, None)


class FakeIQA:
    def score_paths(self, paths, batch_size=16, on_progress=None):
        out = []
        for p in paths:
            if not Path(p).exists():
                out.append(ScoreResult.failed("missing"))
            else:
                out.append(ScoreResult(score=7.0, breakdown={"musiq": 70.0},
                                       raw_metrics={"laplacian_var": 1.0, "mean_brightness": 2.0},
                                       view_type="exterior", view_confidence=0.9, body_probs=P_PICKUP))
        return out


def test_run_phase2_writes_valid_json(tmp_path):
    s = Settings(output_dir=tmp_path)
    s.images_dir.mkdir(parents=True)
    p1 = img_file(s.images_dir / "1_01.jpg", 1)
    ads = [
        make_ad("1", "2018 TOYOTA HILUX REVO", [ImageRecord(index=1, local_path=str(p1), source_url="u1"),
                                                ImageRecord(index=2, local_path=None, source_url="u2",
                                                            error="download failed")]),
        make_ad("2", "2019 FOO BAR 1.0", []),
    ]
    ads[1].seller_headline = "รถบ้าน มือเดียว"
    s.raw_path.write_text("\n".join(a.model_dump_json() for a in ads) + "\n", encoding="utf-8")

    ds = run_phase2(s, iqa=FakeIQA())

    raw = s.final_path.read_text(encoding="utf-8")
    assert "รถบ้าน มือเดียว" in raw and "\\u0e" not in raw
    loaded = Dataset.model_validate_json(raw)
    assert loaded.meta["ad_count"] == 2 and loaded.meta["image_count"] == 2
    assert loaded.meta["iqa_method"].startswith("MUSIQ")
    assert set(loaded.meta["view_prompts"]) == {"exterior", "interior", "engine", "detail", "other"}
    assert loaded.meta["selection_strategy"]
    a1, a2 = loaded.ads
    assert (a1.category, a1.category_source) == ("กระบะ", "dictionary")
    assert (a2.category, a2.category_source) == ("อื่นๆ", "fallback")
    assert all(im.quality_score is not None for im in a1.images)
    assert a1.images[0].quality_score == 7.0
    assert a1.images[0].quality_breakdown == {"musiq": 70.0}
    assert (a1.images[0].view_type, a1.images[0].view_confidence) == ("exterior", 0.9)
    assert a1.images[1].quality_breakdown == {"musiq": 0.0} and a1.images[1].view_type is None
    assert a1.images[1].quality_score == 0.0 and a1.images[1].error == "download failed"
    assert a1.cover_image == str(p1)
    assert ds.meta["ad_count"] == 2
    # CLIP body-type cross-check is a flag only: the dictionary category is never overwritten
    assert (a1.category_clip, a1.category_clip_confidence, a1.category_mismatch) == ("กระบะ", 0.7, False)
    assert (a2.category_clip, a2.category_clip_confidence, a2.category_mismatch) == (None, None, None)
    assert set(loaded.meta["category_clip_prompts"]) == {"เก๋ง", "กระบะ", "SUV/PPV", "รถตู้/MPV"}


def test_run_phase2_flags_mismatch_without_changing_category(tmp_path):
    s = Settings(output_dir=tmp_path)
    s.images_dir.mkdir(parents=True)
    p1 = img_file(s.images_dir / "1_01.jpg", 1)
    ad = make_ad("1", "2018 TOYOTA VIOS 1.5", [ImageRecord(index=1, local_path=str(p1), source_url="u1")])
    s.raw_path.write_text(ad.model_dump_json() + "\n", encoding="utf-8")
    [a] = run_phase2(s, iqa=FakeIQA()).ads
    assert a.category == "เก๋ง"
    assert (a.category_clip, a.category_mismatch) == ("กระบะ", True)


def test_run_phase2_ad_with_no_downloaded_images(tmp_path):
    s = Settings(output_dir=tmp_path)
    s.images_dir.mkdir(parents=True)
    ad = make_ad("1", "2018 TOYOTA VIOS 1.5", [ImageRecord(index=1, local_path=None, source_url="u1", error="x")])
    s.raw_path.write_text(ad.model_dump_json() + "\n", encoding="utf-8")
    [a] = run_phase2(s, iqa=FakeIQA()).ads
    assert (a.category_clip, a.category_clip_confidence, a.category_mismatch) == (None, None, None)


def test_avg_quality_ignores_failed_images():
    from taladrod.enrich import avg_quality
    imgs = [ImageRecord(index=1, local_path="a", source_url="1", quality_score=8.0),
            ImageRecord(index=2, local_path=None, source_url="2", quality_score=0.0, error="download failed"),
            ImageRecord(index=3, local_path="c", source_url="3", quality_score=6.0)]
    assert avg_quality(make_ad("1", "2018 TOYOTA VIOS", imgs)) == 7.0
    assert avg_quality(make_ad("1", "2018 TOYOTA VIOS", imgs[1:2])) is None


def test_run_phase2_caps_output_at_max_ads(tmp_path):
    # a checkpoint can hold more than MAX_ADS (two scrapers running at once, or several resumes):
    # the deliverable must still respect the cap, keeping the first ads collected
    s = Settings(output_dir=tmp_path, max_ads=2)
    s.images_dir.mkdir(parents=True)
    ads = [make_ad(str(n), "2018 TOYOTA VIOS 1.5", []) for n in (1, 2, 3)]
    s.raw_path.write_text("".join(a.model_dump_json() + "\n" for a in ads), encoding="utf-8")
    ds = run_phase2(s, iqa=FakeIQA())
    assert [a.ad_id for a in ds.ads] == ["1", "2"]
    assert ds.meta["ad_count"] == 2


def test_run_phase2_refuses_to_overwrite_results_when_checkpoint_is_missing(tmp_path):
    # the submission zip ships taladrod_data.json; pressing "Score" without raw_ads.jsonl must not wipe it
    s = Settings(output_dir=tmp_path)
    s.final_path.write_text("KEEP ME", encoding="utf-8")
    with pytest.raises(NoCheckpointError, match="raw_ads.jsonl"):
        run_phase2(s, iqa=FakeIQA())
    assert s.final_path.read_text(encoding="utf-8") == "KEEP ME"


def test_run_phase2_says_when_it_drops_ads_over_the_cap(tmp_path):
    s = Settings(output_dir=tmp_path, max_ads=2)
    s.images_dir.mkdir(parents=True)
    ads = [make_ad(str(n), "2018 TOYOTA VIOS 1.5", []) for n in (1, 2, 3)]
    s.raw_path.write_text("".join(a.model_dump_json() + "\n" for a in ads), encoding="utf-8")
    seen = []
    run_phase2(s, iqa=FakeIQA(), on_progress=lambda d, t, m: seen.append(m))
    assert any("checkpoint has 3 ads, using the first 2" in m for m in seen)


def test_run_phase2_with_max_ads_zero_keeps_the_whole_checkpoint(tmp_path):
    # `taladrod all --max-ads 0` means "scrape nothing", not "throw the scored data away"
    s = Settings(output_dir=tmp_path, max_ads=0)
    s.images_dir.mkdir(parents=True)
    ads = [make_ad(str(n), "2018 TOYOTA VIOS 1.5", []) for n in (1, 2, 3)]
    s.raw_path.write_text("".join(a.model_dump_json() + "\n" for a in ads), encoding="utf-8")
    assert run_phase2(s, iqa=FakeIQA()).meta["ad_count"] == 3
