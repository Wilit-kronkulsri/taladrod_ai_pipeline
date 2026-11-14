"""Smoke-test the dashboard on edge-case datasets (no browser needed)."""
from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from taladrod.models import AdRecord, Dataset, ImageRecord

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def ad(images, price=1, year=2020):
    return AdRecord(ad_id="1", brand="B", model="M", year=year, full_title="2020 B M", price_thb=price,
                    mileage_km=None, ad_source_url="u", scraped_at=datetime.now(timezone.utc), images=images)


@pytest.mark.parametrize("ads", [
    [],                                                   # listing blocked on first run
    [ad([])],                                             # ad with zero images
    [ad([ImageRecord(index=1, local_path=None, source_url="u", quality_score=0.0, error="x")],
        price=None, year=None)],                          # only failed images, no price/year
])
def test_app_handles_edge_datasets(tmp_path, monkeypatch, ads):
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "taladrod_data.json").write_text(
        Dataset(meta={"selection_strategy": "s"}, ads=ads).model_dump_json(), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


def test_app_shows_category_mismatch(tmp_path, monkeypatch):
    flagged = ad([]).model_copy(update={"category": "เก๋ง", "category_clip": "กระบะ",
                                        "category_clip_confidence": 0.7, "category_mismatch": True})
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "taladrod_data.json").write_text(
        Dataset(meta={"selection_strategy": "s"}, ads=[flagged]).model_dump_json(), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    [m] = [m for m in at.metric if m.label == "ประเภทไม่ตรงกับรูป"]
    assert m.value == "1"


def test_app_shows_view_type_of_each_photo(tmp_path, monkeypatch):
    shot = ImageRecord(index=1, local_path="output/images/1_01.jpg", source_url="u", quality_score=7.0,
                       quality_breakdown={"musiq": 70.0}, view_type="interior", view_confidence=0.9)
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "taladrod_data.json").write_text(
        Dataset(meta={"selection_strategy": "s"}, ads=[ad([shot])]).model_dump_json(), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("ภายใน" in c.value and "MUSIQ 70.0" in c.value for c in at.caption)


# ----------------------------------------------------------- run picker / new run / archive warnings
from datetime import datetime as _dt  # noqa: E402

from taladrod.archive import archive_current  # noqa: E402
from taladrod.config import Settings  # noqa: E402
from tests.test_archive import make_run  # noqa: E402


def _app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TALADROD_OUTPUT_DIR", raising=False)
    return AppTest.from_file(APP, default_timeout=60)


def _sidebar_text(at) -> str:
    return " | ".join(e.value for e in list(at.sidebar.caption) + list(at.sidebar.warning))


def test_sidebar_says_run_all_will_start_a_new_run(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    make_run(Settings(output_dir=Path("output")), 3)
    at.run()
    assert "กด Run all จะทำต่อจากเดิม" in _sidebar_text(at)          # 3 of 100: resume
    at.sidebar.slider[0].set_value(3).run()                          # user lowers the cap: now complete
    assert not at.exception, [e.value for e in at.exception]
    assert "จะเก็บชุดนี้ไว้ แล้วดึงชุดใหม่ 3 ประกาศ" in _sidebar_text(at)


def test_sidebar_lists_archives_and_warns_about_deletion(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    s = Settings(output_dir=Path("output"), max_ads=1)
    for day in (10, 11, 12):
        make_run(s, 1)
        archive_current(s, now=_dt(2026, 11, day, 9, 0))
    make_run(s, 1)
    at.run()
    at.sidebar.slider[0].set_value(1).run()
    text = _sidebar_text(at)
    assert "ชุดเก่า 3 ชุด" in text
    assert "2026-11-10_0900" in " | ".join(w.value for w in at.sidebar.warning)


def test_viewing_an_archive_disables_run_buttons(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    s = Settings(output_dir=Path("output"), max_ads=2)
    make_run(s, 2)
    archive_current(s, now=_dt(2026, 11, 13, 9, 0))
    make_run(s, 1)
    at.run()
    assert [m.value for m in at.metric if m.label == "ประกาศ"] == ["1"]
    assert not any(b.disabled for b in at.sidebar.button)
    at.sidebar.selectbox[0].set_value("2026-11-13_0900").run()
    assert not at.exception, [e.value for e in at.exception]
    assert [m.value for m in at.metric if m.label == "ประกาศ"] == ["2"]
    assert all(b.disabled for b in at.sidebar.button)


def test_archive_without_json_shows_no_data(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    s = Settings(output_dir=Path("output"), max_ads=1)
    make_run(s, 1)
    s.final_path.unlink()                         # archived before it was ever scored
    archive_current(s, now=_dt(2026, 11, 13, 9, 0))
    make_run(s, 1)
    at.run()
    at.sidebar.selectbox[0].set_value("2026-11-13_0900").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("ยังไม่มีข้อมูล" in i.value for i in at.info)


def test_sidebar_never_shows_more_ads_than_the_cap_and_explains_every_state(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()
    assert "ยังไม่มีข้อมูลชุดปัจจุบัน กด Run all เพื่อเริ่มดึง" in _sidebar_text(at)     # no checkpoint at all
    s = Settings(output_dir=Path("output"), max_ads=2)
    make_run(s, 3)                                    # checkpoint holds more ads than the cap (it happened: 110)
    archive_current(s, now=_dt(2026, 11, 13, 9, 0))
    make_run(s, 3)
    at.run()
    at.sidebar.slider[0].set_value(2).run()
    text = _sidebar_text(at)
    assert "ชุดปัจจุบันครบแล้ว" in text and "ครบ 3" not in text
    at.sidebar.selectbox[0].set_value("2026-11-13_0900").run()
    assert "กำลังดูชุดเก่า" in _sidebar_text(at) and "กด Run all จะเก็บ" not in _sidebar_text(at)
