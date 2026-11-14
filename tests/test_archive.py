from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from taladrod.archive import (AlreadyRunningError, ArchiveError, archive_current, dir_size_mb, doomed_archives,
                              is_complete, list_archives, run_lock, start_new_run_if_complete)
from taladrod.config import Settings
from taladrod.models import AdRecord, Dataset, ImageRecord
from taladrod.pipeline import load_dataset
from taladrod.scraper import load_checkpoint


def make_run(s: Settings, n_ads: int) -> None:
    """A finished run in ``s.output_dir``: n ads, one (fake) image file each, checkpoint + JSON."""
    s.images_dir.mkdir(parents=True, exist_ok=True)
    ads = []
    for n in range(1, n_ads + 1):
        img = s.images_dir / f"{n}_01.jpg"
        Image.new("RGB", (8, 8)).save(img)  # a real (tiny) picture: the dashboard opens it
        ads.append(AdRecord(ad_id=str(n), brand="B", model="M", year=2020, full_title="2020 B M", price_thb=1,
                            mileage_km=None, ad_source_url="u", scraped_at=datetime.now(timezone.utc),
                            cover_image=img.as_posix(),
                            images=[ImageRecord(index=1, local_path=img.as_posix(), source_url="u",
                                                quality_score=7.0)]))
    s.raw_path.write_text("".join(a.model_dump_json() + "\n" for a in ads), encoding="utf-8")
    s.final_path.write_text(Dataset(meta={"ad_count": n_ads}, ads=ads).model_dump_json(), encoding="utf-8")


def test_is_complete(tmp_path):
    s = Settings(output_dir=tmp_path / "output", max_ads=2)
    assert not is_complete(s)                       # no checkpoint
    make_run(s, 1)
    assert not is_complete(s)                       # stopped half-way: resume, do not archive
    make_run(s, 2)
    assert is_complete(s)
    assert not is_complete(s.model_copy(update={"max_ads": 0}))


def test_archive_moves_run_and_fixes_image_paths(tmp_path):
    s = Settings(output_dir=tmp_path / "output", max_ads=2)
    make_run(s, 2)
    dest = archive_current(s, now=datetime(2026, 11, 14, 19, 13))
    assert dest == tmp_path / "output_archive" / "2026-11-14_1913"
    assert not s.output_dir.exists()
    ds = load_dataset(dest / "taladrod_data.json")
    paths = [i.local_path for a in ds.ads for i in a.images] + [a.cover_image for a in ds.ads]
    assert paths and all(Path(p).exists() for p in paths)          # still openable after the move
    assert all(Path(i.local_path).exists()
               for a in load_checkpoint(dest / "raw_ads.jsonl").values() for i in a.images)


def test_archive_keeps_three_newest(tmp_path):
    s = Settings(output_dir=tmp_path / "output", max_ads=1)
    for day in (10, 11, 12, 13):
        make_run(s, 1)
        archive_current(s, now=datetime(2026, 11, day, 9, 0))
    assert [p.name for p in list_archives(s)] == ["2026-11-13_0900", "2026-11-12_0900", "2026-11-11_0900"]


def test_doomed_archives_names_what_the_next_run_deletes(tmp_path):   # shown to the user before the click
    s = Settings(output_dir=tmp_path / "output", max_ads=1)
    for day in (10, 11, 12):
        make_run(s, 1)
        archive_current(s, now=datetime(2026, 11, day, 9, 0))
    assert [p.name for p in doomed_archives(s)] == ["2026-11-10_0900"]
    assert doomed_archives(s.model_copy(update={"keep_archives": 5})) == []


def test_two_archives_in_the_same_minute_do_not_collide(tmp_path):
    s = Settings(output_dir=tmp_path / "output", max_ads=1)
    t = datetime(2026, 11, 14, 19, 13)
    make_run(s, 1)
    a = archive_current(s, now=t)
    make_run(s, 1)
    b = archive_current(s, now=t)
    assert a != b and a.exists() and b.exists()


def test_failed_move_leaves_everything_in_place(tmp_path, monkeypatch):
    s = Settings(output_dir=tmp_path / "output", max_ads=1)
    make_run(s, 1)

    def locked(self, target):
        raise PermissionError("in use")

    monkeypatch.setattr(Path, "rename", locked)
    with pytest.raises(ArchiveError, match="output"):
        archive_current(s)
    monkeypatch.undo()
    assert s.final_path.exists() and list_archives(s) == []


def test_start_new_run_only_when_complete(tmp_path):
    s = Settings(output_dir=tmp_path / "output", max_ads=2)
    make_run(s, 1)
    assert start_new_run_if_complete(s) is None and s.raw_path.exists()
    make_run(s, 2)
    assert start_new_run_if_complete(s) is not None and not s.raw_path.exists()


def test_run_lock_refuses_second_run_and_cleans_up(tmp_path):
    s = Settings(output_dir=tmp_path / "output")
    with run_lock(s):
        assert s.lock_path.exists()
        with pytest.raises(AlreadyRunningError, match="output.running"):
            with run_lock(s):
                pass
        assert s.lock_path.exists()          # the refused run must not remove the first run's lock
    assert not s.lock_path.exists()
    with pytest.raises(ValueError):
        with run_lock(s):
            raise ValueError("boom")
    assert not s.lock_path.exists()          # released on failure too


def test_dir_size_mb(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.bin").write_bytes(b"0" * 1_048_576)
    assert dir_size_mb(tmp_path) == pytest.approx(1.0)
    assert dir_size_mb(tmp_path / "missing") == 0.0


def test_folders_the_user_put_in_output_archive_are_not_archives(tmp_path):
    s = Settings(output_dir=tmp_path / "output", max_ads=1)
    notes = s.archive_dir / "0_notes"
    backup = s.archive_dir / "my_backup"
    for d in (notes, backup):
        d.mkdir(parents=True)
    for day in (10, 11, 12, 13):
        make_run(s, 1)
        archive_current(s, now=datetime(2026, 11, day, 9, 0))
    assert [p.name for p in list_archives(s)] == ["2026-11-13_0900", "2026-11-12_0900", "2026-11-11_0900"]
    assert notes.exists() and backup.exists()        # never counted, never deleted


class DeadSite:
    def get_html(self, url):
        raise ConnectionError("no internet")


def test_new_run_is_not_started_when_the_site_is_unreachable(tmp_path):
    # otherwise the finished run is moved away and output/ is left empty with nothing to replace it
    s = Settings(output_dir=tmp_path / "output", max_ads=1)
    make_run(s, 1)
    with pytest.raises(ArchiveError, match="no internet"):
        start_new_run_if_complete(s, client=DeadSite())
    assert s.final_path.exists() and list_archives(s) == []


def test_keep_archives_must_be_at_least_one():
    with pytest.raises(ValueError):
        Settings(keep_archives=0)
