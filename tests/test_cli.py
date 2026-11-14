import json
from datetime import datetime, timezone

import pytest

from taladrod import cli
from taladrod.models import AdRecord, Dataset, ImageRecord


def test_all_with_zero_ads_does_not_touch_network(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PoliteClient", lambda *a, **k: pytest.fail("network used"))
    assert cli.main(["all", "--max-ads", "0", "--output", str(tmp_path)]) == 0
    assert (tmp_path / "taladrod_data.json").exists()


def test_max_ads_cap(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["all", "--max-ads", "101"])
    assert e.value.code != 0
    assert "100" in capsys.readouterr().err


def test_sanity_report(tmp_path):
    imgs = [ImageRecord(index=i, local_path=f"output/images/1_{i:02d}.jpg", source_url=f"u{i}",
                        quality_score=float(i), quality_breakdown={"musiq": i * 10.0}, view_type="exterior")
            for i in range(1, 8)]
    ad = AdRecord(ad_id="1", brand="B", model="M", year=2020, full_title="2020 B M", price_thb=1, mileage_km=1,
                  ad_source_url="u", scraped_at=datetime.now(timezone.utc), images=imgs)
    (tmp_path / "taladrod_data.json").write_text(Dataset(meta={}, ads=[ad]).model_dump_json(), encoding="utf-8")
    out = tmp_path / "iqa_sanity.md"
    assert cli.main(["sanity", "--output", str(tmp_path), "--report", str(out), "-n", "2"]) == 0
    text = out.read_text(encoding="utf-8")
    assert "1_07.jpg" in text and "1_01.jpg" in text and "1_04.jpg" not in text
    assert "| MUSIQ | view |" in text and "| 70.0 | exterior |" in text


def test_score_without_checkpoint_keeps_existing_results(tmp_path, capsys):
    (tmp_path / "taladrod_data.json").write_text("KEEP ME", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        cli.main(["score", "--output", str(tmp_path)])
    assert e.value.code != 0 and "raw_ads.jsonl" in str(e.value)
    assert (tmp_path / "taladrod_data.json").read_text(encoding="utf-8") == "KEEP ME"


# ----------------------------------------------------------- new run / archive / lock
from taladrod.config import Settings  # noqa: E402
from tests.test_archive import make_run  # noqa: E402


class Calls(list):
    site: dict


@pytest.fixture
def stub_phases(monkeypatch):
    """Replace both phases with recorders so these tests never touch the network or the models."""
    calls = Calls()
    site = {"up": True}

    class Site:
        def get_html(self, url):
            if not site["up"]:
                raise ConnectionError("no internet")
            return ""

    calls.site = site
    monkeypatch.setattr(cli, "PoliteClient", lambda *a, **k: Site())
    monkeypatch.setattr(cli, "run_phase1", lambda s, client=None, on_progress=None: calls.append("scrape") or [])
    monkeypatch.setattr(cli, "run_phase2", lambda s, on_progress=None: calls.append("score")
                        or Dataset(meta={"ad_count": 0, "image_count": 0}, ads=[]))
    return calls


def test_all_archives_a_complete_run_first(tmp_path, stub_phases, capsys):
    out = tmp_path / "output"
    make_run(Settings(output_dir=out), 2)
    assert cli.main(["all", "--max-ads", "2", "--output", str(out)]) == 0
    assert len(list((tmp_path / "output_archive").glob("*/raw_ads.jsonl"))) == 1
    assert stub_phases == ["scrape", "score"]
    assert "archived" in capsys.readouterr().out


def test_all_resumes_an_incomplete_run(tmp_path, stub_phases):
    out = tmp_path / "output"
    make_run(Settings(output_dir=out), 1)
    assert cli.main(["all", "--max-ads", "2", "--output", str(out)]) == 0
    assert not (tmp_path / "output_archive").exists() and (out / "raw_ads.jsonl").exists()


def test_scrape_and_score_never_archive(tmp_path, stub_phases):
    out = tmp_path / "output"
    make_run(Settings(output_dir=out), 2)
    assert cli.main(["scrape", "--max-ads", "2", "--output", str(out)]) == 0
    assert cli.main(["score", "--output", str(out)]) == 0
    assert not (tmp_path / "output_archive").exists()


def test_refuses_when_already_running(tmp_path, stub_phases):
    out = tmp_path / "output"
    make_run(Settings(output_dir=out), 2)
    (tmp_path / "output.running").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        cli.main(["all", "--max-ads", "2", "--output", str(out)])
    assert e.value.code != 0 and "output.running" in str(e.value)
    assert stub_phases == [] and not (tmp_path / "output_archive").exists()
    assert (tmp_path / "output.running").exists()  # someone else's lock is left alone


def test_all_does_not_archive_when_the_site_is_unreachable(tmp_path, stub_phases):
    out = tmp_path / "output"
    make_run(Settings(output_dir=out), 2)
    stub_phases.site["up"] = False
    with pytest.raises(SystemExit) as e:
        cli.main(["all", "--max-ads", "2", "--output", str(out)])
    assert e.value.code != 0 and "no internet" in str(e.value)
    assert stub_phases == [] and not list((tmp_path / "output_archive").glob("*"))
    assert (out / "taladrod_data.json").exists()


def test_all_says_which_old_archives_it_deleted(tmp_path, stub_phases, capsys):
    from datetime import datetime as dt
    from taladrod.archive import archive_current
    s = Settings(output_dir=tmp_path / "output", max_ads=1)
    for day in (10, 11, 12):
        make_run(s, 1)
        archive_current(s, now=dt(2026, 11, day, 9, 0))
    make_run(s, 1)
    assert cli.main(["all", "--max-ads", "1", "--output", str(s.output_dir)]) == 0
    assert "deleted old archive 2026-11-10_0900" in capsys.readouterr().out
