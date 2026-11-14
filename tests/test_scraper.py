import json
from pathlib import Path

import pytest
import responses

from taladrod import http as http_mod
from taladrod.config import Settings
from taladrod.http import BlockedError, PoliteClient
from taladrod.scraper import load_checkpoint, run_phase1

FX = Path(__file__).parent / "fixtures"
LIST_HTML = (FX / "list_new.html").read_text(encoding="utf-8")
DETAIL_HTML = (FX / "detail_15img.html").read_text(encoding="utf-8")
JPEG = b"\xff\xd8\xff\xe0fakejpeg"


class FakeClient:
    """Serves fixtures instead of the network. Every detail page returns the 15-image fixture with
    its ad id rewritten so ads stay distinct."""

    def __init__(self, fail_image_urls=(), block_after=None):
        self.detail_calls: list[str] = []
        self.fail_image_urls = set(fail_image_urls)
        self.block_after = block_after

    def get_html(self, url: str) -> str:
        if "schc.aspx" in url:
            return LIST_HTML
        if self.block_after is not None and len(self.detail_calls) >= self.block_after:
            raise BlockedError("waf challenge")
        cid = url.split("cid=")[1]
        self.detail_calls.append(cid)
        return DETAIL_HTML.replace("รหัสอ้างอิง 2932903", f"รหัสอ้างอิง {cid}")

    def get_bytes(self, url: str):
        if any(f in url for f in self.fail_image_urls):
            raise ConnectionError("boom")
        return JPEG, "image/jpeg"


def settings(tmp_path, max_ads):
    return Settings(output_dir=tmp_path, max_ads=max_ads, delay_range=(0, 0))


def test_respects_max_ads(tmp_path):
    s = settings(tmp_path, 3)
    ads = run_phase1(s, client=FakeClient())
    assert len(ads) == 3
    assert len(s.raw_path.read_text(encoding="utf-8").splitlines()) == 3
    assert [a.ad_id for a in ads] == ["2932905", "2932904", "2932903"]  # display order


def test_resume_skips_existing(tmp_path):
    run_phase1(settings(tmp_path, 2), client=FakeClient())
    c = FakeClient()
    ads = run_phase1(settings(tmp_path, 3), client=c)
    assert c.detail_calls == ["2932903"]
    assert len(ads) == 3
    assert len(load_checkpoint(settings(tmp_path, 3).raw_path)) == 3


def test_stops_when_data_runs_out(tmp_path):
    ads = run_phase1(settings(tmp_path, 100), client=FakeClient())
    assert len(ads) == 11


def test_image_failure_recorded(tmp_path):
    ads = run_phase1(settings(tmp_path, 1), client=FakeClient(fail_image_urls=["74_2.jpg"]))
    imgs = ads[0].images
    assert imgs[1].error and imgs[1].local_path is None
    assert imgs[0].error is None and Path(imgs[0].local_path).exists()


def test_filenames_unique_over_10(tmp_path):
    ads = run_phase1(settings(tmp_path, 1), client=FakeClient())
    names = [Path(i.local_path).name for i in ads[0].images]
    assert names[10] == "2932905_11.jpg"
    assert len(set(names)) == 15
    assert all((tmp_path / "images" / n).exists() for n in names)


def test_blocked_stops_cleanly_and_keeps_checkpoint(tmp_path):
    s = settings(tmp_path, 5)
    ads = run_phase1(s, client=FakeClient(block_after=2))
    assert len(ads) == 2
    assert len(load_checkpoint(s.raw_path)) == 2


def test_progress_callback(tmp_path):
    seen = []
    run_phase1(settings(tmp_path, 2), client=FakeClient(), on_progress=lambda d, t, m: seen.append((d, t)))
    assert seen[-1] == (2, 2)


# ----------------------------------------------------------- PoliteClient
@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(http_mod.time, "sleep", lambda s: None)


@responses.activate
def test_retry_then_success(no_sleep):
    url = "https://www.taladrod.com/w40/icar/cardet.aspx?cid=1"
    responses.add(responses.GET, url, status=503)
    responses.add(responses.GET, url, status=503)
    responses.add(responses.GET, url, body="<html>ok รถ</html>", status=200,
                  content_type="text/html; charset=utf-8")
    c = PoliteClient(Settings(delay_range=(0, 0)))
    assert "รถ" in c.get_html(url)
    assert len(responses.calls) == 3


@responses.activate
def test_waf_challenge_retried_then_raises_blocked(no_sleep):
    url = "https://www.taladrod.com/w40/icar/cardet.aspx?cid=1"
    for _ in range(4):
        responses.add(responses.GET, url, status=202, body="", headers={"x-amzn-waf-action": "challenge"})
    c = PoliteClient(Settings(delay_range=(0, 0), retries=4))
    with pytest.raises(BlockedError):
        c.get_html(url)
    assert len(responses.calls) == 4


@responses.activate
def test_404_not_retried(no_sleep):
    url = "https://imgc2.taladrod.com/x.jpg"
    responses.add(responses.GET, url, status=404)
    c = PoliteClient(Settings(delay_range=(0, 0)))
    with pytest.raises(Exception):
        c.get_bytes(url)
    assert len(responses.calls) == 1


# ----------------------------------------------------------- final-review fixes
class BlockOnImagesClient(FakeClient):
    def get_bytes(self, url):
        if "74_3.jpg" in url:
            raise BlockedError("waf challenge on image host")
        return JPEG, "image/jpeg"


def test_block_during_images_stops_without_saving_partial_ad(tmp_path):
    s = settings(tmp_path, 3)
    ads = run_phase1(s, client=BlockOnImagesClient())
    assert ads == []                      # ad with a blocked image is not checkpointed
    assert load_checkpoint(s.raw_path) == {}


def test_truncated_checkpoint_line_is_skipped(tmp_path):
    s = settings(tmp_path, 2)
    run_phase1(s, client=FakeClient())
    with s.raw_path.open("a", encoding="utf-8") as f:
        f.write('{"ad_id": "999", "brand": "TOY')   # killed mid-write
    assert set(load_checkpoint(s.raw_path)) == {"2932905", "2932904"}
    ads = run_phase1(settings(tmp_path, 3), client=FakeClient())
    assert len(ads) == 3


class ListingHttpErrorClient(FakeClient):
    def get_html(self, url):
        import requests
        raise requests.HTTPError("403 Forbidden")


def test_listing_http_error_returns_checkpoint(tmp_path):
    s = settings(tmp_path, 3)
    assert run_phase1(s, client=ListingHttpErrorClient()) == []


class DetailAlwaysFailsClient(FakeClient):
    def get_html(self, url):
        if "schc.aspx" in url:
            return LIST_HTML
        self.detail_calls.append(url)
        import requests
        raise requests.HTTPError("403 Forbidden")


def test_stops_after_consecutive_detail_failures(tmp_path):
    c = DetailAlwaysFailsClient()
    run_phase1(Settings(output_dir=tmp_path, max_ads=100, delay_range=(0, 0), max_consecutive_failures=3), client=c)
    assert len(c.detail_calls) == 3
