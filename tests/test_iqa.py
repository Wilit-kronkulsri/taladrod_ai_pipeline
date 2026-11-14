from pathlib import Path

import cv2
import numpy as np
import pytest

from taladrod.config import Settings
from taladrod.iqa import BODY_PROMPTS, VIEW_PROMPTS, ImageScorer, ScoreResult, raw_metrics

GROUPS = list(VIEW_PROMPTS.values()) + list(BODY_PROMPTS.values())  # 5 view axes, then 4 body axes


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class FakeEncoder:
    """9-d space: one axis per prompt group (5 view types, then 4 body types), so tests can predict
    the result without the real CLIP weights. An image 240 px tall is an "exterior" shot, any other
    height an "interior"; wider than 400 px is a "pickup", otherwise a "sedan"."""
    logit_scale = 100.0

    def encode_texts(self, texts):
        out = np.zeros((len(texts), len(GROUPS)), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, next(n for n, g in enumerate(GROUPS) if t in g)] = 1.0
        return out

    def encode_images(self, images):
        rows = []
        for im in images:
            view = [0.9, 0.1, 0.1, 0.1, 0.1] if im.height == 240 else [0.1, 0.9, 0.1, 0.1, 0.1]
            body = [0.1, 0.9, 0.1, 0.1] if im.width > 400 else [0.9, 0.1, 0.1, 0.1]
            rows.append(_unit(view + body))
        return np.stack(rows)


def fake_quality(path) -> float:
    """Stands in for MUSIQ (0-100): more edge energy = higher score."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return float(min(cv2.Laplacian(gray, cv2.CV_64F).var() / 20.0, 100.0))


def make_img(tmp_path: Path, name: str, blur=False, wide=False, interior=False) -> Path:
    rng = np.random.default_rng(0)
    img = np.full((200 if interior else 240, 480 if wide else 320, 3), 128, np.uint8)
    cv2.rectangle(img, (40, 90), (280, 190), (30, 30, 200), -1)          # "car body"
    cv2.circle(img, (90, 190), 25, (20, 20, 20), -1)
    cv2.circle(img, (230, 190), 25, (20, 20, 20), -1)
    img = cv2.add(img, rng.integers(0, 40, img.shape, dtype=np.uint8))  # texture
    if blur:
        img = cv2.GaussianBlur(img, (31, 31), 0)
    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture
def iqa():
    return ImageScorer(Settings(), encoder=FakeEncoder(), quality=fake_quality)


def test_score_is_musiq_over_ten(iqa, tmp_path):
    p = make_img(tmp_path, "a.jpg")
    [r] = iqa.score_paths([p])
    assert isinstance(r, ScoreResult) and r.error is None
    raw = fake_quality(p)
    assert r.breakdown == {"musiq": round(raw, 1)}
    assert r.score == round(raw / 10, 1)
    assert 0.0 <= r.score <= 10.0


def test_blur_scores_lower(iqa, tmp_path):
    sharp, blurry = iqa.score_paths([make_img(tmp_path, "s.jpg"), make_img(tmp_path, "b.jpg", blur=True)])
    assert blurry.score < sharp.score
    assert blurry.raw_metrics["laplacian_var"] < sharp.raw_metrics["laplacian_var"]


def test_view_type_and_confidence(iqa, tmp_path):
    ext, inn = iqa.score_paths([make_img(tmp_path, "e.jpg"), make_img(tmp_path, "i.jpg", interior=True)])
    assert (ext.view_type, inn.view_type) == ("exterior", "interior")
    assert 0.6 < ext.view_confidence <= 1.0
    assert set(VIEW_PROMPTS) == {"exterior", "interior", "engine", "detail", "other"}


def test_body_probs_only_for_exterior_shots(iqa, tmp_path):
    sedan, pickup, inside = iqa.score_paths([make_img(tmp_path, "s.jpg"), make_img(tmp_path, "p.jpg", wide=True),
                                             make_img(tmp_path, "i.jpg", interior=True)])
    assert set(sedan.body_probs) == set(BODY_PROMPTS)
    assert abs(sum(sedan.body_probs.values()) - 1.0) < 1e-6
    assert max(sedan.body_probs, key=sedan.body_probs.get) == "เก๋ง"
    assert max(pickup.body_probs, key=pickup.body_probs.get) == "กระบะ"
    assert inside.body_probs == {}  # a cabin shot says nothing about the body type


def test_corrupt_and_missing_files(iqa, tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image at all")
    good = make_img(tmp_path, "g.jpg")
    res = iqa.score_paths([bad, tmp_path / "missing.jpg", good], batch_size=2)
    for r in res[:2]:
        assert r.score == 0.0 and r.error
        assert r.breakdown == {"musiq": 0.0}
        assert r.view_type is None and r.body_probs == {}
    assert res[2].error is None and res[2].score > 0


def test_quality_model_failure_on_one_image_does_not_stop_the_batch(tmp_path):
    def flaky(path):
        if Path(path).name == "x.jpg":
            raise RuntimeError("boom")
        return 70.0

    iqa = ImageScorer(Settings(), encoder=FakeEncoder(), quality=flaky)
    bad, ok = iqa.score_paths([make_img(tmp_path, "x.jpg"), make_img(tmp_path, "y.jpg")])
    assert bad.score == 0.0 and "boom" in bad.error
    assert ok.score == 7.0 and ok.error is None


def test_batches_keep_order(iqa, tmp_path):
    paths = [make_img(tmp_path, f"{i}.jpg", blur=(i % 2 == 1)) for i in range(5)]
    one_batch = iqa.score_paths(paths, batch_size=16)
    small = iqa.score_paths(paths, batch_size=2)
    assert [r.score for r in one_batch] == [r.score for r in small]


def test_progress_reports_every_batch(iqa, tmp_path):
    seen = []
    iqa.score_paths([make_img(tmp_path, f"{i}.jpg") for i in range(3)], batch_size=2,
                    on_progress=lambda d, t: seen.append((d, t)))
    assert seen == [(2, 3), (3, 3)]


def test_raw_metrics_values(tmp_path):
    p = make_img(tmp_path, "m.jpg")
    m = raw_metrics(cv2.imread(str(p)))
    assert set(m) == {"laplacian_var", "mean_brightness"}
    assert 0 <= m["mean_brightness"] <= 255


# ----------------------------------------------------------- real models (need pyiqa + weights)
@pytest.fixture(scope="session")
def real_iqa():
    pytest.importorskip("pyiqa")
    try:
        return ImageScorer(Settings()).load()
    except OSError as e:  # weights cannot be downloaded (offline / sandbox)
        pytest.skip(f"model weights unavailable: {e}")


@pytest.mark.slow
def test_real_models_blur_lowers_score(real_iqa, tmp_path):
    src = next(iter(sorted(Path("output/images").glob("*.jpg"))), None)
    if src is None:
        src = make_img(tmp_path, "car.jpg")
    img = cv2.imread(str(src))
    cv2.imwrite(str(tmp_path / "o.jpg"), img)
    cv2.imwrite(str(tmp_path / "b.jpg"), cv2.GaussianBlur(img, (31, 31), 0))
    o, b = real_iqa.score_paths([tmp_path / "o.jpg", tmp_path / "b.jpg"])
    assert o.error is None and b.error is None
    assert b.score < o.score
    assert o.view_type in VIEW_PROMPTS
