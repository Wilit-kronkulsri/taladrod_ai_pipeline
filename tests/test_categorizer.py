from datetime import datetime, timezone

import pytest

from taladrod.categorizer import categorize
from taladrod.models import AdRecord


def ad(title, model=None, body=None, headline=None):
    return AdRecord(ad_id="1", brand=None, model=model, year=None, full_title=title, price_thb=None,
                    mileage_km=None, ad_source_url="u", body_type_raw=body, seller_headline=headline,
                    scraped_at=datetime.now(timezone.utc))


@pytest.mark.parametrize("title,expected", [
    ("2018 TOYOTA HILUX REVO 2.4 E", ("กระบะ", "dictionary")),
    ("2021 MITSUBISHI TRITON 18-23 2.4 GLS", ("กระบะ", "dictionary")),
    ("2020 ISUZU D-MAX 1.9 SPARK", ("กระบะ", "dictionary")),
    ("2023 TOYOTA COROLLA CROSS 1.8 Hybrid", ("SUV/PPV", "dictionary")),
    ("2019 TOYOTA COROLLA ALTIS 1.6 G", ("เก๋ง", "dictionary")),
    ("2014 TOYOTA YARIS 1.2 G", ("เก๋ง", "dictionary")),
    ("2011 ISUZU MU-7 3.0 I-TEQ", ("SUV/PPV", "dictionary")),
    ("2018 TOYOTA ALPHARD 2.5 SC PACKAGE", ("รถตู้/MPV", "dictionary")),
    ("2024 HONDA CRV 1.5 TURBO", ("SUV/PPV", "dictionary")),
    ("2020 BMW X1 2.0 sDrive20d M SPORT", ("SUV/PPV", "dictionary")),
    ("2018 BMW SERIES 4 430i COUPE", ("เก๋ง", "dictionary")),
    ("2025 TESLA MODEL Y LONG RANGE", ("SUV/PPV", "dictionary")),
    ("2020 FOO BAR 1.0", ("อื่นๆ", "fallback")),
])
def test_dictionary(title, expected):
    # the detail parser puts the word after the brand into `model`
    assert categorize(ad(title, model=title.split()[2])) == expected


def test_site_body_type_wins():
    assert categorize(ad("2018 TOYOTA ALTIS", body="รถตู้")) == ("รถตู้/MPV", "site")


def test_case_insensitive_and_word_boundaries():
    # "CITY" must not match inside another word, "Hilux" in any case
    assert categorize(ad("2015 honda city 1.5 v")) == ("เก๋ง", "dictionary")
    assert categorize(ad("2015 Toyota Hilux Vigo")) == ("กระบะ", "dictionary")
    assert categorize(ad("2015 FOO VELOCITY")) == ("อื่นๆ", "fallback")


def test_pickup_hints_in_headline():
    # Unknown model but seller says it's a 4-door pickup ("กระบะ" / "CAB")
    assert categorize(ad("2019 FOTON TUNLAND 2.8", headline="กระบะ 4 ประตู ขับดี")) == ("กระบะ", "dictionary")


def test_short_model_codes_only_match_model_field():
    # 2-letter codes like GS / ZS / LM are trims elsewhere; only trust them as the model itself
    assert categorize(ad("2020 FOO BAR 1.5 GS", model="BAR")) == ("อื่นๆ", "fallback")
    assert categorize(ad("2022 MG ZS 1.5 X", model="ZS")) == ("SUV/PPV", "dictionary")
