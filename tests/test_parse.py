from pathlib import Path

import pytest

from taladrod.parse import (
    ListingStub, ParseError, iter_pages, parse_detail_page, parse_list_page,
    parse_mileage, parse_price, parse_year,
)

FX = Path(__file__).parent / "fixtures"
BASE = "https://www.taladrod.com/w40/isch/schc.aspx?fno:new"


def read(name: str) -> str:
    return (FX / name).read_text(encoding="utf-8")


def test_parse_price():
    assert parse_price("459,000 บาท") == 459000
    assert parse_price("649,000") == 649000
    assert parse_price("ติดต่อผู้ขาย") is None
    assert parse_price(None) is None
    assert parse_price("") is None


def test_parse_mileage():
    assert parse_mileage("เลขไมล์ 120,000 กม.") == 120000
    assert parse_mileage("7,100") == 7100
    assert parse_mileage("") is None
    assert parse_mileage(None) is None


def test_parse_year():
    assert parse_year("2561") == 2018
    assert parse_year("2018") == 2018
    assert parse_year(" 2020 ") == 2020
    assert parse_year("abc") is None
    assert parse_year(None) is None


def test_list_page():
    stubs = parse_list_page(read("list_new.html"), BASE)
    assert len(stubs) == 11
    assert stubs[0].ad_id == "2932905"
    assert stubs[2] == ListingStub(
        ad_id="2932903",
        url="https://www.taladrod.com/w40/icar/cardet.aspx?cid=2932903",
        year=2020, price_thb=649000, model="X1",
        name_mmt="BMW X1 2.0 sDrive20d M SPOR F48 ปี16-23",
        headline="X1 20D MSPORT รุ่น TOP สุด รถมือเดียวออก",
    )
    assert len({s.ad_id for s in stubs}) == len(stubs)


def test_list_page_without_data_raises():
    with pytest.raises(ParseError):
        parse_list_page("<html><body>ไม่พบรถคันที่คุณค้นหา</body></html>", BASE)


def test_iter_pages_in_order():
    stubs = parse_list_page(read("list_new.html"), BASE)
    pages = list(iter_pages(stubs, page_size=4))
    assert [len(p) for p in pages] == [4, 4, 3]
    assert [s.ad_id for p in pages for s in p] == [s.ad_id for s in stubs]


def test_detail_page_15_images():
    url = "https://www.taladrod.com/w40/icar/cardet.aspx?cid=2932903"
    ad = parse_detail_page(read("detail_15img.html"), url)
    assert ad.ad_id == "2932903"
    assert (ad.brand, ad.model, ad.year) == ("BMW", "X1", 2020)
    assert ad.full_title == "2020 BMW X1 2.0 sDrive20d M SPORT"
    assert ad.price_thb == 649000
    assert ad.mileage_km == 101000
    assert ad.ad_source_url == url
    assert len(ad.images) == 15 > 10
    assert [im.index for im in ad.images] == list(range(1, 16))
    assert len({im.source_url for im in ad.images}) == 15
    assert ad.images[0].source_url.startswith("https://imgc2.taladrod.com/c/cidx/014/089/74_1.jpg")
    assert ad.images[14].source_url.startswith("https://imgc2.taladrod.com/c/cidx/014/089/74_15.jpg")
    # "similar cars" thumbnails from other ads must not leak in
    assert not any("T3" in im.source_url or "/04_1" in im.source_url for im in ad.images)


def test_detail_page_8_images():
    ad = parse_detail_page(read("detail_8img.html"), "https://www.taladrod.com/w40/icar/cardet.aspx?cid=2932854")
    assert (ad.brand, ad.model, ad.year, ad.price_thb, ad.mileage_km) == ("JAECOO", "5 EV", 2025, 499000, 7100)
    assert len(ad.images) == 8


def test_detail_page_without_mileage():
    ad = parse_detail_page(read("detail_no_mileage.html"), "https://www.taladrod.com/w40/icar/cardet.aspx?cid=2932854")
    assert ad.mileage_km is None
    assert ad.price_thb == 499000


def test_detail_falls_back_to_stub():
    stub = ListingStub(ad_id="9", url="https://www.taladrod.com/w40/icar/cardet.aspx?cid=9", year=2014,
                       price_thb=245000, model="MU-7", name_mmt="ISUZU MU-7 3.0 I-TEQ 4 WD ปี06-13", headline="")
    html = "<html><head><title>x</title></head><body>รหัสอ้างอิง 9</body></html>"
    ad = parse_detail_page(html, stub.url, stub=stub)
    assert (ad.ad_id, ad.brand, ad.model, ad.year, ad.price_thb) == ("9", "ISUZU", "MU-7", 2014, 245000)
    assert ad.images == []
