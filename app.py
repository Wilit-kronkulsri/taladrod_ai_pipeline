"""Streamlit dashboard: run the pipeline and explore the results.

    streamlit run app.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from taladrod.archive import (AlreadyRunningError, ArchiveError, dir_size_mb, doomed_archives, is_complete,
                              list_archives, run_lock, start_new_run_if_complete)
from taladrod.config import Settings
from taladrod.enrich import avg_quality
from taladrod.http import PoliteClient
from taladrod.iqa import ImageScorer
from taladrod.pipeline import NoCheckpointError, load_dataset, run_phase2
from taladrod.scraper import load_checkpoint, run_phase1

st.set_page_config(page_title="TaladRod AI Pipeline", page_icon="🚗", layout="wide")

CATEGORIES = ["เก๋ง", "กระบะ", "SUV/PPV", "รถตู้/MPV", "อื่นๆ"]


VIEW_LABELS = {"exterior": "ภายนอก", "interior": "ภายใน", "engine": "ห้องเครื่อง",
               "detail": "ชิ้นส่วนระยะใกล้", "other": "เอกสาร/อื่นๆ"}
IMG_COLUMNS = ["ad_id", "ชื่อรถ", "ประเภท", "index", "local_path", "source_url", "score", "musiq", "view",
               "laplacian_var", "mean_brightness", "duplicate_of", "error", "is_cover"]


@st.cache_resource(show_spinner="กำลังโหลดโมเดล MUSIQ และ CLIP (ครั้งแรกจะดาวน์โหลด ~700MB)…")
def get_iqa(model: str, pretrained: str) -> ImageScorer:
    return ImageScorer(Settings(clip_model=model, clip_pretrained=pretrained)).load()


@st.cache_data(show_spinner=False)
def load(path: str, mtime: float):
    ds = load_dataset(Path(path))
    if ds is None:
        return None, pd.DataFrame(), pd.DataFrame()
    ads = pd.DataFrame([{
        "ad_id": a.ad_id, "ชื่อรถ": a.full_title, "ยี่ห้อ": a.brand, "ปี": a.year, "ประเภท": a.category,
        "ที่มาของประเภท": a.category_source, "AI ทายจากรูป": a.category_clip,
        "ความมั่นใจ": a.category_clip_confidence, "ไม่ตรง": bool(a.category_mismatch),
        "ราคา (บาท)": a.price_thb, "เลขไมล์ (กม.)": a.mileage_km,
        "จำนวนรูป": len(a.images),
        "คะแนนเฉลี่ย": avg_quality(a), "รูปที่โหลดไม่ได้": sum(1 for i in a.images if i.error),
        "รูปปก": next((i.source_url for i in a.images if i.local_path == a.cover_image), None),
        "ลิงก์ประกาศ": a.ad_source_url,
    } for a in ds.ads])
    imgs = pd.DataFrame([{
        "ad_id": a.ad_id, "ชื่อรถ": a.full_title, "ประเภท": a.category, "index": i.index,
        "local_path": i.local_path, "source_url": i.source_url, "score": i.quality_score,
        "musiq": (i.quality_breakdown or {}).get("musiq"), "view": VIEW_LABELS.get(i.view_type),
        "laplacian_var": (i.raw_metrics or {}).get("laplacian_var"),
        "mean_brightness": (i.raw_metrics or {}).get("mean_brightness"),
        "duplicate_of": i.is_duplicate_of, "error": i.error, "is_cover": i.local_path == a.cover_image,
    } for a in ds.ads for i in a.images])
    return ds.meta, ads, imgs


# ------------------------------------------------------------------ sidebar: run the pipeline
st.sidebar.title("🚗 TaladRod AI Pipeline")
CURRENT = "ปัจจุบัน"
max_ads = st.sidebar.slider("จำนวนประกาศสูงสุด", 1, 100, 100,
                            help="เลือกตามลำดับที่หน้า 'รถมาใหม่วันนี้' แสดง สูงสุด 100 ต่อรอบ")
current = Settings.from_env(max_ads=max_ads)  # output/ is always the current run; buttons act on it
archives = list_archives(current)
chosen = st.sidebar.selectbox("ชุดข้อมูล", [CURRENT] + [a.name for a in archives],
                              help="ชุดเก่าถูกเก็บอัตโนมัติเมื่อกด Run all บนชุดที่ครบแล้ว")
viewing_archive = chosen != CURRENT
settings = current.model_copy(update={"output_dir": current.archive_dir / chosen}) if viewing_archive else current

c1, c2, c3 = st.sidebar.columns(3)
do_scrape = c1.button("Scrape", disabled=viewing_archive)
do_score = c2.button("Score", disabled=viewing_archive)
do_all = c3.button("Run all", type="primary", disabled=viewing_archive)
st.sidebar.caption("Scrape = ดึงข้อมูล+รูป (รันต่อจากเดิมได้) · Score = จัดหมวด+ให้คะแนนรูป (ไม่ใช้เน็ต)")

if do_scrape or do_score or do_all:
    with st.status("กำลังทำงาน…", expanded=True) as status:
        st.caption("อย่ากดอะไรในหน้านี้ระหว่างรัน เพราะจะหยุดงาน ถ้าหยุดกลางทาง กด Run all อีกครั้งเพื่อทำต่อ")
        bar = st.progress(0.0)
        msg = st.empty()

        def progress(done: int, total: int, text: str) -> None:
            bar.progress(min(done / total, 1.0) if total else 1.0)
            msg.write(text)

        try:
            with run_lock(current):  # one run per output folder at a time
                if do_all:  # on a finished run "Run all" means: keep it, fetch a fresh set
                    # checks the site first: an unreachable site must not cost us the finished run
                    archived = start_new_run_if_complete(current, client=PoliteClient(current))
                    if archived:
                        st.write(f"เก็บชุดเดิมไว้ที่ {archived.as_posix()}")
                if do_scrape or do_all:
                    ads = run_phase1(current, on_progress=progress)
                    st.write(f"Phase 1: มี {len(ads)} ประกาศใน checkpoint")
                if do_score or do_all:
                    ds = run_phase2(current, iqa=get_iqa(current.clip_model, current.clip_pretrained),
                                    on_progress=progress)
                    st.write(f"Phase 2: {ds.meta['ad_count']} ประกาศ, {ds.meta['image_count']} รูป")
            status.update(label="เสร็จแล้ว", state="complete")
        except NoCheckpointError as e:
            st.error(f"ไม่ได้ให้คะแนนใหม่ ผลเดิมยังอยู่ครบ: {e}")
            status.update(label="ไม่ได้รัน", state="error")
        except (ArchiveError, AlreadyRunningError) as e:
            st.error(f"ไม่ได้เริ่มรัน ข้อมูลเดิมอยู่ครบ: {e}")
            status.update(label="ไม่ได้รัน", state="error")
    st.cache_data.clear()
    archives = list_archives(current)

# ------------------------------------------------------------------ sidebar: what the next click will do
n_have = len(load_checkpoint(current.raw_path))
if viewing_archive:
    st.sidebar.caption("กำลังดูชุดเก่า ปุ่มรันถูกปิด เลือก 'ปัจจุบัน' ในช่องชุดข้อมูลเพื่อรัน")
elif is_complete(current):
    # no ad count here: a checkpoint can hold more ads than the cap, the results never do
    st.sidebar.caption(f"ชุดปัจจุบันครบแล้ว กด Run all จะเก็บชุดนี้ไว้ แล้วดึงชุดใหม่ {max_ads} ประกาศ "
                       "(ราว 1 ชั่วโมงสำหรับ 100 ประกาศ)")
    for doomed in doomed_archives(current):
        st.sidebar.warning(f"Run all ครั้งถัดไปจะลบชุด {doomed.name}")
elif n_have:
    st.sidebar.caption(f"ชุดปัจจุบันมี {n_have} จาก {max_ads} ประกาศ กด Run all จะทำต่อจากเดิม")
else:
    st.sidebar.caption("ยังไม่มีข้อมูลชุดปัจจุบัน กด Run all เพื่อเริ่มดึง")
if archives:
    st.sidebar.caption(f"ชุดเก่า {len(archives)} ชุด · {dir_size_mb(current.archive_dir):.0f} MB "
                       f"(เก็บ {current.keep_archives} ชุดล่าสุด)")
free_gb = shutil.disk_usage(Path(current.output_dir).resolve().anchor or ".").free / 1024 ** 3
if free_gb < 2:
    st.sidebar.warning(f"ที่ว่างในดิสก์เหลือ {free_gb:.1f} GB")

final = settings.final_path
if final.exists():
    st.sidebar.download_button("ดาวน์โหลด JSON", final.read_bytes(), "taladrod_data.json", "application/json")
meta, ads_df, img_df = load(str(final), final.stat().st_mtime if final.exists() else 0.0)

if meta is None or ads_df.empty:
    st.info("ยังไม่มีข้อมูล กด **Run all** ที่แถบด้านซ้ายเพื่อเริ่ม (หรือรัน `taladrod all` ใน terminal)"
            + (" ชุดก่อนหน้ายังอยู่ เลือกดูได้จากช่อง **ชุดข้อมูล** ที่แถบด้านซ้าย" if archives and not viewing_archive else ""))
    st.stop()
if img_df.empty:  # ads exist but no images were found/downloaded: keep the column layout
    img_df = pd.DataFrame(columns=IMG_COLUMNS)

tab_overview, tab_explore, tab_ad, tab_retake, tab_about = st.tabs(
    ["📊 ภาพรวม", "🔎 สำรวจประกาศ", "🖼️ รายละเอียดประกาศ", "📸 รูปที่ควรถ่ายใหม่", "ℹ️ วิธีให้คะแนน"])

# ------------------------------------------------------------------ overview
with tab_overview:
    ok = img_df[img_df["error"].isna()]
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k6.metric("ประเภทไม่ตรงกับรูป", int(ads_df["ไม่ตรง"].sum()),
              help="ประกาศที่ CLIP ดูรูปแล้วทายประเภทไม่ตรงกับ dictionary ควรให้คนตรวจ")
    k1.metric("ประกาศ", len(ads_df))
    k2.metric("รูปทั้งหมด", len(img_df))
    k3.metric("คะแนนรูปเฉลี่ย", f"{ok['score'].mean():.1f}/10" if len(ok) else "-")
    k4.metric("รูปซ้ำ", int(img_df["duplicate_of"].notna().sum()))
    k5.metric("รูปโหลดไม่ได้", int(img_df["error"].notna().sum()))

    left, right = st.columns(2)
    with left:
        st.subheader("จำนวนประกาศตามประเภท")
        st.bar_chart(ads_df["ประเภท"].value_counts().reindex(CATEGORIES, fill_value=0))
    with right:
        st.subheader("การกระจายคะแนนรูป")
        bins = pd.cut(ok["score"], bins=[0, 2, 4, 5, 6, 7, 8, 9, 10], include_lowest=True)
        st.bar_chart(bins.value_counts().sort_index().rename(lambda b: str(b)))

    left, right = st.columns(2)
    with left:
        st.subheader("คะแนนเฉลี่ยตามชนิดรูป")
        st.dataframe(ok.groupby("view")["score"].agg(จำนวนรูป="count", คะแนนเฉลี่ย="mean").round(1), width="stretch")
    with right:
        st.subheader("คะแนนเฉลี่ยตามประเภทรถ")
        st.dataframe(ok.groupby("ประเภท")["score"].agg(จำนวนรูป="count", คะแนนเฉลี่ย="mean").round(1), width="stretch")

# ------------------------------------------------------------------ explore
with tab_explore:
    f1, f2, f3, f4 = st.columns([2, 2, 1, 2])
    cats = f1.multiselect("ประเภท", CATEGORIES, default=CATEGORIES)
    prices = ads_df["ราคา (บาท)"].dropna()
    pmin, pmax = (int(prices.min()), int(prices.max())) if len(prices) else (0, 1)
    price = f2.slider("ช่วงราคา", pmin, max(pmax, pmin + 1), (pmin, max(pmax, pmin + 1)), step=10_000)
    min_score = f3.number_input("คะแนนเฉลี่ยขั้นต่ำ", 0.0, 10.0, 0.0, 0.5)
    q = f4.text_input("ค้นหาชื่อรถ")
    only_mismatch = st.checkbox("เฉพาะที่ประเภทไม่ตรงกับรูป")
    view = ads_df[
        ads_df["ประเภท"].isin(cats)
        & (ads_df["ไม่ตรง"] | (not only_mismatch))
        & (ads_df["ราคา (บาท)"].isna() | ads_df["ราคา (บาท)"].between(*price))
        & (ads_df["คะแนนเฉลี่ย"].fillna(0) >= min_score)
        & ads_df["ชื่อรถ"].str.contains(q, case=False, regex=False)
    ]
    st.caption(f"{len(view)} จาก {len(ads_df)} ประกาศ")
    st.dataframe(view, hide_index=True, width="stretch", height=560, column_config={
        "รูปปก": st.column_config.ImageColumn("รูปปก", width="small"),
        "ลิงก์ประกาศ": st.column_config.LinkColumn("ลิงก์ประกาศ", display_text="เปิด"),
        "ราคา (บาท)": st.column_config.NumberColumn(format="%d"),
        "เลขไมล์ (กม.)": st.column_config.NumberColumn(format="%d"),
        "คะแนนเฉลี่ย": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f"),
    })


def show_gallery(rows: pd.DataFrame, cols: int = 4) -> None:
    grid = st.columns(cols)
    for n, (_, r) in enumerate(rows.iterrows()):
        with grid[n % cols]:
            if r["local_path"] and Path(r["local_path"]).exists():
                st.image(r["local_path"], width="stretch")
            else:
                st.warning("ไม่มีไฟล์รูป")
            badges = []
            if r["is_cover"]:
                badges.append("⭐ รูปปก")
            if isinstance(r["duplicate_of"], str):
                badges.append("♻️ ซ้ำ")
            if isinstance(r["error"], str):
                badges.append("⚠️ " + r["error"][:40])
            st.markdown(f"**{r['score']:.1f}/10** {' '.join(badges)}")
            if pd.notna(r["musiq"]):
                st.caption(f"{r['view'] or 'ไม่ทราบชนิดรูป'} · MUSIQ {r['musiq']}"
                           f" · Laplacian {r['laplacian_var']} · ความสว่าง {r['mean_brightness']}")
            st.caption(f"[รูปต้นฉบับ]({r['source_url']})")


# ------------------------------------------------------------------ one ad
with tab_ad:
    labels = {f"{r['ชื่อรถ']} · {r['ad_id']}": r["ad_id"] for _, r in ads_df.iterrows()}
    choice = st.selectbox("เลือกประกาศ", list(labels))
    if choice:
        a = ads_df[ads_df["ad_id"] == labels[choice]].iloc[0]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ประเภท", a["ประเภท"])
        m2.metric("ราคา", f"{a['ราคา (บาท)']:,.0f} ฿" if pd.notna(a["ราคา (บาท)"]) else "-")
        m3.metric("เลขไมล์", f"{a['เลขไมล์ (กม.)']:,.0f} กม." if pd.notna(a["เลขไมล์ (กม.)"]) else "ไม่ระบุ")
        m4.metric("ปี", int(a["ปี"]) if pd.notna(a["ปี"]) else "-")
        m5.metric("คะแนนเฉลี่ย", a["คะแนนเฉลี่ย"] if pd.notna(a["คะแนนเฉลี่ย"]) else "-")
        clip_note = ""
        if pd.notna(a["AI ทายจากรูป"]):
            clip_note = (f" · AI ทายจากรูป: {a['AI ทายจากรูป']} ({a['ความมั่นใจ']:.2f})"
                         + (" ⚠️ ไม่ตรงกับประเภท" if a["ไม่ตรง"] else ""))
        st.markdown(f"[เปิดประกาศต้นทาง]({a['ลิงก์ประกาศ']}) · ประเภทมาจาก: `{a['ที่มาของประเภท']}`{clip_note}")
        show_gallery(img_df[img_df["ad_id"] == a["ad_id"]].sort_values("score", ascending=False))

# ------------------------------------------------------------------ retake list
with tab_retake:
    st.write("รูปคะแนนต่ำสุดจากทุกประกาศ ใช้แจ้งผู้ขายให้ถ่ายใหม่ หรือใช้คัดรูปก่อนขึ้นหน้าเว็บ")
    threshold = st.slider("แสดงรูปที่คะแนนต่ำกว่า", 0.0, 10.0, 6.5, 0.5)
    low = img_df[(img_df["score"] < threshold) & img_df["error"].isna()].sort_values("score")
    st.caption(f"{len(low)} รูป")
    show_gallery(low.head(40))

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(f"""
**วิธีเลือกประกาศ:** {meta.get('selection_strategy')}

**วิธีให้คะแนนรูป:** {meta.get('iqa_method')}  ·  รันบนเครื่องทั้งหมด ไม่เรียก API ภายนอก
ค่า Laplacian และความสว่างใต้รูปเป็นค่าอ้างอิงจาก OpenCV ไม่ได้นำมาคิดคะแนน

**ชนิดรูป:** {meta.get('view_method')}  ·  โมเดล `{meta.get('clip_model')}`
""")
    if meta.get("view_prompts"):
        st.table(pd.DataFrame({"prompt": {VIEW_LABELS.get(k, k): " / ".join(v) for k, v in meta["view_prompts"].items()}}))
    if meta.get("category_clip_prompts"):
        st.markdown("**ตรวจประเภทรถจากรูป:** ใช้เฉพาะรูปภายนอกรถ เทียบกับข้อความด้านล่าง แล้วเฉลี่ยต่อประกาศ "
                    "ผลเป็นธงให้คนตรวจเท่านั้น ไม่เขียนทับประเภทจาก dictionary")
        st.table(pd.DataFrame({"prompt": {k: " / ".join(v) if isinstance(v, list) else v
                                          for k, v in meta["category_clip_prompts"].items()}}))
    st.caption(f"สร้างเมื่อ {meta.get('generated_at')} · ไฟล์ {final}")
