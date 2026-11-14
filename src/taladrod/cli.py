"""Command line: ``taladrod scrape | score | all | sanity``."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .archive import (AlreadyRunningError, ArchiveError, doomed_archives, is_complete, run_lock,
                      start_new_run_if_complete)
from .config import Settings
from .http import PoliteClient
from .pipeline import NoCheckpointError, load_dataset, run_phase2
from .scraper import run_phase1


def _max_ads(value: str) -> int:
    n = int(value)
    if not 0 <= n <= 100:
        raise argparse.ArgumentTypeError("must be between 0 and 100 (the brief caps one run at 100 ads)")
    return n


def _bar(desc: str):
    bar = tqdm(desc=desc, unit="")

    def update(done: int, total: int, msg: str) -> None:
        bar.total = total
        bar.n = done
        bar.set_postfix_str(msg[:60])
        bar.refresh()
    return bar, update


def write_sanity_report(settings: Settings, report: Path, n: int = 5) -> Path:
    """Top-n and bottom-n images by score, as Markdown with inline image links, so a person can
    check that the scores match what the photos look like."""
    ds = load_dataset(settings.final_path)
    if ds is None:
        raise SystemExit(f"{settings.final_path} not found: run `taladrod score` first")
    rows = [(im.quality_score, ad, im) for ad in ds.ads for im in ad.images
            if im.local_path and not im.error and im.quality_score is not None]
    rows.sort(key=lambda r: r[0], reverse=True)

    def table(items):
        lines = ["| score | MUSIQ | view | ad | image |", "|---|---|---|---|---|"]
        for score, ad, im in items:
            b = im.quality_breakdown or {}
            rel = Path(im.local_path).as_posix()
            lines.append(f"| **{score}** | {b.get('musiq')} | {im.view_type} "
                         f"| [{ad.full_title}]({ad.ad_source_url}) | ![{Path(rel).name}](../{rel}) |")
        return "\n".join(lines)

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"# IQA sanity check\n\n{len(rows)} scored images.\n\n## Top {n}\n\n{table(rows[:n])}\n\n"
        f"## Bottom {n}\n\n{table(rows[-n:][::-1])}\n", encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="taladrod", description="TaladRod scraper + local MUSIQ / CLIP image pipeline")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_ in [("scrape", "phase 1: collect ads and images"),
                        ("score", "phase 2: categorize, score images, write JSON (offline)"),
                        ("all", "phase 1 + phase 2")]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--output", type=Path, default=None, help="output folder (default: output)")
        if name != "score":
            sp.add_argument("--max-ads", type=_max_ads, default=None, help="0-100 (default 100)")
    sp = sub.add_parser("sanity", help="write a top/bottom-N image report to check IQA scores by eye")
    sp.add_argument("--output", type=Path, default=None)
    sp.add_argument("--report", type=Path, default=Path("docs/iqa_sanity.md"))
    sp.add_argument("-n", type=int, default=5)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env(output_dir=args.output, max_ads=getattr(args, "max_ads", None))

    if args.cmd == "sanity":
        print(f"wrote {write_sanity_report(settings, args.report, args.n)}")
        return 0
    try:
        with run_lock(settings):  # one run per output folder at a time
            _run(args.cmd, settings)
    except (AlreadyRunningError, ArchiveError, NoCheckpointError) as e:
        raise SystemExit(str(e))
    return 0


def _run(cmd: str, settings: Settings) -> None:
    client = PoliteClient(settings) if cmd in ("scrape", "all") and settings.max_ads > 0 else None
    if cmd == "all":
        # `all` on a finished run means "fetch a fresh set": keep the old one in output_archive/.
        # `scrape` / `score` alone never move anything.
        doomed = doomed_archives(settings) if is_complete(settings) else []
        archived = start_new_run_if_complete(settings, client=client)
        if archived:
            print(f"archived previous run -> {archived}")
            for old in doomed:
                print(f"deleted old archive {old.name} (only the newest {settings.keep_archives} are kept)")
    if cmd in ("scrape", "all"):
        bar, update = _bar("scrape")
        ads = run_phase1(settings, client=client, on_progress=update) if client else []
        bar.close()
        print(f"phase 1: {len(ads)} ads in {settings.raw_path}")
    if cmd in ("score", "all"):
        bar, update = _bar("score")
        try:
            ds = run_phase2(settings, on_progress=update)
        finally:
            bar.close()
        print(f"phase 2: {ds.meta['ad_count']} ads, {ds.meta['image_count']} images -> {settings.final_path}")


if __name__ == "__main__":
    sys.exit(main())
