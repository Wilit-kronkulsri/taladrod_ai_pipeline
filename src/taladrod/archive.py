"""Starting a new run without losing the previous one.

``output/`` is always the *current* run. When it is complete and the user asks for a fresh one,
the whole folder is renamed to ``output_archive/<YYYY-MM-DD_HHMM>/`` and scraping starts from an
empty ``output/``. Only the newest ``Settings.keep_archives`` archives are kept, so disk use stays
bounded without anyone tidying up by hand. A lock file stops two runs from writing to the same
folder at once (that once produced a 110-ad checkpoint).
"""
from __future__ import annotations

import logging
import re
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .config import Settings
from .scraper import load_checkpoint

log = logging.getLogger(__name__)

# only folders this module created count as archives: anything else the user keeps in
# output_archive/ is never listed, counted against the limit, or deleted
ARCHIVE_NAME = re.compile(r"\d{4}-\d{2}-\d{2}_\d{4}(_\d+)?")


class ArchiveError(RuntimeError):
    """The current run could not be moved aside; nothing was changed."""


class AlreadyRunningError(RuntimeError):
    """Another run holds the lock for this output folder."""


def is_complete(settings: Settings) -> bool:
    """True when the current run already holds all the ads it was asked for, so "run all" means
    "fetch a fresh set". A run that stopped early (WAF, crash) is not complete: it gets resumed.

    shortcut: counts ads only, so a day when the site lists fewer than max_ads never looks
    complete and keeps resuming; add a "listing exhausted" marker if that ever happens.
    """
    return settings.max_ads > 0 and len(load_checkpoint(settings.raw_path)) >= settings.max_ads


def list_archives(settings: Settings) -> list[Path]:
    """Archived runs, newest first (folder names sort by time)."""
    root = settings.archive_dir
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and ARCHIVE_NAME.fullmatch(p.name)),
                  key=lambda p: p.name, reverse=True)


def doomed_archives(settings: Settings) -> list[Path]:
    """Archives that the *next* ``archive_current`` call will delete, so the UI can say so first."""
    return list_archives(settings)[max(settings.keep_archives - 1, 0):]


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_048_576 if path.is_dir() else 0.0


def archive_current(settings: Settings, now: Optional[datetime] = None) -> Path:
    """Move ``output/`` to ``output_archive/<timestamp>/`` and prune old archives. Returns the new
    archive folder. Raises ``ArchiveError`` (and changes nothing) if the folder cannot be moved."""
    src = settings.output_dir
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M")
    dest = settings.archive_dir / stamp
    n = 2
    while dest.exists():  # two runs archived within the same minute
        dest = settings.archive_dir / f"{stamp}_{n}"
        n += 1
    old_images = settings.images_dir.as_posix()
    settings.archive_dir.mkdir(parents=True, exist_ok=True)
    try:
        # one rename inside the same parent folder: it either happens completely or not at all
        src.rename(dest)
    except OSError as e:
        raise ArchiveError(
            f"could not move {src} to {dest}: {e}. Close any program that has files from {src} "
            "open (image viewer, Explorer preview, another run), then try again. Nothing was changed.") from e

    # the records store image paths relative to the old location; point them at the new one so
    # the archived run still opens with its pictures
    new_images = (dest / settings.images_dir.name).as_posix()
    for name in (settings.final_path.name, settings.raw_path.name):
        f = dest / name
        if f.exists():
            # bytes, not text: keeps the files' line endings exactly as they were written
            f.write_bytes(f.read_bytes().replace(f'"{old_images}/'.encode(), f'"{new_images}/'.encode()))

    for old in list_archives(settings)[settings.keep_archives:]:
        log.info("deleting old archive %s", old)
        shutil.rmtree(old, ignore_errors=True)
    return dest


def start_new_run_if_complete(settings: Settings, client=None) -> Optional[Path]:
    """Archive the current run if it is complete. Returns the archive path, or None when the
    current run should simply be resumed.

    With a ``client`` the listing page is fetched first: if the site cannot be reached (offline,
    WAF block) nothing is moved, so a finished run is never swapped for an empty folder.
    ``client=None`` skips that check."""
    if not is_complete(settings):
        return None
    if client is not None:
        try:
            client.get_html(settings.start_url)
        except Exception as e:
            raise ArchiveError(f"the site cannot be reached right now ({e}), so the current run was kept "
                               "as it is. Try again later.") from e
    return archive_current(settings)


@contextmanager
def run_lock(settings: Settings) -> Iterator[None]:
    """Hold ``output.running`` for the duration of a run."""
    lock = settings.lock_path
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock.open("x", encoding="utf-8") as f:
            f.write(f"started {datetime.now().isoformat(timespec='seconds')}\n")
    except FileExistsError:
        raise AlreadyRunningError(
            f"another run is already using {settings.output_dir} (lock file {lock}). Wait for it to "
            "finish. If no run is in progress (e.g. the PC was switched off mid-run), delete this file "
            "and try again.") from None
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)
