"""Start / pause / reset control for the experiment.

State persists in the meta table so it survives restarts. Reset archives the
entire data set (nothing is ever deleted) and exits the process; systemd
restarts it with a clean log.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

from .config import Settings
from .db import Database

log = logging.getLogger(__name__)

RESET_EXIT_DELAY_SECONDS = 0.7


class Controller:
    def __init__(self, settings: Settings, db: Database,
                 exit_fn: Callable[[], None] | None = None):
        self._settings = settings
        self._db = db
        self._exit_fn = exit_fn or (lambda: os._exit(1))
        self.paused = db.get_meta("paused") == "1"

    def start(self) -> None:
        self.paused = False
        self._db.set_meta("paused", "0")
        if self._db.get_meta("experiment_start") is None:
            self._db.set_meta("experiment_start", str(time.time()))
        log.info("experiment started/resumed")

    def pause(self) -> None:
        self.paused = True
        self._db.set_meta("paused", "1")
        log.info("experiment paused")

    def experiment_start_ts(self) -> float | None:
        raw = self._db.get_meta("experiment_start")
        return float(raw) if raw else None

    def reset(self) -> str:
        """Archive db + csv, seed a fresh paused db, schedule process exit."""
        self.paused = True
        stamp = time.strftime("%Y%m%d-%H%M%S")
        archive = self._settings.db_path.parent / f"archive-{stamp}"
        archive.mkdir(parents=True, exist_ok=True)

        self._db.close()
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(self._settings.db_path) + suffix)
            if p.exists():
                shutil.move(str(p), archive / p.name)
        if self._settings.csv_dir.exists():
            shutil.move(str(self._settings.csv_dir),
                        archive / self._settings.csv_dir.name)

        fresh = Database(self._settings.db_path)
        fresh.set_meta("paused", "1")
        fresh.close()

        log.warning("experiment reset: data archived to %s; process will "
                    "restart and come back paused", archive)
        threading.Timer(RESET_EXIT_DELAY_SECONDS, self._exit_fn).start()
        return str(archive)
