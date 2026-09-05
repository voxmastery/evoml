import threading

from memescalp.control import Controller
from memescalp.db import Database
from tests.test_db import fill


def test_start_pause_state_persists(settings):
    db = Database(settings.db_path)
    c = Controller(settings, db, exit_fn=lambda: None)
    assert not c.paused

    c.pause()
    assert c.paused and db.get_meta("paused") == "1"
    # A new controller (fresh process) sees the persisted state.
    assert Controller(settings, db, exit_fn=lambda: None).paused

    c.start()
    assert not c.paused
    assert db.get_meta("experiment_start") is not None


def test_start_does_not_overwrite_experiment_start(settings):
    db = Database(settings.db_path)
    db.set_meta("experiment_start", "12345.0")
    c = Controller(settings, db, exit_fn=lambda: None)
    c.start()
    assert db.get_meta("experiment_start") == "12345.0"


def test_reset_archives_and_reseeds(settings, tmp_path):
    db = Database(settings.db_path)
    db.insert_fill(fill())
    settings.csv_dir.mkdir(parents=True, exist_ok=True)
    (settings.csv_dir / "fills.csv").write_text("ts\n1\n")

    exited = threading.Event()
    c = Controller(settings, db, exit_fn=exited.set)
    archive = c.reset()

    # Old data preserved in the archive, nothing deleted.
    archived = list(tmp_path.glob("archive-*/"))
    assert len(archived) == 1 and str(archived[0]).rstrip("/") == archive
    assert (archived[0] / "test.db").exists()
    assert (archived[0] / "csv" / "fills.csv").exists()

    # Fresh database is empty and comes up paused.
    fresh = Database(settings.db_path)
    assert fresh.fills() == []
    assert fresh.get_meta("paused") == "1"
    assert fresh.get_meta("experiment_start") is None
    fresh.close()

    assert exited.wait(3)
