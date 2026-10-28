"""Timing sections nest, reset on take, and the summary accounts for every second."""

import gc
import threading
import time

import pytest

from src.utils.timing import (
    UNTIMED,
    Timer,
    TimingLog,
    summarize,
    time_garbage_collection,
    timed,
    using_timer,
)


def test_sections_nest_and_take_resets():
    timer = Timer()
    with timer.section("batch"):
        with timer.section("rv"):
            time.sleep(0.01)
        with timer.section("rv"):
            pass
        timer.record("manual", 0.5)
    taken = timer.take()
    assert set(taken) == {"batch", "batch/rv", "batch/manual"}
    assert taken["batch/rv"][1] == 2
    assert taken["batch/rv"][0] >= 0.01
    assert taken["batch/manual"] == (0.5, 1)
    assert timer.take() == {}


def test_section_records_on_exception():
    timer = Timer()
    with pytest.raises(ValueError):
        with timer.section("a"):
            raise ValueError
    with timer.section("b"):
        pass
    assert set(timer.take()) == {"a", "b"}  # the stack was unwound


def test_summary_shares_and_untimed_remainder():
    log = TimingLog()
    log.add("setup", 0, {"setup": (1.0, 1)})
    log.add("measured", 1, {"batch": (2.0, 1), "batch/rv": (1.0, 1)})
    log.add("measured", 2, {"batch": (1.0, 1), "batch/rv": (1.0, 1)})
    s = summarize(log.frame())
    rows = {r["section"]: r for r in s.iter_rows(named=True)}
    assert rows["batch"]["share"] == pytest.approx(0.75)
    assert rows[f"batch/{UNTIMED}"]["total_s"] == pytest.approx(1.0)
    assert rows["batch"]["batch_mean_s"] == pytest.approx(1.5)
    assert s["section"].to_list()[:2] == ["batch", "batch/rv"]  # tree order, largest first


def test_garbage_collection_is_recorded_in_the_open_section():
    time_garbage_collection()
    with using_timer(Timer()) as timer, timed("outer"):
        gc.collect()
    assert "outer/gc" in timer.take()


def test_runs_record_in_their_own_timer():
    """Each run's sections go to the timer it installed, also across threads."""
    def run(name, out):
        with using_timer(Timer()) as timer:
            with timed(name):
                time.sleep(0.01)
            out[name] = timer.take()

    out = {}
    threads = [threading.Thread(target=run, args=(n, out)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with using_timer(Timer()) as outer:
        run("c", out)
        with timed("d"):
            pass
    assert {n: set(t) for n, t in out.items()} == {"a": {"a"}, "b": {"b"}, "c": {"c"}}
    assert set(outer.take()) == {"d"}
