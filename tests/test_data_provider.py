"""Requests built per column equal those built row by row (ids and float times)."""

from datetime import datetime, timedelta

import polars as pl

from src.components.simulation_data_provider import SimulationDataProvider
from src.utils.sim_clock import SimulationClock


def test_requests_from_frame_match_rows():
    start = datetime(2024, 12, 31, 20, 47, 55)
    times = [start + timedelta(seconds=k * 7919.123457, microseconds=k * 37) for k in range(2000)]
    df = pl.DataFrame(
        {"pickup_datetime": times, "orig_node": list(range(2000)), "dest_node": list(range(2000, 4000))}
    ).with_columns(
        pl.col("pickup_datetime").cast(pl.Datetime).dt.replace_time_zone("America/New_York")
    )

    def provider():
        p = object.__new__(SimulationDataProvider)
        p._clock = SimulationClock(start_time=start)
        p._request_counter = 41
        return p

    by_row, by_frame = provider(), provider()
    expected = [by_row._request_from_row(row) for row in df.to_dicts()]
    got = list(by_frame._requests_from_frame(df))
    assert got == expected
    assert [r.request_time for r in got] == [r.request_time for r in expected]  # exact floats
    assert by_frame._request_counter == by_row._request_counter == 2041


def test_requests_from_a_start_keep_the_ids_of_a_full_pass():
    start = datetime(2025, 1, 1)
    times = [start + timedelta(seconds=k * 3.5) for k in range(500)]
    df = pl.DataFrame(
        {"pickup_datetime": times, "orig_node": list(range(500)), "dest_node": list(range(500))}
    ).with_columns(pl.col("pickup_datetime").cast(pl.Datetime).dt.replace_time_zone("America/New_York"))

    def provider():
        p = object.__new__(SimulationDataProvider)
        p._clock = SimulationClock(start_time=start)
        p._request_counter = 0
        return p

    full = list(provider()._requests_from_frame(df))
    skipping = provider()
    later = list(skipping._requests_from_frame(df, start=700.0))
    assert later == [r for r in full if r.request_time >= 700.0]
    assert skipping._request_counter == 500
