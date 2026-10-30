"""Build the simulator from an ExperimentConfig and run it."""

import cProfile
import gc
import io
import os
import pstats
from dataclasses import asdict, replace
from pathlib import Path


import config.config as config
from config import get_logger
from config.paths import DATA_DIR
from src.components.network.network_service import NetworkXService
from src.components.simulation.simulation_data_provider import SimulationDataProvider
from src.database.simple_db import DuckDBPathCache
from src.interfaces.stats import BatchStatistics
from src.simulator import Simulator
from src.utils.generate_plots import generate_plots
from src.utils.output_manager import SimulationOutputManager
from src.utils.stats_logger import SimulationStatsLogger
from src.experiment.config import ExperimentConfig
from src.experiment.registry import (
    ASSIGNMENT_SOLVERS,
    CONTROLLERS,
    DEMAND_MODELS,
    FARE_MODELS,
    FLEET_REPLAYS,
    OBJECTIVES,
    REBALANCERS,
    REQUEST_SAMPLERS,
    RTV_BUILDERS,
    RV_BUILDERS,
    TRIP_SOLVERS,
    VEHICLE_VALUES,
    ZONINGS,
    Context,
)
from src.utils.timing import (
    Timer,
    TimingLog,
    format_summary,
    summarize,
    time_garbage_collection,
    timed,
    using_timer,
)

logger = get_logger(__name__)




def simulation_window(cfg: ExperimentConfig, provider: SimulationDataProvider) -> tuple[float, float]:
    """Start and end in simulation seconds."""
    start = config.MIN_START_TIME
    if cfg.start is not None:
        start = (cfg.start - provider.first_request_time()).total_seconds()
    if cfg.hours is None:
        return start, start + config.END_TIME - config.MIN_START_TIME
    return start, start + config.WARMUP_DURATION + cfg.hours * 3600


def build_simulator(ctx: Context, db_cache: DuckDBPathCache) -> Simulator:
    cfg = ctx.config
    objective = OBJECTIVES[cfg.objective](ctx)
    trip_solver = TRIP_SOLVERS[cfg.trip_solver](ctx)
    return Simulator(
        provider=ctx.provider,
        controller=CONTROLLERS[cfg.controller](ctx),
        rv_builder=RV_BUILDERS[cfg.rv_builder](ctx, trip_solver),
        rtv_builder=RTV_BUILDERS[cfg.rtv_builder](trip_solver),
        assignment_solver=ASSIGNMENT_SOLVERS[cfg.assignment_solver](ctx, objective),
        rebalancing_solver=REBALANCERS[cfg.rebalancing](ctx),
        network_service=ctx.network,
        db_cache=db_cache,
        read_only=cfg.read_only,
    )


def run_experiment(cfg: ExperimentConfig) -> Path:
    """Run one simulation; returns the run folder with stats, plots and metadata."""
    for key, registry in [
        ("objective", OBJECTIVES),
        ("fare_model", FARE_MODELS),
        ("trip_solver", TRIP_SOLVERS),
        ("rtv_builder", RTV_BUILDERS),
        ("assignment_solver", ASSIGNMENT_SOLVERS),
        ("rebalancing", REBALANCERS),
        ("controller", CONTROLLERS),
        ("fleet_replay", FLEET_REPLAYS),
        ("zoning", ZONINGS),
        ("demand_model", DEMAND_MODELS),
        ("rv_builder", RV_BUILDERS),
        ("vehicle_value", VEHICLE_VALUES),
        ("request_sampler", REQUEST_SAMPLERS),
    ]:
        if getattr(cfg, key) not in registry:
            raise ValueError(f"Unknown {key} {getattr(cfg, key)!r}; choose from {list(registry)}")

    # Requests, routes and graphs are short-lived, but with the default
    # thresholds the collector ran ~60 full passes over the whole heap per 150
    # batches (20% of the run). Collection does not change results.
    gc.set_threshold(50_000, 20, 20)
    time_garbage_collection()
    with using_timer(Timer()) as timer:  # this run's sections only
        return _run(cfg, timer)


def _run(cfg: ExperimentConfig, timer: Timer) -> Path:
    with timed("setup"):
        with timed("data"):  # graph and fleet; requests are read lazily
            provider = SimulationDataProvider()
        start_time, end_time = simulation_window(cfg, provider)

        output = SimulationOutputManager(base_dir="reports", run_name=cfg.name)
        run_dir = Path(output.get_run_dir())
        metadata = {k: getattr(config, k) for k in dir(config) if k.isupper()}
        metadata |= {"experiment": asdict(replace(cfg, start=str(cfg.start))), "start_time": start_time,
                     "end_time": end_time}
        output.save_metadata(metadata)
        csv_path = str(run_dir / "simulation_stats.csv")

        logger.info("Initializing components...")
        with timed("network"):  # all-pairs travel times into memory
            db_cache = DuckDBPathCache(read_only=cfg.read_only)
            network = NetworkXService(
                provider.graph,
                db=db_cache,
                travel_time_factor=cfg.travel_time_factor,
                tree_cache_dir=DATA_DIR / "cache",
            )
        with timed("components"):
            simulator = build_simulator(
                Context(cfg, provider, network, start_time, end_time), db_cache
            )

    timing = TimingLog()
    timing.add("setup", 0, timer.take())
    profiler = cProfile.Profile() if cfg.profile else None
    stats_logger = SimulationStatsLogger()
    try:
        if profiler:
            profiler.enable()
        for stats in simulator.run(
            request_iterator=provider.iter_requests(start=start_time),
            batch_duration=config.BATCH_DURATION,
            min_start_time=start_time,
            warmup_duration=config.WARMUP_DURATION,
            max_batches=cfg.max_batches,
            end_time=end_time,
        ):
            if not isinstance(stats, BatchStatistics):
                continue
            with timed("stats"):
                if stats.is_warmup:
                    logger.info(f"🔥 Warm-up batch {stats.batch_id}. Stats NOT logged.")
                else:
                    stats_logger.log_batch(stats)
                    if stats.batch_id % 20 == 0:
                        stats_logger.flush_to_csv(csv_path)
            # Logging this batch's stats is counted with the next batch.
            timing.add("warmup" if stats.is_warmup else "measured", stats.batch_id, stats.timings)
        logger.info("🏁 Simulation complete.")
    except KeyboardInterrupt:
        logger.info("🛑 Simulation stopped by user (Ctrl+C).")
    finally:
        if profiler:
            profiler.disable()
            _save_profile(profiler, run_dir)
        with timed("end"):
            stats_logger.flush_to_csv(csv_path)
            stats_logger.print_summary()

    with timed("end"):
        if os.path.exists(csv_path):
            logger.info("📊 Generating plots...")
            with timed("plots"):
                generate_plots(csv_path)
        else:
            logger.warning("No stats CSV found, skipping plot generation.")
    timing.add("end", 0, timer.take())
    _save_timing(timing, run_dir)
    return run_dir


def _save_timing(timing: TimingLog, run_dir: Path) -> None:
    frame = timing.frame()
    frame.write_csv(run_dir / "timing.csv")
    logger.info("⏱️ Where the time went (timing.csv):\n%s", format_summary(summarize(frame)))


def _save_profile(profiler: cProfile.Profile, run_dir: Path) -> None:
    profiler.dump_stats(run_dir / "profile.prof")
    out = io.StringIO()
    pstats.Stats(profiler, stream=out).sort_stats("tottime").print_stats(30)
    logger.info("cProfile, top 30 by own time (profile.prof):\n%s", out.getvalue())
