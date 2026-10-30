"""What the two-stage model believes per zone, against where vehicles sit idle.

    uv run python scripts/analysis/two_stage_coverage.py

Runs the two-stage solver (Thu 2025-01-16, 17:45, 3000 vehicles, w = 0.01, no
fallback rebalancer, 150 batches) and records, for the measured batches: moves
offered and chosen, and per zone the sampled riders per future, how many of
them the solution serves, and idle vehicles. Analysis only: it subclasses the
solver and reads the MILP solution through scipy.
"""
import sys
from datetime import datetime
import numpy as np
import scipy.optimize as so

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import src.experiment as ex
from src.experiment.registry import _two_stage
from src.components.assignment.two_stage_assignment_solver import TwoStageAssignmentSolver

captured = {}
_milp = so.milp
def milp(*a, **k):
    r = _milp(*a, **k); captured["x"] = r.x; return r
so.milp = milp

log = []
class Diag(TwoStageAssignmentSolver):
    def _arcs(self, nodes, times, origin, t_req, deadline, reward):
        a, b, c = super()._arcs(nodes, times, origin, t_req, deadline, reward)
        self._acc.append((len(nodes), a, b))
        return a, b, c
    def _continuous_columns(self, edges, moves, n_unserved):
        self._acc = []
        self._n_first = len(edges) + len(moves) + n_unserved
        self._n_edges, self._moves_offered = len(edges), len(moves)
        return super()._continuous_columns(edges, moves, n_unserved)
    def solve(self, rtv_graph, requests, vehicles, solver_timeout, current_time):
        captured.pop("x", None)
        res = super().solve(rtv_graph, requests, vehicles, solver_timeout, current_time)
        if not self.scenarios or "x" not in captured or captured["x"] is None:
            return res
        x = captured["x"]
        riders = [q for f in self.scenarios for q in f]
        zone = self._zones.zones_of(np.array([q.origin_node for q in riders]))
        # state arcs are the arcs whose source array is the states (len == n_states); in order of futures
        n_states = self._n_edges + self._moves_offered + len(self._vehicles)
        served = np.zeros(len(riders))
        k = self._n_first
        bases = np.cumsum([0] + [len(f) for f in self.scenarios])
        fi = 0
        for (n_src, a, b) in self._acc:
            y = x[k:k + len(a)]; k += len(a)
            if n_src == n_states:
                served_idx = bases[fi] + b[y > 0.5]; served[served_idx] = 1; fi += 1
        idle = [v for v in vehicles if not v.passengers_onboard and not v.remaining_stops_to_serve]
        log.append(dict(t=current_time, zone=zone, served=served, idle_zone=self._zones.zones_of(np.array([v.available_at_node for v in idle], dtype=np.int64)),
                        offered=self._moves_offered, chosen=len(res.rebalancing), S=len(self.scenarios)))
        return res

ex.ASSIGNMENT_SOLVERS["two-stage-diag"] = lambda ctx, obj: _two_stage(Diag, ctx, obj)
cfg = ex.ExperimentConfig(name="diag_zones", fleet=3000, start=datetime(2025, 1, 16, 17, 45), hours=1, max_batches=150,
                          assignment_solver="two-stage-diag", rebalancing="none", scenario_weight=0.01, read_only=True)
ex.run_experiment(cfg)
late = [r for r in log if r["t"] >= log[0]["t"] + 900]
zones = np.unique(np.concatenate([r["zone"] for r in late]))
rows = []
for z in zones:
    n = sum((r["zone"] == z).sum() / r["S"] for r in late) / len(late)
    s = sum(r["served"][r["zone"] == z].sum() / r["S"] for r in late) / len(late)
    idle = sum((r["idle_zone"] == z).sum() for r in late) / len(late)
    rows.append((int(z), n, s, idle))
rows.sort(key=lambda r: -r[1])
print("batches", len(late), "moves offered/batch", np.mean([r["offered"] for r in late]), "chosen/batch", np.mean([r["chosen"] for r in late]))
print("overall share of sampled riders served in the model", sum(r[2] for r in rows) / sum(r[1] for r in rows))
print("zone  sampled/future  served-in-model/future  share  idle vehicles")
for z, n, s, idle in rows[:12] + rows[-8:]:
    print(f"{z:4d} {n:10.1f} {s:10.1f} {s / max(n, 1e-9):6.2f} {idle:8.1f}")
