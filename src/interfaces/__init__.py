"""Public re-exports for the interfaces package."""

from .data_models import (
    VehicleAvailability,
    RebalancingResult,
    FeasibilityResult,
    AssignmentResult,
    FeasibleRoute,
    NetworkPath,
    MetaRequest,
    Infeasible,
    Passenger,
    Vehicle,
    Request,
    Route,
    Stop,
)
from .services import (
    IAssignmentSolver,
    INetworkService,
    IRTVGraphBuilder,
    IRVGraphBuilder,
    IRebalancingSolver,
    RTVGraph,
    RVGraph,
)

__all__ = [
    # SERVICES
    "IRebalancingSolver",
    "IAssignmentSolver",
    "IRTVGraphBuilder",
    "INetworkService",
    "IRVGraphBuilder",
    "RTVGraph",
    "RVGraph",
    # DATA MODELS
    "VehicleAvailability",
    "RebalancingResult",
    "FeasibilityResult",
    "AssignmentResult",
    "FeasibleRoute",
    "NetworkPath",
    "MetaRequest",
    "Infeasible",
    "Passenger",
    "Vehicle",
    "Request",
    "Route",
    "Stop",
]
