from src.interfaces import (
    Request,
    Vehicle,
    ISingleVehicleRoutePlanner,
    FeasibilityResult,
)


class RoutePlanner(ISingleVehicleRoutePlanner):

    def check_feasibility(
        self, vehicle: Vehicle, requests_to_add: set[Request]
    ) -> FeasibilityResult:

        raise NotImplementedError()


if __name__ == "__main__":
    # liste med
    pass
