import numpy as np
import cvxpy as cvx

from obstacles import Planet, Satellite

from numpy.typing import NDArray

class Map:

    planets: list[Planet]
    satellites: list[Satellite]

    lowerbound: float
    upperbound: float

    def __init__(self, planets: list[Planet] = [], satellites: list[Satellite] = []):

        self.planets = planets
        self.satellites = satellites

        self.lowerbound = -10.
        self.upperbound = 10.

        self.n_satellites = len(satellites)
        self.n_planets = len(planets)
        self.n_nonconvex_constraints = self.n_satellites + self.n_planets

    def normalize(self, meter_scale):

        self.lowerbound /= meter_scale
        self.upperbound /= meter_scale

        for planet in self.planets:
            planet.normalize(meter_scale)
        for satellite in self.satellites:
            satellite.normalize(meter_scale)     

    def get_linearized_constraints(self, X: cvx.Variable, X_last: cvx.Parameter, p_last: cvx.Parameter, nu_s: cvx.Variable, r: float) -> list[cvx.Constraint]:
        constraints = []
        for i, planet in enumerate(self.planets):
            constraints += planet.get_linearized_constraints(X, X_last, nu_s[i], r)
        for i, satellite in enumerate(self.satellites):
            constraints += satellite.get_linearized_constraints(X, X_last, p_last, nu_s[i+self.n_planets], r)
        return constraints
    
    def get_linear_cost(self, nu_s: cvx.Variable) -> float:
        cost = 0
        for i in range(self.n_nonconvex_constraints):
            cost += np.sum(nu_s[i].value)
        return cost

    def get_nonlinear_cost(self, X: NDArray, U: NDArray, r: float) -> float:
        cost = 0
        for planet in self.planets:
            vector_to_obstacle = X[0:2, :].T - planet.center
            dist_to_obstacle = np.linalg.norm(vector_to_obstacle, 2, axis=1)
            is_violated = dist_to_obstacle < planet.radius + r
            violation = planet.radius + r - dist_to_obstacle
            cost += np.sum(is_violated * violation)

        for satellite in self.satellites:
            vector_to_obstacle = np.array([X[0:2, k].T - satellite.centers[:, k] for k in range(X.shape[1])])
            dist_to_obstacle = np.linalg.norm(vector_to_obstacle, 2, axis=1)
            is_violated = dist_to_obstacle < satellite.radius + r
            violation = satellite.radius + r - dist_to_obstacle
            cost += np.sum(is_violated * violation)

        return cost