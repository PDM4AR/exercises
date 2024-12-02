from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property
from math import cos, sin

import numpy as np
from dg_commons.sim import SimTime, extract_pose_from_state
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipState
from geometry import angle_from_SE2
from shapely import Polygon
from shapely.geometry import Point, LineString
from shapely.ops import unary_union


@dataclass(frozen=True)
class SpaceshipTarget(PlanningGoal):
    target: DynObstacleState
    pos_tol: float
    vel_tol: float
    dir_tol: float

    def is_fulfilled(self, state: SpaceshipState, at: SimTime = Decimal(0)) -> bool:
        return self._is_fulfilled(state, self.target, self.pos_tol, self.vel_tol, self.dir_tol)

    def get_plottable_geometry(self, at: SimTime | float = 0) -> Polygon:
        return self._plottable_geometry

    @cached_property
    def _plottable_geometry(self) -> Polygon:
        # Make sure norm is aligned with is_fulfilled function
        goal_shape = Point(self.target.x, self.target.y).buffer(self.pos_tol)
        # Calculate the endpoint of the line using self.psi and pos_tol
        line_end_x = self.target.x + (self.pos_tol + 0.1) * cos(self.target.psi)
        line_end_y = self.target.y + (self.pos_tol + 0.1) * sin(self.target.psi)

        # Create a line from the center to the calculated endpoint
        line = LineString([(self.target.x, self.target.y), (line_end_x, line_end_y)])
        line_thickness = 0.05  # Adjust the thickness of the line if needed
        line_buffer = line.buffer(line_thickness, cap_style=2)

        return line_buffer

    @staticmethod
    def _is_fulfilled(
        state: SpaceshipState, target: DynObstacleState, pos_tol: float, vel_tol: float, dir_tol: float
    ) -> bool:
        pose = extract_pose_from_state(state)
        is_within_position = np.linalg.norm(np.array([state.x, state.y]) - np.array([target.x, target.y])) < pos_tol
        state_psi = angle_from_SE2(pose)
        is_within_orientation = (
            abs(state_psi - target.psi) < dir_tol or 2 * np.pi - abs(state_psi - target.psi) < dir_tol
        )
        is_within_velocity = np.linalg.norm(np.array([state.vx, state.vy]) - np.array([target.vx, target.vy])) < vel_tol

        return is_within_position and is_within_orientation and is_within_velocity


@dataclass(frozen=True)
class DockingTarget(SpaceshipTarget):
    # This class defines the goal dock station
    # add_land_space together with pos_tol defines the lenght of the landing base
    add_land_space: float
    # length of the arms
    arms_length: float
    # offset of the landing base from the center of the goal
    offset: float

    @cached_property
    def _plottable_geometry(self) -> Polygon:
        # the real offset y must be the distance cog and thruster end
        offset_y = self.offset

        # define the landing base starting and ending points
        sinpsi = sin(self.target.psi)
        cospsi = cos(self.target.psi)

        line_dock_x_start = self.target.x - offset_y * cospsi - (self.pos_tol + self.add_land_space) * sinpsi
        line_dock_y_start = self.target.y - offset_y * sinpsi + (self.pos_tol + self.add_land_space) * cospsi
        line_dock_x_end = self.target.x - offset_y * cospsi + (self.pos_tol + self.add_land_space) * sinpsi
        line_dock_y_end = self.target.y - offset_y * sinpsi - (self.pos_tol + self.add_land_space) * cospsi

        line_dock = LineString([(line_dock_x_start, line_dock_y_start), (line_dock_x_end, line_dock_y_end)])
        line_thickness = 0.05  # Adjust the thickness of the line if needed
        line_dock_buffer = line_dock.buffer(line_thickness, cap_style=2)

        # define the first arm
        line_catch_x_end = line_dock_x_start + (self.pos_tol + self.arms_length) * cospsi
        line_catch_y_end = line_dock_y_start + (self.pos_tol + self.arms_length) * sinpsi
        line_catch = LineString([(line_dock_x_start, line_dock_y_start), (line_catch_x_end, line_catch_y_end)])
        line_thickness = 0.05  # Adjust the thickness of the line if needed
        line_catch_buffer = line_catch.buffer(line_thickness, cap_style=2)
        combined_polygon = unary_union([line_dock_buffer, line_catch_buffer])

        # define the second arm
        line_catch_x_end = line_dock_x_end + (self.pos_tol + self.arms_length) * cospsi
        line_catch_y_end = line_dock_y_end + (self.pos_tol + self.arms_length) * sinpsi
        line_catch = LineString([(line_dock_x_end, line_dock_y_end), (line_catch_x_end, line_catch_y_end)])
        line_thickness = 0.05  # Adjust the thickness of the line if needed
        line_catch_buffer = line_catch.buffer(line_thickness, cap_style=2)

        # Combine the goal shape and the line into a single geometry (using union)
        combined_polygon = unary_union([combined_polygon, line_catch_buffer])
        return combined_polygon

    def get_landing_base(self):
        offset_y = self.offset

        sinpsi = sin(self.target.psi)
        cospsi = cos(self.target.psi)

        line_dock_x_start = self.target.x - offset_y * cospsi - (self.pos_tol + self.add_land_space) * sinpsi
        line_dock_y_start = self.target.y - offset_y * sinpsi + (self.pos_tol + self.add_land_space) * cospsi
        line_dock_x_end = self.target.x - offset_y * cospsi + (self.pos_tol + self.add_land_space) * sinpsi
        line_dock_y_end = self.target.y - offset_y * sinpsi - (self.pos_tol + self.add_land_space) * cospsi

        line_dock = LineString([(line_dock_x_start, line_dock_y_start), (line_dock_x_end, line_dock_y_end)])
        line_thickness = 0.05  # Adjust the thickness of the line if needed
        line_dock_buffer = line_dock.buffer(line_thickness, cap_style=2)
        return line_dock_buffer

    def get_landing_constraint_points(self):
        """
        Returns some useful points to create constraints for the landing scenario.
        In particular:
                -A: a point with offset of 0.1 (closer to the landing base)
                from the pinpoint goal position.
                -B: end of arm 1 + offset.
                -C: end of arm 2 + offset.
                -A1: point aligned with the starting point of the landing base.
                -A2: point aligned with the ending point of landing base.
                -p: (angular aperture of the dock)/2..
        """
        offset_y = 0.3
        sinpsi = sin(self.target.psi)
        cospsi = cos(self.target.psi)

        center_of_landing_x = self.target.x - 0.1 * cospsi
        center_of_landing_y = self.target.y - 0.1 * sinpsi

        line_dock_x_start = self.target.x - offset_y * cospsi - (self.pos_tol - 0.2) * sinpsi
        line_dock_y_start = self.target.y - offset_y * sinpsi + (self.pos_tol - 0.2) * cospsi
        line_dock_x_end = self.target.x - offset_y * cospsi + (self.pos_tol - 0.2) * sinpsi
        line_dock_y_end = self.target.y - offset_y * sinpsi - (self.pos_tol - 0.2) * cospsi

        t1_x = line_dock_x_start + (self.pos_tol + self.arms_length) * cospsi
        t1_y = line_dock_y_start + (self.pos_tol + self.arms_length) * sinpsi

        t2_x = line_dock_x_end + (self.pos_tol + self.arms_length) * cospsi
        t2_y = line_dock_y_end + (self.pos_tol + self.arms_length) * sinpsi

        A = np.array([center_of_landing_x, center_of_landing_y])

        C = np.array([t1_x, t1_y])
        B = np.array([t2_x, t2_y])
        A1 = np.array([line_dock_x_start, line_dock_y_start])
        A2 = np.array([line_dock_x_end, line_dock_y_end])

        return A, B, C, A1, A2, np.arcsin(np.linalg.norm(B - C) / (2 * np.linalg.norm(B - A)))
    
    def get_landing_constraint_points_fix(self):
        """
        Returns some useful points to create constraints for the landing scenario.
        In particular:
                -A: a point with offset of 0.1 (closer to the landing base)
                from the pinpoint goal position.
                -B: end of arm 1.
                -C: end of arm 2.
                -A1: starting point of landing base.
                -A2: ending point of landing base.
                -p: (angular aperture of the dock)/2..
        """
        offset_y = self.offset
        sinpsi = sin(self.target.psi)
        cospsi = cos(self.target.psi)

        center_of_landing_x = self.target.x - 0.1 * cospsi
        center_of_landing_y = self.target.y - 0.1 * sinpsi

        line_dock_x_start = self.target.x - offset_y * cospsi - (self.pos_tol + self.add_land_space) * sinpsi
        line_dock_y_start = self.target.y - offset_y * sinpsi + (self.pos_tol + self.add_land_space) * cospsi
        line_dock_x_end = self.target.x - offset_y * cospsi + (self.pos_tol + self.add_land_space) * sinpsi
        line_dock_y_end = self.target.y - offset_y * sinpsi - (self.pos_tol + self.add_land_space) * cospsi

        t1_x = line_dock_x_start + (self.pos_tol + self.arms_length) * cospsi
        t1_y = line_dock_y_start + (self.pos_tol + self.arms_length) * sinpsi

        t2_x = line_dock_x_end + (self.pos_tol + self.arms_length) * cospsi
        t2_y = line_dock_y_end + (self.pos_tol + self.arms_length) * sinpsi

        A = np.array([center_of_landing_x, center_of_landing_y])

        C = np.array([t1_x, t1_y])
        B = np.array([t2_x, t2_y])
        A1 = np.array([line_dock_x_start, line_dock_y_start])
        A2 = np.array([line_dock_x_end, line_dock_y_end])

        return A, B, C, A1, A2, np.arcsin(np.linalg.norm(B - C) / (2 * np.linalg.norm(B - A)))
