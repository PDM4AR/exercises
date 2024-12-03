from dataclasses import dataclass, field
from typing import List, Set
import json
from dataclasses import asdict
import numpy as np
from dg_commons import PlayerName
from dg_commons.maps import DgLanelet, DgLanePose
from dg_commons.sim import extract_pose_from_state
from dg_commons.sim.simulator import SimContext
from dg_commons.eval.safety import has_collision, get_min_ttc_max_drac
from dg_commons.eval.efficiency import time_goal_lane_reached
from dg_commons.eval.comfort import get_acc_rms
from dg_commons.seq.sequence import Timestamp

from pdm4ar.exercises_def import PerformanceResults


@dataclass(frozen=True)
class PlayerMetrics(PerformanceResults):
    task_level: int
    """How difficult is the task?(1:simple, 2:moderate, 3: challenging)"""
    score: float = field(init=False)
    """Score of this task"""
    collided: bool
    """Has the player collided?"""
    goal_reached: bool
    """Has the player reached the goal?"""
    min_ttc: float
    """The minimum time-to-collision throughout the simulation"""
    lane_changing_time: Timestamp | None
    """The time it took to reach the goal lane."""
    avg_relative_heading: float
    """The average relative heading of the vehicle wrt the lane it is driving on"""
    max_velocity: float
    """The maximum velocity of the vehicle"""
    min_velocity: float
    """The minimum velocity of the vehicle"""
    discomfort: float
    """The frequency-weighted acceleration of the vehicle according to ISO 2631-1"""
    avg_computation_time: float
    """Average computation time of the get_commands method."""

    _lc_time_penalty: float = field(init=False)
    _ttc_penalty: float = field(init=False)
    _solving_time_penalty: float = field(init=False)
    _discomfort_penalty: float = field(init=False)
    _heading_penalty: float = field(init=False)
    _velocity_penalty: float = field(init=False)

    def __post_init__(self):
        """compute and store the score"""
        max_score = 100
        if self.collided:
            max_score *= 0.2
        if not self.goal_reached:
            max_score *= 0.7
        score = max_score
        # lose points if the lane changing takes longer than 5 seconds
        if self.lane_changing_time is not None:
            lc_time_penalty = (float(self.lane_changing_time) - 5.0) / 5.0
            lc_time_penalty = np.clip(lc_time_penalty, 0.0, 1.0)
            score -= 10.0 * lc_time_penalty
        else:
            lc_time_penalty = 1.0
            score -= 15.0
        # lose points if ttc of the planned trajectory is less than 0.5s
        ttc_penalty = (-self.min_ttc + 0.5) * 2.0
        ttc_penalty = np.clip(ttc_penalty, 0.0, 1.0)
        score -= 15.0 * ttc_penalty
        # lose points if the computation takes longer than 0.5s
        solving_time_penalty = (self.avg_computation_time - 0.5) * 2.0
        solving_time_penalty = np.clip(solving_time_penalty, 0.0, 1.0)
        score -= 5.0 * solving_time_penalty
        # lose points if the discomfort level is larger than 0.6
        discomfort_penalty = (self.discomfort - 0.6) * 3.0
        discomfort_penalty = np.clip(discomfort_penalty, 0.0, 1.0)
        score -= 5.0 * discomfort_penalty
        # lose points if the vehicle is not compliant with the lane
        heading_penalty = (np.abs(self.avg_relative_heading) - 0.1) * 10.0
        heading_penalty = np.clip(heading_penalty, 0.0, 1.0)
        score -= 5.0 * heading_penalty
        # lose points if the vehicle is too fast or too slow
        v_diff = np.maximum(self.max_velocity - 25.0, 5.0 - self.min_velocity)
        velocity_penalty = v_diff / 5.0
        velocity_penalty = np.clip(velocity_penalty, 0.0, 1.0)
        score -= 5.0 * velocity_penalty

        object.__setattr__(self, "_lc_time_penalty", lc_time_penalty)
        object.__setattr__(self, "_ttc_penalty", ttc_penalty)
        object.__setattr__(self, "_solving_time_penalty", solving_time_penalty)
        object.__setattr__(self, "_discomfort_penalty", discomfort_penalty)
        object.__setattr__(self, "_heading_penalty", heading_penalty)
        object.__setattr__(self, "_velocity_penalty", velocity_penalty)
        object.__setattr__(self, "score", np.maximum(0.0, score))

    def __str__(self):
        return json.dumps(asdict(self))

    def reduce_to_score(self) -> float:
        return self.score


@dataclass(frozen=True)
class HighwayTaskPerformance(PerformanceResults):
    avg_score: float = 0.0
    collision_rate: float = 0.0
    success_rate: float = 0.0
    avg_lc_time_penalty: float = 0.0
    avg_risk_penalty: float = 0.0
    avg_heading_penalty: float = 0.0
    avg_velocity_penalty: float = 0.0


def get_task_performance(metrics_all: list[PlayerMetrics]) -> HighwayTaskPerformance:
    avg_score = 0.0
    collision_rate = -0.0
    success_rate = 0.0
    avg_lc_time_penalty = 0.0
    avg_risk_penalty = 0.0
    avg_heading_penalty = 0.0
    avg_velocity_penalty = 0.0

    for metrics in metrics_all:
        avg_score += metrics.score
        collision_rate += 1 if metrics.collided else 0
        success_rate += 1 if metrics.goal_reached else 0
        avg_lc_time_penalty += metrics._lc_time_penalty
        avg_risk_penalty += metrics._ttc_penalty
        avg_heading_penalty += metrics._heading_penalty
        avg_velocity_penalty += metrics._velocity_penalty
    num_scenarios = len(metrics_all) if len(metrics_all) > 0 else 1
    return HighwayTaskPerformance(
        avg_score=avg_score / num_scenarios,
        collision_rate=collision_rate / num_scenarios,
        success_rate=success_rate / num_scenarios,
        avg_lc_time_penalty=avg_lc_time_penalty / num_scenarios,
        avg_risk_penalty=avg_risk_penalty / num_scenarios,
        avg_heading_penalty=avg_heading_penalty / num_scenarios,
        avg_velocity_penalty=avg_velocity_penalty / num_scenarios,
    )


@dataclass(frozen=True)
class HighwayFinalPerformance(PerformanceResults):
    final_score: float
    task_performances: dict[int, HighwayTaskPerformance]


def ex12_metrics(sim_context: SimContext) -> PlayerMetrics:
    lanelet_network = sim_context.dg_scenario.lanelet_network
    collision_reports = sim_context.collision_reports
    ego_name = PlayerName("Ego")
    ego_goal_lane = sim_context.missions[ego_name]
    ego_log = sim_context.log[ego_name]
    ego_states = ego_log.states
    ego_commands = ego_log.commands

    # collision
    collided = has_collision(collision_reports)

    # efficiency metrics
    time_to_reach = time_goal_lane_reached(lanelet_network, ego_goal_lane, ego_states, pos_tol=0.8, heading_tol=0.08)
    if time_to_reach is None:
        has_reached_the_goal = False
    else:
        has_reached_the_goal = True

    # safety metrics
    if has_reached_the_goal:
        min_ttc, _, _, _, _, _ = get_min_ttc_max_drac(
            sim_context.log, sim_context.models, sim_context.missions, ego_name, (0.0, time_to_reach)
        )
    else:
        min_ttc, _, _, _, _, _ = get_min_ttc_max_drac(
            sim_context.log, sim_context.models, sim_context.missions, ego_name
        )

    # comfort matrics
    discomfort = get_acc_rms(ego_commands)

    # computation time
    avg_comp_time = np.average(ego_log.info.values)

    # lane orientation
    avg_heading = 0.0
    max_velocity = -np.inf
    min_velocity = np.inf
    for state in ego_log.states.values:
        lanelet_ids = lanelet_network.find_lanelet_by_position(
            [
                [state.x, state.y],
            ]
        )[0]
        if len(lanelet_ids) == 0:
            break
        else:
            lanelet = lanelet_network.find_lanelet_by_id(lanelet_ids[0])
            dg_lanelet = DgLanelet.from_commonroad_lanelet(lanelet)
            pose = extract_pose_from_state(state)
            dg_pose: DgLanePose = dg_lanelet.lane_pose_from_SE2_generic(pose)
            if state.vx < 0:
                avg_heading += np.pi
            else:
                avg_heading += abs(dg_pose.relative_heading)
        if max_velocity < state.vx:
            max_velocity = state.vx
        if min_velocity > state.vx:
            min_velocity = state.vx
    avg_heading /= len(ego_log.states)

    task_level = int(sim_context.description.split("_")[0])

    player_metrics = PlayerMetrics(
        task_level=task_level,
        collided=collided,
        goal_reached=has_reached_the_goal,
        min_ttc=min_ttc,
        lane_changing_time=time_to_reach,
        avg_relative_heading=avg_heading,
        max_velocity=max_velocity,
        min_velocity=min_velocity,
        discomfort=discomfort,
        avg_computation_time=avg_comp_time,
    )

    return player_metrics
