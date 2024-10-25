from dataclasses import dataclass
from typing import List, Set

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
    player_name: PlayerName
    "Player's name"
    collided: bool
    """Has the player collided?"""
    collided_with: List[PlayerName]
    """List of players with which the player has collided"""
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

    def __repr__(self):
        repr: str = ""
        for k, v in self.__dict__.items():
            value = f"{v:>5.2f}" if isinstance(v, float) else f"{v:>5}"
            repr += f"\t{k:<20}=\t" + value + ",\n"

        return f"EpisodeOutcome(\n" + repr + "\n)"

    def reduce_to_score(self) -> float:
        """Higher is better"""
        max_score = 100
        if self.collided:
            max_score *= 0.2
        if not self.goal_reached:
            max_score *= 0.5
        score = max_score
        # lose points if the lane changing takes too long
        if self.lane_changing_time is not None:
            score -= 2.0 * np.maximum(0.0, float(self.lane_changing_time) - 5.0)
        else:
            score -= 15.0
        # lose points if the planned trajectory is risky
        score -= 15.0 * np.maximum(0.0, -self.min_ttc + 1.0)
        # lose points if the computation takes too long
        score -= 5.0 * np.maximum(0.0, self.avg_computation_time - 0.5)
        # lose points if the planned trajectory is not comfortable
        score -= 5.0 * np.maximum(0.0, self.discomfort - 0.6)
        # lose points if the vehicle is not compliant with the lane
        score -= 5.0 * np.maximum(0.0, np.abs(self.avg_relative_heading) - 0.1)
        # lose points if the vehicle is too fast or too slow
        v_diff = np.maximum(self.max_velocity - 15.0, 5.0 - self.min_velocity)
        score -= 5.0 * np.maximum(0.0, v_diff)
        return np.maximum(0.0, score)


def ex12_metrics(sim_context: SimContext) -> PlayerMetrics:
    collided_players: Set[PlayerName] = set()
    for cr in sim_context.collision_reports:
        collided_players.update((cr.players.keys()))

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
            # penalized for being outside the lanelet network as driving against the traffic
            avg_heading += np.pi
        else:
            lanelet = lanelet_network.find_lanelet_by_id(lanelet_ids[0])
            dg_lanelet = DgLanelet.from_commonroad_lanelet(lanelet)
            pose = extract_pose_from_state(state)
            dg_pose: DgLanePose = dg_lanelet.lane_pose_from_SE2_generic(pose)
            avg_heading += abs(dg_pose.relative_heading)
        if max_velocity < state.vx:
            max_velocity = state.vx
        if min_velocity > state.vx:
            min_velocity = state.vx
    avg_heading /= len(ego_log.states)

    player_metrics = PlayerMetrics(
        player_name=ego_name,
        collided=collided,
        collided_with=list(collided_players),
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
