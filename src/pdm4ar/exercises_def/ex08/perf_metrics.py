from dataclasses import dataclass
from typing import List, Set, Tuple

import numpy as np
from dg_commons import iterate_with_dt, seq_integrate, DgSampledSequence, PlayerName
from dg_commons.maps import DgLanelet, DgLanePose
from dg_commons.sim import extract_pose_from_state
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.simulator import SimContext
from math import sqrt, pi

from pdm4ar.exercises_def import PerformanceResults


@dataclass(frozen=True)
class PlayerMetrics(PerformanceResults):
    player_name: PlayerName
    "Player's name"
    goal_reached: bool
    """Has the player reached the goal?"""
    collided: bool
    """Has the player collided?"""
    travelled_distance: float
    """The length of the trajectory travelled by the robot"""
    episode_duration: float
    """The time it took till the end the simulation."""
    avg_relative_heading: float
    """The average relative heading of the vehicle wrt the lane it is driving on"""
    actuation_effort: float
    """Integral of the commands sent to the robot normalized by the time taken"""
    avg_computation_time: float
    """Average computation time of the get_commands method."""


@dataclass(frozen=True)
class AvgPlayerMetrics(PerformanceResults):
    goal_success_rate: float
    """Average over the players for goal reaching rate."""
    collision_rate: float
    """Average over the players for collision with an obstacle."""
    avg_distance_travelled: float
    """The length of the trajectory travelled by the robot"""
    avg_episode_duration: float
    """The time it took till the end the simulation."""
    avg_relative_heading: float
    """The average relative heading of the vehicle wrt the lane it is driving on"""
    avg_actuation_effort: float
    """Integral of the commands sent to the robot normalized by the time taken."""
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
        score = (self.goal_success_rate - self.collision_rate) * 1e3
        score -= self.avg_relative_heading * 1e2
        score -= (self.avg_distance_travelled + self.avg_episode_duration + self.avg_computation_time +
                  self.avg_actuation_effort) * 1e1
        return score


def ex08_metrics(sim_context: SimContext) -> Tuple[AvgPlayerMetrics, List[PlayerMetrics]]:
    agents_perf: List[PlayerMetrics] = []
    collided_players: Set[PlayerName] = set()
    for cr in sim_context.collision_reports:
        collided_players.update((cr.players.keys()))

    for player_name, agent_log in sim_context.log.items():
        if "PDM4AR" not in player_name:
            continue

        states: DgSampledSequence[VehicleState] = agent_log.states

        # if the last state of the sim is inside the goal
        last_state = states.values[-1]
        has_reached_the_goal: bool = sim_context.missions[player_name].is_fulfilled(last_state)

        # collision
        has_collided = True if player_name in collided_players else False

        # distance travelled
        dist: float = 0
        for it in iterate_with_dt(states):
            dist += sqrt((it.v1.x - it.v0.x) ** 2 + (it.v1.y - it.v0.y) ** 2)

        # time duration
        duration = float(states.get_end() - states.get_start())

        # actuation effort
        abs_acc = agent_log.commands.transform_values(lambda u: abs(u.acc))
        actuation_effort = seq_integrate(abs_acc).values[-1] / duration
        abs_ddelta = agent_log.commands.transform_values(lambda u: abs(u.ddelta))
        actuation_effort += seq_integrate(abs_ddelta).values[-1] / duration

        # lane orientation
        network = sim_context.dg_scenario.lanelet_network
        avg_heading = 0
        for state in agent_log.states.values:
            lanelet_ids = network.find_lanelet_by_position([[state.x, state.y], ])[0]
            if len(lanelet_ids) == 0:
                # penalized for being outside the lanelet network as driving against the traffic
                avg_heading += pi
            else:
                lanelet = network.find_lanelet_by_id(lanelet_ids[0])
                dg_lanelet = DgLanelet.from_commonroad_lanelet(lanelet)
                pose = extract_pose_from_state(state)
                dg_pose: DgLanePose = dg_lanelet.lane_pose_from_SE2_generic(pose)
                avg_heading += abs(dg_pose.relative_heading)
        avg_heading /= len(agent_log.states)

        # computation time
        avg_comp_time = np.average(agent_log.info.values)

        # create the player metrics
        pm = PlayerMetrics(
                player_name=player_name,
                goal_reached=has_reached_the_goal,
                collided=has_collided,
                travelled_distance=dist,
                episode_duration=duration,
                avg_relative_heading=avg_heading,
                actuation_effort=actuation_effort,
                avg_computation_time=avg_comp_time
        )
        agents_perf.append(pm)

    goal_success_rate = [p.goal_reached for p in agents_perf].count(True) / len(agents_perf)
    collision_rate = [p.collided for p in agents_perf].count(True) / len(agents_perf)
    avg_travelled_distance = np.average([p.travelled_distance for p in agents_perf])
    avg_duration = np.average([p.episode_duration for p in agents_perf])
    avg_relative_heading = np.average([p.avg_relative_heading for p in agents_perf])
    avg_actuation_effort = np.average([p.actuation_effort for p in agents_perf])
    avg_computation_time = np.average([p.avg_computation_time for p in agents_perf])

    avg_player_metrics = AvgPlayerMetrics(
            goal_success_rate=goal_success_rate,
            collision_rate=collision_rate,
            avg_distance_travelled=avg_travelled_distance,
            avg_episode_duration=avg_duration,
            avg_relative_heading=avg_relative_heading,
            avg_actuation_effort=avg_actuation_effort,
            avg_computation_time=avg_computation_time
    )

    return avg_player_metrics, agents_perf
