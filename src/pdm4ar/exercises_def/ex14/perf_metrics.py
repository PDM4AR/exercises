from dataclasses import dataclass
from math import sqrt
from typing import List, Set, Tuple

import numpy as np
from dg_commons import DgSampledSequence, PlayerName, iterate_with_dt, seq_integrate
from dg_commons.sim.models.diff_drive import DiffDriveState
from dg_commons.sim.simulator import SimContext
from pdm4ar.exercises_def import PerformanceResults
from shapely.geometry import Point, Polygon


@dataclass(frozen=True)
class PlayerMetrics(PerformanceResults):
    player_name: PlayerName
    "Player's name"
    collided: bool
    """Has the player crashed?"""
    num_goal_delivered: int
    """Number of goals delivered by the player."""
    travelled_distance: float
    """Distance travelled by the player."""
    waiting_time: float
    """Total waiting time before the goals are delivered by the player."""
    actuation_effort: float
    """Actuation effort of the player in the simulation."""
    avg_computation_time: float
    """Average computation time of the get_commands method."""


@dataclass(frozen=True)
class AllPlayerMetrics(PerformanceResults):
    num_collided_players: int
    """Number of players that crashed."""
    num_goals_delivered: int
    """Number of goals delivered by all players."""
    total_travelled_distance: float
    """Total distance travelled by all players."""
    total_waiting_time: float
    """Total waiting time before all goals get delivered by all players."""
    total_actuation_effort: float
    """Total actuation effort of all players in the simulation."""
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
        score = self.num_goals_delivered * 100
        score -= self.num_collided_players * 500
        score -= self.total_travelled_distance * 0.1
        score -= self.total_waiting_time * 0.1
        score -= self.total_actuation_effort * 0.1
        score -= self.avg_computation_time * 100
        return score


def ex14_metrics(sim_context: SimContext) -> Tuple[AllPlayerMetrics, List[PlayerMetrics]]:
    agents_perf: List[PlayerMetrics] = []
    collided_players: Set[PlayerName] = set()
    for cr in sim_context.collision_reports:
        collided_players.update((cr.players.keys()))

    goal_manager = sim_context.shared_goals_manager
    for player_name, agent_log in sim_context.log.items():
        if "PDM4AR" not in player_name:
            continue
        
        goals_delivered = goal_manager.get_goals_delivered_by_agent(player_name)
        # number of goals delivered
        num_goal_delivered = len(goals_delivered)
        # total waiting time before all goals get delivered
        total_waiting_time = sum(goal_manager.all_goals[goal_id].delivery_time for goal_id in goals_delivered)
        # collision
        has_collided = True if player_name in collided_players else False

        states: DgSampledSequence[DiffDriveState] = agent_log.states
        # distance travelled
        dist: float = 0
        for it in iterate_with_dt(states):
            dist += sqrt((it.v1.x - it.v0.x) ** 2 + (it.v1.y - it.v0.y) ** 2)

        # actuation effort
        abs_omega_l = agent_log.commands.transform_values(lambda u: abs(u.omega_l))
        actuation_effort = seq_integrate(abs_omega_l).values[-1]
        abs_omega_r = agent_log.commands.transform_values(lambda u: abs(u.omega_r))
        actuation_effort += seq_integrate(abs_omega_r).values[-1]

        # computation time
        avg_comp_time = np.average(agent_log.info.values)

        # create the player metrics
        pm = PlayerMetrics(
            player_name=player_name,
            collided=has_collided,
            num_goal_delivered=num_goal_delivered,
            waiting_time=total_waiting_time,
            travelled_distance=dist,
            actuation_effort=actuation_effort,
            avg_computation_time=avg_comp_time,
        )
        agents_perf.append(pm)

    num_collided_players = [p.collided for p in agents_perf].count(True)
    num_goals_delivered = sum([p.num_goal_delivered for p in agents_perf])
    total_travelled_distance = sum([p.travelled_distance for p in agents_perf])
    total_waiting_time = sum([p.waiting_time for p in agents_perf])
    total_actuation_effort = sum([p.actuation_effort for p in agents_perf])
    avg_computation_time = sum([p.avg_computation_time for p in agents_perf]) / len(agents_perf)

    all_player_metrics = AllPlayerMetrics(
        num_collided_players=num_collided_players,
        num_goals_delivered=num_goals_delivered,
        total_travelled_distance=total_travelled_distance,
        total_waiting_time=total_waiting_time,
        total_actuation_effort=total_actuation_effort,
        avg_computation_time=avg_computation_time,
    )
    return all_player_metrics, agents_perf