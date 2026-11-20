from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Mapping

from dg_commons import PlayerName, X, fd
from dg_commons.sim import SimTime, logger
from dg_commons.sim.agents import GlobalPlanner
from dg_commons.sim.models import extract_pose_from_state
from dg_commons.sim.shared_goals import (
    CollectionPoint,
    SharedPolygonGoal,
    SharedPolygonGoalsManager,
)
from dg_commons.sim.simulator import SimContext, Simulator
from dg_commons.sim.simulator_structures import (
    InitSimGlobalObservations,
    PlayerObservations,
    SharedGoalObservation,
    SimObservations,
)
from geometry import translation_from_SE2
from shapely.geometry import Point


@dataclass(frozen=True, unsafe_hash=True)
class Ex14PlayerObservations(PlayerObservations):
    """Player observation enriched with the robot's complete inventory. A robot might be able to carry more than
     one goal at a time."""

    collected_goal_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Ex14SimObservations(SimObservations):
    """Adjusts the runtime observation to include enriched per-agent capacities."""
    players: Mapping[PlayerName, Ex14PlayerObservations]


@dataclass(frozen=True)
class Ex14InitSimGlobalObservations(InitSimGlobalObservations):
    """Adjusts the global planning observation to include enriched per-agent capacities."""

    agent_capacities: Mapping[PlayerName, int] = fd({})


class Ex14SharedPolygonGoalsManager(SharedPolygonGoalsManager):
    """Shared-goal manager supporting per-agent capacities and inventories."""

    def __init__(
        self,
        shared_goals: List[SharedPolygonGoal],
        collection_points: List[CollectionPoint],
        agent_capacities: Mapping[PlayerName, int],
    ):
        super().__init__(shared_goals=shared_goals, collection_points=collection_points)
        
        self.agent_capacities: Dict[PlayerName, int] = dict(agent_capacities)
        
        self.agent_inventories: Dict[PlayerName, List[str]] = {
            name: [] for name in agent_capacities
        }
        self.agent_carrying: Dict[PlayerName, str | None] = {
            name: None for name in agent_capacities
        }

    def update(
        self, agents_states: Dict[PlayerName, X], simulation_time_decimal: SimTime
    ) -> Dict[str, list]:
        """Apply automatic pickups and deliveries for the current simulation step. Wraps the update of the dg-commons
        simulator to include the multi-goal capacity pick-up and delivery transitions"""
        
        simulation_time = float(simulation_time_decimal)
        events: Dict[str, list] = {"goals_collected": [], "goals_delivered": []}

        for agent_name, state in agents_states.items():
            # Goal and collection regions are triggered by the robot's reference point.
            pose = extract_pose_from_state(state)
            agent_point = Point(translation_from_SE2(pose))
            inventory = self.agent_inventories[agent_name]

            # Collect every goal under the robot until its individual capacity is full.
            for goal_id, goal in self.shared_goals.items():
                if len(inventory) >= self.agent_capacities[agent_name]:
                    break
                if not goal.is_collected() and goal.polygon.contains(agent_point):
                    goal.collected_by = agent_name
                    goal.collection_time = simulation_time
                    inventory.append(goal_id)
                    events["goals_collected"].append((agent_name, goal_id))

            # Keep the legacy single-goal field usable by existing dg-commons code.
            self.agent_carrying[agent_name] = inventory[0] if inventory else None

            # Entering any collection point delivers the robot's complete inventory.
            if inventory:
                for point_id, collection_point in self.collection_points.items():
                    if collection_point.polygon.contains(agent_point):
                        for goal_id in tuple(inventory):
                            collection_point.collected_goals.append(goal_id)
                            self.all_goals[goal_id].delivery_time = simulation_time
                            del self.shared_goals[goal_id]
                            events["goals_delivered"].append(
                                (agent_name, goal_id, point_id)
                            )
                        inventory.clear()
                        self.agent_carrying[agent_name] = None
                        break

        return events

    def get_agent_inventory(self, agent_name: PlayerName) -> tuple[str, ...]:
        return tuple(self.agent_inventories[agent_name])


class Ex14GlobalPlanner(GlobalPlanner):
    """Adds exercise-specific capacity data before calling the student planner."""

    def __init__(
        self, planner: GlobalPlanner, agent_capacities: Mapping[PlayerName, int]
    ):
        self.planner = planner
        self.agent_capacities = dict(agent_capacities)

    def send_plan(self, init_sim_global_obs: InitSimGlobalObservations) -> str:
        enriched_obs = Ex14InitSimGlobalObservations(
            players_obs=init_sim_global_obs.players_obs,
            initial_states=init_sim_global_obs.initial_states,
            seed=init_sim_global_obs.seed,
            dg_scenario=init_sim_global_obs.dg_scenario,
            shared_goals=init_sim_global_obs.shared_goals,
            collection_points=init_sim_global_obs.collection_points,
            agent_capacities=fd(self.agent_capacities),
        )
        return self.planner.send_plan(enriched_obs)


class Ex14Simulator(Simulator):
    """Simulator variant that exposes each robot's complete inventory."""

    last_observations = Ex14SimObservations(players=fd({}), time=Decimal(0))

    def pre_update(self, sim_context: SimContext):
        if not self._need_to_update_commands(sim_context):
            return

        goal_manager = sim_context.shared_goals_manager
        players_observations: dict[PlayerName, Ex14PlayerObservations] = {}
        for player_name in sim_context.players:
            model = sim_context.models[player_name]
            inventory = (
                goal_manager.get_agent_inventory(player_name)
                if goal_manager is not None
                else ()
            )
            players_observations[player_name] = Ex14PlayerObservations(
                state=model.get_state(),
                occupancy=model.get_footprint(),
                collected_goal_id=inventory[0] if inventory else None,
                collected_goal_ids=inventory,
            )

        available_goals_obs = None
        if goal_manager is not None:
            available_goals_obs = {
                goal.goal_id: SharedGoalObservation(occupancy=goal.polygon)
                for goal in goal_manager.get_available_goals()
            }

        self.last_observations = Ex14SimObservations(
            players=fd(players_observations),
            time=sim_context.time,
            available_goals=fd(available_goals_obs) if available_goals_obs else None,
        )

        logger.debug(f"Pre update function, sim time {sim_context.time}")
        logger.debug(f"Last observations:\n{self.last_observations}")
