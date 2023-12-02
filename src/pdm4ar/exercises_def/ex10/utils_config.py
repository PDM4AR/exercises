from typing import Any

import yaml
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.simulator import SimContext


def _load_config(file_path: str) -> dict[str, Any]:
    with open(file_path, "r") as file:
        config: dict[str, Any] = yaml.safe_load(file)
    return config


def get_sim_context(config_dict: Mapping, seed: Optional[int] = None, config_name: str = "") -> SimContext:
    dgscenario = get_dgscenario(config_dict, seed)
    simcontext = _get_empty_sim_context(dgscenario)
    simcontext.description = f"Environment-{config_name}"

    _, gates = build_road_boundary_obstacle(simcontext.dg_scenario.scenario)

    # add embodied clones of the nominal agent
    agents_dict = config_dict["agents"]
    for pn in agents_dict.keys():
        player_name = PlayerName(pn)
        x0 = DiffDriveState(**agents_dict[pn]["state"])
        goal_n = agents_dict[pn]["goal"]
        goal = PolygonGoal(gates[goal_n].buffer(1))
        color = agents_dict[pn]["color"]
        _add_player(simcontext, x0, player_name, goal=goal, color=color)
    return simcontext


def _add_player(simcontext: SimContext, x0: DiffDriveState, new_name: PlayerName, goal: PlanningGoal,
                color: str = "royalblue"):
    model = DiffDriveModel(x0=x0, vg=DiffDriveGeometry.default(color=color), vp=DiffDriveParameters.default_car())

    new_models: Dict[PlayerName, DiffDriveModel] = {new_name: model}
    new_players = {new_name: Pdm4arAgent(
            sg=deepcopy(model.model_geometry),
            sp=deepcopy(model.model_params)
    )
    }
    # update
    simcontext.models.update(new_models)
    simcontext.players.update(new_players)
    simcontext.missions[new_name] = goal
    return


if __name__ == "__main__":
    from pathlib import Path
    from pprint import pprint

    configs = ["config_planet.yaml", "config_satellites.yaml", "config_mov_target.yaml"]
    for c in configs:
        config_file = Path(__file__).parent / c
        config = _load_config(str(config_file))
        pprint(config)

        # test actual sim context creation
        sim_context = sim_context_from_yaml(str(config_file))
        pprint(sim_context)