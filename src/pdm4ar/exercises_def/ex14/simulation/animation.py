"""Episode animation for ex14.

Follows dg-commons' create_animation, but draws the ex14 goal attributes that the generic version does not know about:
- goals designated to a robot are filled with that robot's color (shared goals stay yellow),
- goal labels show the value and, if any, the designated robot,
- every robot shows how many goals it carries out of its capacity,
- goals carried together are spread along the robot's heading instead of being stacked.
"""

import math
from itertools import chain
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
from dg_commons import PlayerName
from dg_commons.sim.simulator import SimContext
from dg_commons.sim.simulator_animation import adjust_axes_limits, get_lights_colors_from_cmds
from dg_commons.sim.simulator_structures import LogEntry
from dg_commons.sim.simulator_visualisation import SimRenderer, ZOrders
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from shapely.affinity import translate
from tqdm import tqdm

SHARED_GOAL_COLOR = "yellow"


def create_ex14_animation(
    file_path: str,
    sim_context: SimContext,
    figsize: Optional[Union[list, tuple]] = None,
    dt: float = 30,
    dpi: int = 120,
    plot_limits: Union[str, Sequence[Sequence[float]], PlayerName] = "auto",
) -> None:
    """Same arguments as dg_commons.sim.simulator_animation.create_animation."""
    sim_viz = SimRenderer(sim_context, figsize=figsize)
    time_begin = sim_context.log.get_init_time()
    time_end = sim_context.log.get_last_time()
    if not time_begin < time_end:
        raise ValueError(f"Begin time {time_begin} cannot be greater than end time {time_end}")
    ax: Axes = sim_viz.commonroad_renderer.ax
    fig = ax.figure
    fig.tight_layout()
    ax.set_aspect("equal")

    goal_manager = sim_context.shared_goals_manager
    capacities: Mapping[PlayerName, int] = getattr(goal_manager, "agent_capacities", {})
    states, actions, texts, goals, goal_texts, load_texts = {}, {}, {}, {}, {}, {}
    collection_points, collection_point_texts = {}, {}

    def _get_list() -> list[Artist]:
        return (
            list(chain.from_iterable(states.values()))
            + list(chain.from_iterable(actions.values()))
            + list(texts.values())
            + list(goals.values())
            + list(goal_texts.values())
            + list(load_texts.values())
            + list(collection_point_texts.values())
            + list(collection_points.values())
        )

    def _goal_color(goal) -> str:
        owner = getattr(goal, "assigned_robot", None)
        if owner is not None and owner in sim_context.models:
            return sim_context.models[owner].model_geometry.color
        return SHARED_GOAL_COLOR

    def _goal_label(goal_id: str, goal) -> str:
        label = goal_id
        value = getattr(goal, "value", 0)
        if value:
            label += f"\n${value:.0f}"
        owner = getattr(goal, "assigned_robot", None)
        if owner is not None:
            label += f"\n->{owner}"
        return label

    def _carried_at(pname: PlayerName, t: float) -> list:
        """Goals carried by `pname` at time t, in pickup order."""
        carried = [
            g
            for g in goal_manager.all_goals.values()
            if g.collected_by == pname
            and g.collection_time is not None
            and g.collection_time <= t
            and (g.delivery_time is None or t < g.delivery_time)
        ]
        return sorted(carried, key=lambda g: g.collection_time)

    def init_plot() -> Iterable[Artist]:
        ax.clear()
        with sim_viz.plot_arena(ax=ax):
            init_log_entry: Mapping[PlayerName, LogEntry] = sim_context.log.at_interp(time_begin)
            for pname, plog in init_log_entry.items():
                states[pname], actions[pname] = sim_viz.plot_player(
                    ax=ax,
                    state=plog.state,
                    command=plog.commands,
                    lights_colors=get_lights_colors_from_cmds(plog.commands, t=0),
                    player_name=pname,
                    alpha=0.8,
                    plot_wheels=True,
                    plot_lights=True,
                )
            adjust_axes_limits(
                ax=ax,
                plot_limits=plot_limits,
                players_states={p: log_entry.state for p, log_entry in init_log_entry.items()},
            )
            if goal_manager is not None:
                for cp_id, cp in goal_manager.collection_points.items():
                    collection_points[cp_id] = sim_viz.plot_shapely_polygon(
                        ax=ax, spolygon=cp.polygon, color="blue", alpha=0.5, zorder=ZOrders.GOAL
                    )
                    centroid = cp.polygon.centroid
                    collection_point_texts[cp_id] = ax.text(
                        centroid.x, centroid.y, "0", ha="center", va="center", fontsize=30, zorder=ZOrders.TIME_TEXT
                    )
                for goal_id, goal in goal_manager.all_goals.items():
                    goals[goal_id] = sim_viz.plot_shapely_polygon(
                        ax=ax,
                        spolygon=goal.polygon,
                        color=_goal_color(goal),
                        edgecolor="black",
                        linewidth=0.8,
                        zorder=ZOrders.OBJECT,
                    )
                    # the label sits below the goal, since goals are too small to hold it
                    min_x, min_y, max_x, max_y = goal.polygon.bounds
                    goal_texts[goal_id] = ax.text(
                        (min_x + max_x) / 2,
                        min_y - 0.1,
                        _goal_label(goal_id, goal),
                        ha="center",
                        va="top",
                        fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.7, pad=1, linewidth=0),
                        zorder=ZOrders.TIME_TEXT,
                    )
                for pname in sim_context.models:
                    load_texts[pname] = ax.text(
                        0,
                        0,
                        "",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.7, pad=1, linewidth=0),
                        zorder=ZOrders.TIME_TEXT,
                    )
            texts["time"] = ax.text(
                0.02,
                0.96,
                "",
                transform=ax.transAxes,
                bbox=dict(facecolor="lightgreen", alpha=0.5),
                zorder=ZOrders.TIME_TEXT,
            )
        return _get_list()

    def update_plot(frame: int = 0) -> Iterable[Artist]:
        t: float = frame * dt / 1000.0
        log_at_t: Mapping[PlayerName, LogEntry] = sim_context.log.at_interp(t)
        for pname, box_handle in states.items():
            states[pname], actions[pname] = sim_viz.plot_player(
                ax=ax,
                player_name=pname,
                state=log_at_t[pname].state,
                command=log_at_t[pname].commands,
                lights_colors=get_lights_colors_from_cmds(log_at_t[pname].commands, t=t),
                model_poly=box_handle,
                lights_patches=actions[pname],
                plot_wheels=True,
                plot_lights=True,
            )
        if goal_manager is not None:
            for goal_id, goal in goal_manager.all_goals.items():
                collection_time = goal.collection_time if goal.collection_time is not None else float("inf")
                delivery_time = goal.delivery_time if goal.delivery_time is not None else float("inf")
                goals[goal_id].set_visible(t < delivery_time)
                goal_texts[goal_id].set_visible(t < collection_time)
                if collection_time <= t < delivery_time and goal.collected_by is not None:
                    # carried goals ride on the robot, spread along its heading so that all of them are visible
                    state = log_at_t[goal.collected_by].state
                    carried = _carried_at(goal.collected_by, t)
                    slot = next((i for i, g in enumerate(carried) if g is goal), 0)
                    offset = (slot - (len(carried) - 1) / 2.0) * 2.2 * math.sqrt(goal.polygon.area / math.pi)
                    centroid = goal.polygon.centroid
                    moving_goal = translate(
                        goal.polygon,
                        xoff=state.x + offset * math.cos(state.psi) - centroid.x,
                        yoff=state.y + offset * math.sin(state.psi) - centroid.y,
                    )
                    sim_viz.plot_shapely_polygon(ax=ax, spolygon=moving_goal, artist=goals[goal_id])
            for pname, load_text in load_texts.items():
                state = log_at_t[pname].state
                load_text.set_position((state.x, state.y + 0.9))
                load_text.set_text(f"{len(_carried_at(pname, t))}/{capacities.get(pname, 1)}")
            for cp_id, cp in goal_manager.collection_points.items():
                delivered = sum(1 for goal_id in cp.collected_goals if goal_manager.all_goals[goal_id].delivery_time <= t)
                collection_point_texts[cp_id].set_text(str(delivered))

        adjust_axes_limits(
            ax=ax, plot_limits=plot_limits, players_states={p: log_entry.state for p, log_entry in log_at_t.items()}
        )
        texts["time"].set_text(f"t = {t:.2f}s")
        texts["time"].set_transform(ax.transAxes)
        return _get_list()

    # Min frame rate is 1 fps
    dt = min(1000.0, dt)
    frame_count = int(np.ceil(float(time_end - time_begin) / (dt / 1000.0)) + 1)
    plt.ioff()
    anim = FuncAnimation(fig=fig, func=update_plot, init_func=init_plot, frames=frame_count, blit=True, interval=dt)
    if not file_path.endswith((".mp4", ".gif", ".avi")):
        file_path += ".mp4"
    with tqdm(total=frame_count, unit="frame") as progress:
        anim.save(
            file_path,
            dpi=dpi,
            writer="ffmpeg",
            fps=int(math.ceil(1000.0 / dt)),
            progress_callback=lambda *_: progress.update(1),
        )
    ax.clear()
