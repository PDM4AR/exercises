from dg_commons.sim.simulator_visualisation import SimRenderer, ZOrders
from matplotlib import pyplot as plt
from pdm4ar.exercises_def.ex14 import sim_context_from_yaml

if __name__ == "__main__":
    # matplotlib.use('TkAgg')
    from pathlib import Path

    configs = ["config_1.yaml", "config_2.yaml"]
    for c in configs:
        config_file = Path(__file__).parents[2] / "exercises_def/ex14" / c
        # test actual sim context creation
        sim_context = sim_context_from_yaml(str(config_file))
        sim_renderer = SimRenderer(sim_context)
        shapely_viz = sim_renderer.shapely_viz
        ax = sim_renderer.commonroad_renderer.ax

        with sim_renderer.plot_arena(ax):
            for s_obstacle in sim_context.dg_scenario.static_obstacles:
                shapely_viz.add_shape(s_obstacle.shape, color=s_obstacle.geometry.color, zorder=ZOrders.ENV_OBSTACLE)
            for pn, goal in sim_context.missions.items():
                shapely_viz.add_shape(
                    goal.get_plottable_geometry(),
                    color=sim_context.models[pn].model_geometry.color,
                    zorder=ZOrders.GOAL,
                    alpha=0.5,
                )
            for pn, model in sim_context.models.items():
                footprint = model.get_footprint()
                shapely_viz.add_shape(
                    footprint, color=sim_context.models[pn].model_geometry.color, zorder=ZOrders.MODEL, alpha=0.5
                )

            ax = shapely_viz.ax
            ax.autoscale()
            ax.set_aspect("equal")
            file_name = config_file.name.split(".")[0] + ".png"
            plt.savefig(file_name, dpi=300)
            plt.close()
