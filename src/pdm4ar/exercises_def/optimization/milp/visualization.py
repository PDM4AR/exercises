from typing import Dict, Optional
from pdm4ar.exercises_def.optimization.milp.data import *
from matplotlib import pyplot as plt
from reprep import Report, MIME_PDF

from pdm4ar.exercises_def.optimization.milp.structures import MilpFeasibility, PirateProblem1, ProblemSolutions


def visualize_journey_plan(
    r: Report, title: str, problem: PirateProblem1, est_solutions: ProblemSolutions, est_costs: Dict, width: int
) -> None:

    r.text(title, "")

    min_x = float('inf')
    min_y = float('inf')
    max_x = 0
    max_y = 0

    for island in problem.islands:
        if island.x > max_x:
            max_x = island.x
        if island.x < min_x:
            min_x = island.x
        if island.y > max_y:
            max_y = island.y
        if island.y < min_y:
            min_y = island.y

    island_radius = 2
    min_x -= 3*island_radius
    max_x += 3*island_radius
    min_y -= 3*island_radius
    max_y += 3*island_radius

    ratio = (max_x-min_x)/(max_y-min_y)

    px = 1/plt.rcParams['figure.dpi']
    
    for name_cost, solution in est_solutions.__dict__.items():

        rfig = r.figure(cols=1)
        if isinstance(est_costs[name_cost], float):
            nid = f"{name_cost}: {est_costs[name_cost]:.2f}"
        else:
            nid = f"{name_cost}: {est_costs[name_cost]}"
        with rfig.plot(
            nid=nid, mime=MIME_PDF, figsize=(width*px*ratio,2000*px)
        ) as _:
            ax = plt.gca()
            ax.grid()

            # Adjust axis
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_aspect(1)

            x1, x2 = ax.get_window_extent().get_points()[:, 0]
            xscale = (x2-x1)/(max_x-min_x)
            y1, y2 = ax.get_window_extent().get_points()[:, 1]
            yscale = (y2-y1)/(max_y-min_y)
            text_size = yscale*island_radius/8

            scale = np.sqrt(xscale**2 + yscale**2)

            # Draw islands
            dict_colors = {0: 'lawngreen', 1: 'orange', 2: 'violet'}
            for island in problem.islands:

                draw_circle = plt.Circle(
                    (island.x, island.y),
                    island_radius,
                    color=dict_colors[island.arch%len(dict_colors.keys())],
                    fill=True,
                    linewidth=0.01,
                )

                text_id_level = plt.Text(island.x, island.y+0.60*island_radius, f"id: {island.id} - arch: {island.arch}", fontsize=text_size, ha='center', va='center')
                text_xy = plt.Text(island.x, island.y+0.25*island_radius, f"x: {island.x:.2f} - y: {island.y:.2f}", fontsize=text_size, ha='center', va='center')
                text_dep_arr = plt.Text(island.x, island.y-0.1*island_radius, f"dep: {island.departure:.2f} - arr: {island.arrival:.2f}", fontsize=text_size, ha='center', va='center')
                text_supply = plt.Text(island.x, island.y-0.40*island_radius, f"t compass: {island.time_compass}", fontsize=text_size, ha='center', va='center')
                text_crew = plt.Text(island.x, island.y-0.7*island_radius, f"crew: {island.crew}",fontsize = text_size, ha='center', va='center')
                
                ax.add_artist(draw_circle)
                ax.add_artist(text_id_level)
                ax.add_artist(text_xy)
                ax.add_artist(text_dep_arr)
                ax.add_artist(text_supply)
                ax.add_artist(text_crew)
        
            if solution is not None:
                if solution.status == MilpFeasibility.feasible:
                    # Draw lines
                    for idx_solution in range(len(solution.voyage_plan)-1):
                        island_id = solution.voyage_plan[idx_solution]
                        next_island_id = solution.voyage_plan[idx_solution+1]
                        island = next((x for x in problem.islands if x.id == island_id), None)
                        next_island = next((x for x in problem.islands if x.id == next_island_id), None)

                        ax.plot(
                            [island.x, next_island.x], [island.y, next_island.y], '-ob', linewidth=0.05*scale*island_radius, markersize=1.15*scale*island_radius, zorder=0)

            ax.figure.savefig("Islands.png")