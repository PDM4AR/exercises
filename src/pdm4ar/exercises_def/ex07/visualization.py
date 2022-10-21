from typing import Optional
from pdm4ar.exercises_def.ex07.data import *
from matplotlib import pyplot as plt
from reprep import Report, MIME_PDF

from pdm4ar.exercises_def.ex07.structures import MilpFeasibility, ProblemVoyage1, \
    ProblemSolutions, SolutionViolations, SolutionsCosts, CostTolerance, ReportType


# -----------------------------------    REPORT  TYPE    -----------------------------------------
#                                                                                                |
REPORT_TYPE = ReportType.report_viz # choose among none, terminal, report_txt, and report_viz    |
#                                                                                                | 
#-------------------------------------------------------------------------------------------------

class Viz:
    
    def __init__(self, report_type=REPORT_TYPE):
        self.report_type = report_type
        print(f"Report type: {self.report_type.name}\n")

    def display_text_terminal(self, name_cost, gt_cost, est_cost, timing, violations) -> None:

        if gt_cost is not None and CostTolerance.compare(gt_cost, est_cost):
            prefix, suffix = "\033[31m", "\033[0m"
        else:
            prefix, suffix = "", ""
        good, bad = "",""  #"  \U0001F603", "  \U0001F641"
        
        text_cost = prefix + f"{name_cost} cost: " + (f"{est_cost:.2f}" if isinstance(est_cost, float) else f"{est_cost}") + suffix
        if gt_cost is not None:
            text_cost += f"GT cost: " + (f"{gt_cost:.2f}" if isinstance(gt_cost, float) else f"{gt_cost}") + \
                        (good if CostTolerance.compare(gt_cost, est_cost) else bad)

        print(text_cost)

        if timing is not None:
            print('\033[31m'+"Time exceeded: "+'\033[0m'+f"{1000*timing:.3f} ms")
        
        if violations is not None:
            text_violation = ""
            if getattr(violations, name_cost) is not None:
                for violation_name, violation in getattr(violations, name_cost).__dict__.items():
                    if violation is not None:
                        text_violation += f"{violation_name}: " + \
                            (f"{violation}\n" if isinstance(violation, int) else f"{violation:.3f}\n")
                if len(text_violation) > 0:
                    print('\033[31m'+"Violations:"+'\033[0m'+f"\n{text_violation}")

    def display_text_report(self, r_viz, name_cost, gt_cost, est_cost, timing, violations) -> None:

        r_viz.text("Cost: ", f"{est_cost:.2f}" if isinstance(est_cost, float) else f"{est_cost}")
        if gt_cost is not None:
            good, bad = "",""  #"  \U0001F603", "  \U0001F641"
            r_viz.text("GT cost: ", (f"{gt_cost:.2f}" if isinstance(gt_cost, float) else f"{gt_cost}") +
                        (good if CostTolerance.compare(gt_cost, est_cost) else bad))

        if timing is not None:
            r_viz.text("Time exceeded: ", f"{1000*timing:.3f} ms")
        if violations is not None:
            text_violation = ""
            if getattr(violations, name_cost) is not None:
                for violation_name, violation in getattr(violations, name_cost).__dict__.items():
                    if violation is not None:
                        text_violation += f"{violation_name}: " + \
                            (f"{violation}\n" if isinstance(violation, int) else f"{violation:.3f}\n")
                if len(text_violation) > 0:
                    r_viz.text("Violations: ", text_violation)

    def visualize(
        self, r: Report, problem: ProblemVoyage1, gt_costs: SolutionsCosts,
        est_solutions: Optional[ProblemSolutions] = None, est_costs: Optional[SolutionsCosts] = None,
        violations: Optional[SolutionViolations] = None, timing: Optional[float] = None
    ) -> None:

        if self.report_type == ReportType.none:
            return

        if self.report_type >= ReportType.report_viz:
            width_image = 200
            island_radius_viz = 2
            text_size_m = 0.2 # meters

            min_x = min(problem.islands, key = lambda island: island.x).x
            min_y = min(problem.islands, key = lambda island: island.y).y
            max_x = max(problem.islands, key = lambda island: island.x).x
            max_y = max(problem.islands, key = lambda island: island.y).y

            min_x -= 3*island_radius_viz
            max_x += 3*island_radius_viz
            min_y -= 3*island_radius_viz
            max_y += 3*island_radius_viz

            ratio = (max_x-min_x)/(max_y-min_y)

            px = 1/plt.rcParams['figure.dpi']      

            dict_colors = {0: 'lawngreen', 1: 'orange', 2: 'violet', 3: 'gold', 4: 'coral'}
        
        for name_cost, solution in est_solutions.__dict__.items():

            with r.subsection(name_cost) as r_sub:

                if self.report_type >= ReportType.report_viz:
                    
                    rfig = r_sub.figure(cols=1)
                    
                    with rfig.plot(
                        nid=" ", mime=MIME_PDF, figsize=(width_image*px*ratio,width_image*px) if width_image is not None else None
                    ) as _:

                        # display_text(rfig, value_cost, timing, violations, name_cost)

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
                        text_size = yscale*text_size_m

                        scale = np.sqrt(xscale**2 + yscale**2)

                        # Draw islands
                        for island in problem.islands:

                            draw_circle = plt.Circle(
                                (island.x, island.y),
                                island_radius_viz,
                                color=dict_colors[island.arch%len(dict_colors.keys())],
                                fill=True,
                                linewidth=0.01,
                            )

                            text_id_level = plt.Text(island.x, island.y+0.60*island_radius_viz, 
                                f"id: {island.id} - arch: {island.arch}", fontsize=text_size, ha='center', va='center')
                            text_xy = plt.Text(island.x, island.y+0.25*island_radius_viz, 
                                f"x: {island.x:.1f} - y: {island.y:.1f}", fontsize=text_size, ha='center', va='center')
                            text_dep_arr = plt.Text(island.x, island.y-0.1*island_radius_viz, 
                                f"d: {island.departure:.1f} - a: {island.arrival:.1f}", fontsize=text_size, ha='center', va='center')
                            text_supply = plt.Text(island.x, island.y-0.40*island_radius_viz, 
                                f"t compass: {island.time_compass}", fontsize=text_size, ha='center', va='center')
                            text_crew = plt.Text(island.x, island.y-0.7*island_radius_viz, 
                                f"crew: {island.crew}",fontsize = text_size, ha='center', va='center')
                            
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
                                        [island.x, next_island.x], [island.y, next_island.y], zorder=0,
                                        color='cornflowerblue', linestyle='solid', linewidth=0.05*scale*island_radius_viz, 
                                        marker='o', markerfacecolor='cornflowerblue', markersize=1.15*scale*island_radius_viz)

                gt_cost = getattr(gt_costs, name_cost) if gt_costs is not None else None
                est_cost = getattr(est_costs, name_cost)

                if self.report_type >= ReportType.terminal:
                    self.display_text_terminal(name_cost, gt_cost, est_cost, timing, violations)

                if self.report_type >= ReportType.report_txt:
                    self.display_text_report(r_sub, name_cost, gt_cost, est_cost, timing, violations)
