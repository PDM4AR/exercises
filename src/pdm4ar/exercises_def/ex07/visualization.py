from enum import IntEnum
from typing import Optional
from matplotlib import pyplot as plt
from reprep import Report, MIME_PDF

from .structures import *
from .data import *

# -------------------------------------       VIZ  OPTIONS     ------------------------------------------
#
# Choose among none, terminal, report_txt, report_viz, report_viz_extra
REPORT_TYPE = ReportType.report_viz
#
# Choose width of the image: small value for fast rendering,
# trading off in large maps non-overlapping of figures, and most importantly
# readability of the extra text if "report_viz_extra" is selected.
FIGURE_WIDTH = 800
#
# Set to False to remove colors from terminal and enhance contrast in the report.
ACTIVATE_COLORS = True
#
# --------------------------------------------------------------------------------------------------------


class Viz:
    def __init__(
        self,
        report_type: IntEnum = REPORT_TYPE,
        fig_width: int = FIGURE_WIDTH,
        activate_colors: bool = ACTIVATE_COLORS,
    ):
        self.report_type = report_type
        self.activate_colors = activate_colors
        # self.print_report_type()
        self.fig_width = fig_width

    def print_report_type(self) -> None:
        pre, post = ("\033[33;100m", "\033[0m") if self.activate_colors else ("", "")
        report_type_text = f"Report type: {self.report_type.name}"
        print(f"\n{pre}{report_type_text}{post}\n")

    def print_title(self, title: str) -> None:
        pre, post = (
            ("\033[55;45m  ", "  \033[0m") if self.activate_colors else ("# ", "")
        )
        print(f"\n\n{pre}{title}{post}")

    def display_text_terminal(
        self,
        problem: ProblemVoyage,
        feasibility_score: int,
        voyage_plan: Optional[List[int]],
        est_cost: Cost,
        gt_cost: Cost,
        cost_score: CostScore,
        timing: float,
        violations: Violations,
    ) -> None:

        if self.activate_colors:
            good, bad = "  \U0001F603", "  \U0001F641"
            green, red, reset = "\033[38;2;0;255;0m", "\033[38;2;255;0;0m", "\033[0m"
            arrow = ""
        else:
            good, bad, green, red, reset, arrow = "", "", "", "", "", " <--"

        if timing is not None:
            print(red + "Time exceeded: " + reset + f"{1000*timing:.3f} ms")

        text_feasibility = f"\nFeasibility: \n\tEst.: {est_cost.feasibility.name}"
        if gt_cost is not None:
            text_feasibility += f"\n\tGT: {gt_cost.feasibility.name}"
            text_feasibility += f"\n\t{green if feasibility_score == 1 else red}Score: {feasibility_score}{reset}"
        else:
            text_feasibility += f"\n\tScore: {feasibility_score}"
        print(text_feasibility)

        n_violations = 0

        text_violation = ""
        n_constraints = 0
        for violation_name in violations.__annotations__.keys():
            constraint = getattr(problem.constraints, violation_name, "")
            if constraint is not None:
                n_constraints += 1
                violation = getattr(violations, violation_name)
                if violation is not None:
                    text_violation += (
                        f"\n\t{red if violation else green}{violation_name}"
                        + f"{': ' if violation_name != 'voyage_order' else ''}"
                        + (
                            f"{constraint:.2f}"
                            if isinstance(constraint, float)
                            else f"{constraint}"
                        )
                        + f"{arrow}{reset}"
                    )
                    n_violations += 1 if violation else 0

        fraction_violations = f"{n_violations}/{n_constraints}"
        # text_violation = f"\nConstraints {red if n_violations > 0 else green}" + \
        #                  f"{fraction_violations} violation{'s' if n_violations != 1 else ''}" + \
        #                  f":{reset}{text_violation}"
        if not self.activate_colors:
            text_violation = (
                f"\n\t{fraction_violations} violation{'s' if n_violations != 1 else ''}"
                + text_violation
            )
        text_violation = "\nConstraints: " + text_violation
        print(text_violation)

        cost = est_cost.cost

        text_cost = "\nCost: "
        if est_cost.feasibility == MilpFeasibility.feasible:
            text_cost += "\n\tEst.: " + (
                f"{cost:.3f}" if isinstance(cost, float) else f"{cost}"
            )
        else:
            text_cost += f"\n\tEst.: {est_cost.feasibility.name}"

        if gt_cost is not None:
            if gt_cost.feasibility == MilpFeasibility.feasible:
                cost = gt_cost.cost
                text_cost += "\n\tGT: " + (
                    f"{cost:.3f}" if isinstance(cost, float) else f"{cost}"
                )
            else:
                text_cost += f"\n\tGT: {gt_cost.feasibility.name}"

            if (
                feasibility_score == 1
                and est_cost.feasibility == MilpFeasibility.feasible
            ) or (
                feasibility_score == 0
                and est_cost.feasibility == MilpFeasibility.unfeasible
            ):
                if self.activate_colors:
                    r, g, b = self.get_cost_score_color(cost_score.cost.cost)
                    color = f"\033[38;2;{r};{g};{b}m"
                else:
                    color = ""
                text_cost += (
                    "\n\t" + color + f"Score: {cost_score.cost.cost:.3f}" + reset
                )
        else:
            text_cost += f"\n\tScore: {cost_score.cost.cost}"

        print(text_cost, "\n")

    def display_text_report(
        self,
        r_viz,
        problem: ProblemVoyage,
        feasibility_score: int,
        voyage_plan: Optional[List[int]],
        est_cost: Cost,
        gt_cost: Cost,
        cost_score: CostScore,
        timing: float,
        violations: Violations,
    ) -> None:

        text_problem = ""
        for problem_data_name in problem.__annotations__.keys():
            problem_data = getattr(problem, problem_data_name)
            if isinstance(problem_data, IntEnum):
                text_problem += f"{problem_data_name}: " + f"{problem_data.name}\n"
            elif isinstance(problem_data, int):
                text_problem += f"{problem_data_name}: " + f"{problem_data}\n"
            elif isinstance(problem_data, Constraints):
                for constraint_name in problem_data.__annotations__.keys():
                    constraint = getattr(problem_data, constraint_name)
                    text_problem += f"{constraint_name}: " + (
                        f"{constraint:.2f}\n"
                        if isinstance(constraint, float)
                        else f"{constraint}\n"
                    )
        r_viz.text("Problem:", text_problem)

        if timing is not None:
            r_viz.text("Time exceeded: ", f"{1000*timing:.3f} ms")

        text_feasibility = f"Est.: {est_cost.feasibility.name}"
        if gt_cost is not None:
            text_feasibility += f"\nGT: {gt_cost.feasibility.name}"
            text_feasibility += f"\nScore: {feasibility_score}"
        r_viz.text("Feasibility:", text_feasibility)

        if est_cost.feasibility == MilpFeasibility.feasible:
            text_solution = "".join(list(map(lambda u: f"{u}->", voyage_plan)))[:-2]
            r_viz.text("Solution:", text_solution)

        text_violation = ""
        n_constraints = 0
        n_violations = 0
        for violation_name in violations.__annotations__.keys():
            constraint = getattr(problem.constraints, violation_name, "")
            if constraint is not None:
                n_constraints += 1
                violation = getattr(violations, violation_name)
                if violation is not None:
                    text_violation += (
                        "\u2716" if violation else "\u2714"
                    ) + f" {violation_name}\n"
                    n_violations += 1 if violation else 0

        fraction_violations = f"{n_violations}/{n_constraints}"
        text_violation = (
            f"{fraction_violations} violation{'s' if n_violations != 1 else ''}:\n"
            + text_violation
        )
        r_viz.text("Constraints: ", text_violation)

        if est_cost.feasibility == MilpFeasibility.feasible:
            cost = est_cost.cost
            text_cost = "Est.: " + (
                f"{cost:.3f}" if isinstance(cost, float) else f"{cost}"
            )
        else:
            text_cost = "Est.: " + est_cost.feasibility.name

        if gt_cost is not None:
            if gt_cost.feasibility == MilpFeasibility.feasible:
                cost = gt_cost.cost
                text_cost += f"\nGT: " + (
                    f"{cost:.3f}" if isinstance(cost, float) else f"{cost}"
                )
            else:
                text_cost += f"\nGT: {gt_cost.feasibility.name}"
            if (
                feasibility_score == 1
                and est_cost.feasibility == MilpFeasibility.feasible
            ) or (
                feasibility_score == 0
                and est_cost.feasibility == MilpFeasibility.unfeasible
            ):
                text_cost += f"\nScore: {cost_score.cost.cost:.3f}"
        else:
            text_cost += f"\nScore: {cost_score.cost.cost}"
        r_viz.text("Cost:", text_cost)

    @staticmethod
    def get_cost_score_color(value) -> Tuple[int, int, int]:
        if value <= 0.5:
            rgb = 1.0, value / 0.5, 0
        else:
            rgb = (1 - value) / 0.5, 1.0, 0

        rgb = tuple([min(255, max(round(255 * color), 0)) for color in rgb])

        return rgb

    def visualize(
        self,
        r_sec: Report,
        problem: ProblemVoyage,
        feasibility_score: int,
        gt_cost: Optional[Cost],
        cost_score: Cost,
        voyage_plan: Optional[List[int]],
        est_cost: Optional[Cost],
        violations: Optional[Violations] = None,
        timing: Optional[float] = None,
    ) -> None:

        if self.report_type == ReportType.none:
            return

        if self.report_type >= ReportType.terminal:
            self.display_text_terminal(
                problem,
                feasibility_score,
                voyage_plan,
                est_cost,
                gt_cost,
                cost_score,
                timing,
                violations,
            )

        if self.report_type >= ReportType.report_viz:
            island_radius_viz = (
                2  # THIS IS JUST FOR VISUALIZATION, NO RELATED TO PROBLEM SOLVING
            )
            text_size_m = 0.2  # meters

            min_x = min(problem.islands, key=lambda island: island.x).x
            min_y = min(problem.islands, key=lambda island: island.y).y
            max_x = max(problem.islands, key=lambda island: island.x).x
            max_y = max(problem.islands, key=lambda island: island.y).y

            min_x -= 3 * island_radius_viz
            max_x += 3 * island_radius_viz
            min_y -= 3 * island_radius_viz
            max_y += 3 * island_radius_viz

            ratio = (max_x - min_x) / (max_y - min_y)

            px = 1 / plt.rcParams["figure.dpi"]

            fig_size = (
                (self.fig_width * px * ratio, self.fig_width * px)
                if self.fig_width is not None
                else None
            )

            if self.activate_colors:
                arch_island_colors = {
                    0: "lawngreen",
                    1: "orange",
                    2: "violet",
                    3: "gold",
                    4: "coral",
                }
                arch_text_colors = {key: "black" for key in arch_island_colors.keys()}
            else:
                arch_island_colors = {0: "black", 1: "yellow"}
                arch_text_colors = {0: "yellow", 1: "black"}

        with r_sec.subsection("") as r_sub:

            if self.report_type >= ReportType.report_viz:

                rfig = r_sub.figure(cols=1)

                with rfig.plot(nid=" ", mime=MIME_PDF, figsize=fig_size) as _:

                    ax = plt.gca()
                    ax.grid(zorder=0)

                    # Adjust axis
                    ax.set_xlim(min_x, max_x)
                    ax.set_ylim(min_y, max_y)
                    ax.set_aspect(1)

                    x1, x2 = ax.get_window_extent().get_points()[:, 0]
                    xscale = (x2 - x1) / (max_x - min_x)
                    y1, y2 = ax.get_window_extent().get_points()[:, 1]
                    yscale = (y2 - y1) / (max_y - min_y)
                    text_size = yscale * text_size_m

                    scale = np.sqrt(xscale**2 + yscale**2)

                    # Draw islands
                    islands_pos = np.array(
                        [[island.x, island.y] for island in problem.islands]
                    )

                    islands_color = [
                        arch_island_colors[island.arch % len(arch_island_colors.keys())]
                        for island in problem.islands
                    ]

                    ax.scatter(
                        islands_pos[:, 0],
                        islands_pos[:, 1],
                        (island_radius_viz * scale) ** 2,
                        islands_color,
                        zorder=1,
                    )

                    if self.report_type >= ReportType.report_viz_extra:
                        for island in problem.islands:
                            text_color = arch_text_colors[
                                island.arch % len(arch_text_colors.keys())
                            ]
                            ax.text(
                                island.x,
                                island.y + 0.6 * island_radius_viz,
                                f"id: {island.id} - arch: {island.arch}",
                                fontsize=text_size,
                                ha="center",
                                va="center",
                                color=text_color,
                                zorder=2,
                            )
                            ax.text(
                                island.x,
                                island.y + 0.25 * island_radius_viz,
                                f"x: {island.x:.1f} - y: {island.y:.1f}",
                                fontsize=text_size,
                                ha="center",
                                va="center",
                                color=text_color,
                                zorder=2,
                            )
                            ax.text(
                                island.x,
                                island.y - 0.1 * island_radius_viz,
                                f"d: {island.departure:.1f} - a: {island.arrival:.1f}",
                                fontsize=text_size,
                                ha="center",
                                va="center",
                                color=text_color,
                                zorder=2,
                            )
                            ax.text(
                                island.x,
                                island.y - 0.4 * island_radius_viz,
                                f"t compass: {island.nights}",
                                fontsize=text_size,
                                ha="center",
                                va="center",
                                color=text_color,
                                zorder=2,
                            )
                            ax.text(
                                island.x,
                                island.y - 0.7 * island_radius_viz,
                                f"delta crew: {island.delta_crew}",
                                fontsize=text_size,
                                ha="center",
                                va="center",
                                color=text_color,
                                zorder=2,
                            )
                            # ax.annotate(f"id: {island.id}\n - arch: {island.arch}", (island.x, island.y))

                    if (
                        est_cost.feasibility == MilpFeasibility.feasible
                        and voyage_plan is not None
                    ):
                        # Draw lines
                        # for idx_solution in range(len(est_solution.voyage_plan)-1):
                        #     island_id = est_solution.voyage_plan[idx_solution]
                        #     next_island_id = est_solution.voyage_plan[idx_solution+1]
                        #     island = next((x for x in problem.islands if x.id == island_id), None)
                        #     next_island = next((x for x in problem.islands if x.id == next_island_id), None)

                        #     ax.plot(
                        #         [island.x, next_island.x], [island.y, next_island.y], zorder=0,
                        #         color='cornflowerblue', linestyle='solid', linewidth=1,
                        #         marker='o', markerfacecolor='cornflowerblue', markersize=1.3*scale*island_radius_viz)
                        islands_pos = []
                        for island_id in voyage_plan:
                            island = problem.islands[island_id]
                            islands_pos.append([island.x, island.y])
                        islands_pos = np.array(islands_pos)
                        ax.plot(
                            islands_pos[:, 0],
                            islands_pos[:, 1],
                            zorder=0,
                            color="cornflowerblue",
                            linestyle="solid",
                            linewidth=0.2 * scale * island_radius_viz,
                            marker="o",
                            markerfacecolor="cornflowerblue",
                            markersize=1.3 * scale * island_radius_viz,
                        )

            if self.report_type >= ReportType.report_txt:
                self.display_text_report(
                    r_sub,
                    problem,
                    feasibility_score,
                    voyage_plan,
                    est_cost,
                    gt_cost,
                    cost_score,
                    timing,
                    violations,
                )
