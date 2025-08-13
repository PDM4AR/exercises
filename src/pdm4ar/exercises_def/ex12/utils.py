from functools import reduce
from itertools import permutations
from operator import attrgetter
from reprep import MIME_GIF, MIME_PNG, Report
import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph

# from zuper_commons.text import remove_escapes
# from zuper_typing import debug_print
from dg_commons import PlayerName
from dg_commons.sim import CollisionReport
from dg_commons.sim.simulator import SimContext
from dg_commons.sim.collision_structures import combine_collision_reports
from dg_commons.sim.collision_visualisation import plot_collision


def investigate_collision_report(coll_reports: list[CollisionReport]) -> tuple[list[CollisionReport], DiGraph]:
    """We get a collision report for every step of the simulation in which a collision is detected.
    Yet an accident is *one* accident even if the two cars are in a collision state for multiple simulation
    steps.
    This function aims to compress the list of collision report to a list of "accidents"
    Assumptions:
        - For each episode two players can have an accidents with each other only once
    :param coll_reports: The original list of collision reports
    """

    players_involved: set[PlayerName] = set()
    accidents: set[tuple[PlayerName]] = set()
    for report in coll_reports:
        players = tuple(report.players.keys())
        for p in players:
            players_involved.add(p)
        accidents.add(players)

    # We represent the chain of accidents as a direct graph
    Gcoll = DiGraph()
    for player in players_involved:
        ts_first_collision = min([report.at_time for report in coll_reports if player in report.players])
        Gcoll.add_node(player, ts_first_collision=ts_first_collision)

    for players in accidents:
        # we add an edge from p1->p2 iff p1["ts_first_collision"]<=p2["ts_first_collision"]
        for p1, p2 in permutations(players):
            if Gcoll.nodes[p1]["ts_first_collision"] <= Gcoll.nodes[p2]["ts_first_collision"]:
                Gcoll.add_edge(p1, p2)

    list_accidents: list[CollisionReport] = []
    for involved_ps in accidents:
        report_involved_ps = [r for r in coll_reports if set(involved_ps) == set(r.players.keys())]
        accident_report = reduce(combine_collision_reports, report_involved_ps)
        list_accidents.append(accident_report)
    coll_reports = list_accidents

    # add info to node graph about who is at fault
    for player in Gcoll.nodes:
        # sort by time
        coll_reports.sort(key=attrgetter("at_time"))
        is_p_at_fault = next(r.players[player].at_fault for r in coll_reports if player in r.players)
        nx.set_node_attributes(Gcoll, {player: {"at_fault": is_p_at_fault}})

    return coll_reports, Gcoll


def get_collision_reports(sim_context: SimContext, skip_collision_viz: bool = False) -> Report:
    """
    Generate detailed collision report
    :param sim_context:
    :param skip_collision_viz: If True speeds up skipping the visualisation of the single collision instants
    :return:
    """
    r = Report("AccidentsReport")
    accidents_report, coll_graph = investigate_collision_report(sim_context.collision_reports)

    fig_graph = r.figure(cols=1)
    with fig_graph.plot(nid="CollisionGraph", mime=MIME_PNG) as _:
        node_colors = ["darkred" if _["at_fault"] else "forestgreen" for _ in coll_graph.nodes.values()]
        edgecolors = [
            sim_context.models[_].model_geometry.color if _ in sim_context.models.keys() else "black"
            for _ in coll_graph.nodes
        ]
        nx.draw(coll_graph, with_labels=True, node_color=node_colors, edgecolors=edgecolors)
    plt.close()

    for i, acc_report in enumerate(accidents_report):
        acc_id = "-".join(list(acc_report.players.keys()))
        r.subsection(f"Accident-{acc_id}")
        # r.text(f"Accident-{acc_id}-report", text=remove_escapes(debug_print(acc_report)))
        # damage_metrics = compute_damage_metrics(
        #     coll_report=acc_report, sim_log=sim_context.log, sim_models=sim_context.models
        # )
        # r.text(
        #     f"Accident-{acc_id}-damages",
        #     text=remove_escapes(debug_print(damage_metrics)),
        # )
        if not skip_collision_viz:
            collisions_wrt_accident = [
                creport for creport in sim_context.collision_reports if set(acc_report.players) == set(creport.players)
            ]
            coll_fig = r.figure(cols=5)
            for j, coll_report in enumerate(collisions_wrt_accident):
                with coll_fig.plot(f"Collision-{i}-{j}", MIME_PNG) as _:
                    plot_collision(coll_report, sim_log=sim_context.log)
                plt.close()

    return r
