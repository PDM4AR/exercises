from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Any, Literal, Union
import numpy as np

from pdm4ar.exercises_def.ex07.structures import Island, ProblemVoyage, ProblemVoyage1, \
        ProblemSolutions, SlackCosts, SlackViolations, SolutionViolations, aViolations


@dataclass(frozen=True)
class MilpCase(Enum):
    voyage1 = "voyage1"
    voyage1_1 = "voyage1_1"
    voyage1_2 = "voyage1_2"
    voyage1_3 = "voyage1_3"
    voyage1_4 = "voyage1_4"
    voyage2 = "voyage2"
    testvoyage1 = "testvoyage1"
    testvoyage2 = "testvoyage2"
    easy = "easy"
    medium = "medium"
    hard = "hard"
    feasible = "feasible"
    unfeasible = "unfeasible"
    unbounded = "unbounded"
    error = "error"

    def __eq__(self, other) -> bool:
        return self.value == other.value

    def __contains__(self, other) -> bool:
        return self.value in other.value or other.value in self.value

    @classmethod
    def get_test_milp_cases_types(cls):
        return [cls.voyage1]

    @classmethod
    def get_test_milp_cases_difficulties(cls):
        return [cls.easy, cls.medium, cls.hard]


def island_generator(id, level, x_range, y_range, departure_range, arrival_range, time_compass_range, crew_range):
    x = (x_range[1]-x_range[0])*np.random.random_sample() + x_range[0]
    y = (y_range[1]-y_range[0])*np.random.random_sample() + y_range[0]
    departure = (departure_range[1]-departure_range[0])*np.random.random_sample() + departure_range[0]
    arrival = (arrival_range[1]-arrival_range[0])*np.random.random_sample() + arrival_range[0]
    time_compass = np.random.randint(time_compass_range[0], time_compass_range[1]+1)
    crew = np.random.randint(crew_range[0], crew_range[1]+1)

    return Island(id,level,x,y,departure,arrival,time_compass,crew)

def island_is_overlapping(generated_island, islands, viz_radius=2):
    for island in islands:
        if (generated_island.x - island.x)**2 + (generated_island.y - island.y)**2 < (2*viz_radius)**2:
            return False
    return True

def milp_generator(milp_type: Literal, milp_difficulty: Literal, milp_seed: int
    ) -> Tuple[ProblemVoyage1, None, SlackViolations, SlackCosts]:
    
    if milp_type not in MilpCase.get_test_milp_cases_types():
        raise NotImplementedError(milp_type)

    islands = []

    if milp_type in MilpCase.voyage1:
        np.random.seed(milp_seed)
        if milp_difficulty == MilpCase.easy:
            start_crew = np.random.randint(25,50)
            n_time_steps = np.random.randint(3,8)
            n_islands_step = np.random.randint(3,8)
            x_range = [np.random.randint(10,31), np.random.randint(35,56)]
            y_range = [np.random.randint(35,60), np.random.randint(80,110)]
            departure_range = [np.random.randint(6,9), np.random.randint(10,13)]
            arrival_range = [np.random.randint(16,19), np.random.randint(20,23)]
            time_compass_range = [np.random.randint(1,4), np.random.randint(5,8)]
            crew_range = [np.random.randint(-8,-1), np.random.randint(5,8)]
            x_offset_levels = np.random.randint(x_range[1]-x_range[0]+5, x_range[1]-x_range[0]+20)
            y_offset_levels = [0,0]

            # TODO magic numbers here should be random too
            max_distance_individual_sail = 0.75*(y_range[1]-y_range[0]) + 0.75*(x_range[1]-x_range[0]) + x_offset_levels
            min_theoretical_sail = arrival_range[0] - departure_range[1]
            max_theoretical_sail = arrival_range[1] - departure_range[0]
            max_duration_individual_sail = min_theoretical_sail + 0.7*(max_theoretical_sail-min_theoretical_sail)
            min_fix_ship_individual_island = time_compass_range[0]+1
            mean_crew_change = crew_range[0] + (crew_range[1] - crew_range[0])
            if mean_crew_change > 0:
                max_crew = start_crew + int(1.1*n_time_steps*(mean_crew_change))
                min_crew = max(round(0.8*start_crew),1)
            elif mean_crew_change < 0:
                max_crew = round(1.2*start_crew)
                min_crew = max(start_crew - int(1.1*n_time_steps*(mean_crew_change)),1)
            else:
                max_crew = round(1.2*start_crew)
                min_crew = max(round(0.8*start_crew),1)

            # TODO these values should be more or less proportional to the value of the constraints and costs
            slack_violations = aViolations(0,0,0,0,-1,-5,5,1.5,10.5)
            slack_violations = SlackViolations(slack_violations, slack_violations, slack_violations, slack_violations, slack_violations)
            slack_costs = SlackCosts(2,7,6.5,60,1)

        elif milp_difficulty == MilpCase.medium:
            start_crew = np.random.randint(40,60)
            n_time_steps = np.random.randint(8,16)
            n_islands_step = np.random.randint(8,16)
            x_range = [np.random.randint(10,41), np.random.randint(80,120)]
            y_range = [np.random.randint(-45,10), np.random.randint(110,150)]
            departure_range = [np.random.randint(6,8), np.random.randint(9,12)]
            arrival_range = [np.random.randint(15,17), np.random.randint(17,21)]
            time_compass_range = [np.random.randint(1,4), np.random.randint(5,8)]
            crew_range = [np.random.randint(-12,-5), np.random.randint(3,20)]
            x_offset_levels = np.random.randint(x_range[1]-x_range[0]+0, x_range[1]-x_range[0]+10)
            y_offset_levels = [-10,10]

            # TODO magic numbers here should be random too
            max_distance_individual_sail = 0.5*(y_range[1]-y_range[0]) + 0.5*(x_range[1]-x_range[0]) + x_offset_levels
            min_theoretical_sail = arrival_range[0] - departure_range[1]
            max_theoretical_sail = arrival_range[1] - departure_range[0]
            max_duration_individual_sail = min_theoretical_sail + 0.5*(max_theoretical_sail-min_theoretical_sail)
            min_fix_ship_individual_island = time_compass_range[0]+1
            mean_crew_change = (crew_range[1] - crew_range[0])/2
            if mean_crew_change > 0:
                max_crew = start_crew + int(0.8*n_time_steps*(mean_crew_change))
                min_crew = max(round(0.9*start_crew),1)
            elif mean_crew_change < 0:
                max_crew = round(1.1*start_crew)
                min_crew = max(start_crew - int(0.8*n_time_steps*(mean_crew_change)),1)
            else:
                max_crew = round(1.1*start_crew)
                min_crew = max(round(0.9*start_crew),1)

            # TODO these values should be more or less proportional to the value of the constraints and costs
            slack_violations = aViolations(0,0,0,0,-1,-2,2,1.5,7.5)
            slack_violations = SlackViolations(slack_violations, slack_violations, slack_violations, slack_violations, slack_violations)
            slack_costs = SlackCosts(1,2,4,30,0.5)

        elif milp_difficulty == MilpCase.hard:
            start_crew = np.random.randint(120,200)
            n_time_steps = np.random.randint(16, 24)
            n_islands_step = np.random.randint(12,20)
            x_range = [np.random.randint(10,41), np.random.randint(80,120)]
            y_range = [np.random.randint(-45,30), np.random.randint(110,150)]
            departure_range = [np.random.randint(6,8), np.random.randint(9,12)]
            arrival_range = [np.random.randint(15,17), np.random.randint(17,21)]
            time_compass_range = [np.random.randint(1,4), np.random.randint(5,8)]
            crew_range = [np.random.randint(-15,0), np.random.randint(3,20)]
            x_offset_levels = np.random.randint(x_range[1]-x_range[0]-50, x_range[1]-x_range[0]+10)
            y_offset_levels = [-30, 30]

            # TODO magic numbers here should be random too
            max_distance_individual_sail = 0.5*(y_range[1]-y_range[0]) + 0.5*(x_range[1]-x_range[0]) + x_offset_levels
            min_theoretical_sail = arrival_range[0] - departure_range[1]
            max_theoretical_sail = arrival_range[1] - departure_range[0]
            max_duration_individual_sail = min_theoretical_sail + 0.5*(max_theoretical_sail-min_theoretical_sail)
            min_fix_ship_individual_island = time_compass_range[0]+1
            mean_crew_change = (crew_range[1] - crew_range[0])/2
            if mean_crew_change > 0:
                max_crew = start_crew + int(0.3*n_time_steps*(mean_crew_change))
                min_crew = max(round(0.95*start_crew),1)
            elif mean_crew_change < 0:
                max_crew = round(1.05*start_crew)
                min_crew = max(start_crew - int(0.3*n_time_steps*(mean_crew_change)),1)
            else:
                max_crew = round(1.05*start_crew)
                min_crew = max(round(0.95*start_crew),1)

            # TODO these values should be more or less proportional to the value of the constraints and costs
            slack_violations = aViolations(0,0,0,0,-0,-0,0,0,0)
            slack_violations = SlackViolations(slack_violations, slack_violations, slack_violations, slack_violations, slack_violations)
            slack_costs = SlackCosts(0,0,0,0,0)

        else:
            raise ValueError(milp_difficulty)

    elif milp_type in MilpCase.voyage2:
        raise NotImplementedError(milp_type)

    else:
        raise ValueError(milp_type)


    max_trials_samples = 200
    island_id = 0
    for k in range(n_time_steps):
        x_range = [i+x_offset_levels for i in x_range]
        y_offset = np.random.randint(y_offset_levels[0], y_offset_levels[1]+1)
        y_range = [i+y_offset for i in y_range]
        for _ in range(n_islands_step):
            pos_ok = False
            n_trials_samples = 0
            while pos_ok is not True:
                if n_trials_samples >= max_trials_samples:
                    raise ValueError(f"Exceeded the amount of trials ({n_trials_samples}) to generate a non overlapping island."
                                    f" Try to increase the x_range ({x_range}) or y_range ({y_range}), or run again with a different seed.")
                generated_island = island_generator(island_id, k, x_range,y_range,departure_range,arrival_range,time_compass_range,crew_range)
                pos_ok = island_is_overlapping(generated_island, islands)
                n_trials_samples += 1
            islands.append(generated_island)
            island_id += 1
    islands = tuple(islands)

    problem = ProblemVoyage1(start_crew, islands, min_fix_ship_individual_island, 
                        min_crew, max_crew, max_duration_individual_sail, max_distance_individual_sail)

    return problem, None, slack_violations, slack_costs


def database_reader(database: Dict, num_ex: int
    ) -> Tuple[ProblemVoyage1, ProblemSolutions, SlackViolations, SlackCosts]:

    problem = database[num_ex]["problem"]
    solution = database[num_ex]["solution"]
    slack_violations = database[num_ex]["slack_violations"]
    slack_costs = database[num_ex]["slack_costs"]

    return problem, solution, slack_violations, slack_costs
