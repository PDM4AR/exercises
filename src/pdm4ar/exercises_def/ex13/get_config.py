from typing import List


def get_config() -> List[str]:
    "Choose manually which .yaml file to load and test."

    configs = [
        # "config_local.yaml",
        # public tests
        "config_planet.yaml",
        "config_satellites.yaml",
        "config_satellites_diff.yaml",
        "config_asteroid.yaml",
        # private tests
        "config_planet_2.yaml",
        "config_satellites_2.yaml",
        "config_satellites_diff_2.yaml",
        "config_asteroid_2.yaml",  # to be added
        # not defined yet
        "config_asteroids_planets.yaml",
        "config_asteroids_planets_2.yaml",
        "config_asteroids_planets_satellites.yaml",
    ]

    return configs
