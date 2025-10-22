from typing import List


def get_config() -> List[str]:
    "Choose manually which .yaml file to load and test."

    configs = [
        # public tests
        "config_planet_target.yaml",
        "config_planet.yaml",
        "config_asteroid.yaml",
        # "config_local.yaml",
    ]

    return configs
