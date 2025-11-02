from typing import List


def get_config() -> List[str]:
    "Choose manually which .yaml file to load and test."

    configs = [
        # public tests
        "scenario1.yaml",
        "scenario2.yaml",
        "scenario3.yaml",
        # "scenario_custom.yaml",
    ]

    return configs
