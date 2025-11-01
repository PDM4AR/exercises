from typing import List


def get_config() -> List[str]:
    "Choose manually which .yaml file to load and test."

    configs = [
        # public tests
        "config_1_public.yaml",
        "config_2_public.yaml",
        "config_3_public.yaml",
        # "config_local.yaml",
    ]

    return configs
