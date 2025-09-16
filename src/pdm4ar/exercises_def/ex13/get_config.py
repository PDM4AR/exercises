from typing import List

def get_config() -> List[str]:
    "Choose manually which .yaml file to load and test."

    configs = ["config_planet.yaml", 
               "config_satellites.yaml", 
               "config_satellites_diff.yaml",
               #"config_local.yaml"
               ]
    
    return configs