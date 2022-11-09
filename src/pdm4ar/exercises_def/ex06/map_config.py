from pdm4ar.exercises_def.ex06.structures import Point

################################################################
###################### Default Map Config ######################
################################################################

DEFAULT_MAP_CONFIG = [
    {
        "obstacles": [
            {
                "type": "polygon",
                "params": {
                    "center": Point(150, 150),
                    "avg_radius": 15,
                    "irregularity": 0,
                    "spikiness": 0,
                    "num_vertices": 4,
                },
            },
            {
                "type": "triangle",
                "params": {
                    "center": Point(50, 100),
                    "avg_radius": 15,
                    "irregularity": 0,
                    "spikiness": 0,
                },
            },
            {
                "type": "polygon",
                "params": {
                    "center": Point(45, 45),
                    "avg_radius": 15,
                    "irregularity": 0,
                    "spikiness": 0,
                    "num_vertices": 4,
                },
            },
        ],
        "path": [
            (0, 0),
            (30, 45),
            (60, 60),
            (90, 75),
            (120, 110),
            (130, 155),
        ],
    },
    {
        "obstacles": [
            {
                "type": "polygon",
                "params": {
                    "center": Point(150, 150),
                    "avg_radius": 15,
                    "irregularity": 0.2,
                    "spikiness": 0,
                    "num_vertices": 4,
                },
            },
            {
                "type": "triangle",
                "params": {
                    "center": Point(50, 100),
                    "avg_radius": 15,
                    "irregularity": 0.2,
                    "spikiness": 0,
                },
            },
            {
                "type": "circle",
                "params": {
                    "center": Point(45, 45),
                    "max_dist": 3,
                    "min_radius": 7,
                    "max_radius": 13,
                },
            },
        ],
        "path": [
            (0, 0),
            (30, 45),
            (60, 60),
            (90, 75),
            (120, 110),
            (130, 155),
        ],
    },
    {
        "obstacles": [
            {
                "type": "polygon",
                "params": {
                    "center": Point(125, 125),
                    "avg_radius": 15,
                    "irregularity": 0.2,
                    "spikiness": 0,
                    "num_vertices": 6,
                },
            },
            {
                "type": "circle",
                "params": {
                    "center": Point(25, 35),
                    "max_dist": 15,
                    "min_radius": 5,
                    "max_radius": 17,
                },
            },
            {
                "type": "circle",
                "params": {
                    "center": Point(80, 45),
                    "max_dist": 5,
                    "min_radius": 5,
                    "max_radius": 17,
                },
            },
            {
                "type": "polygon",
                "params": {
                    "center": Point(0, 100),
                    "avg_radius": 15,
                    "irregularity": 0.2,
                    "spikiness": 0,
                    "num_vertices": 6,
                },
            },
            {
                "type": "triangle",
                "params": {
                    "center": Point(100, 0),
                    "avg_radius": 15,
                    "irregularity": 0.2,
                    "spikiness": 0,
                },
            },
        ],
        "path": [
            (0, 0),
            (10, 15),
            (35, 45),
            (60, 55),
            (80, 80),
            (100, 95),
            (120, 115),
            (130, 155),
        ],
    },
    {
        "obstacles": [
            {
                "type": "polygon",
                "params": {
                    "center": Point(125, 125),
                    "avg_radius": 15,
                    "irregularity": 0.6,
                    "spikiness": 0,
                    "num_vertices": 8,
                },
            },
            {
                "type": "polygon",
                "params": {
                    "center": Point(25, 35),
                    "avg_radius": 15,
                    "irregularity": 0.6,
                    "spikiness": 0,
                    "num_vertices": 7,
                },
            },
            {
                "type": "polygon",
                "params": {
                    "center": Point(80, 45),
                    "avg_radius": 15,
                    "irregularity": 0.6,
                    "spikiness": 0,
                    "num_vertices": 6,
                },
            },
            {
                "type": "polygon",
                "params": {
                    "center": Point(0, 100),
                    "avg_radius": 15,
                    "irregularity": 0.6,
                    "spikiness": 0,
                    "num_vertices": 6,
                },
            },
            {
                "type": "polygon",
                "params": {
                    "center": Point(100, 0),
                    "avg_radius": 15,
                    "irregularity": 0.6,
                    "spikiness": 0,
                    "num_vertices": 6,
                },
            },
        ],
        "path": [
            (0, 0),
            (10, 15),
            (35, 45),
            (60, 55),
            (80, 80),
            (100, 95),
            (120, 115),
            (130, 155),
        ],
    },
    {
        "obstacles": [
            {
                "type": "polygon",
                "params": {
                    "center": Point(125, 125),
                    "avg_radius": 15,
                    "irregularity": 0.8,
                    "spikiness": 0,
                    "num_vertices": 8,
                },
            },
            {
                "type": "polygon",
                "params": {
                    "center": Point(25, 35),
                    "avg_radius": 15,
                    "irregularity": 0.8,
                    "spikiness": 0,
                    "num_vertices": 7,
                },
            },
            {
                "type": "triangle",
                "params": {
                    "center": Point(80, 45),
                    "avg_radius": 15,
                    "irregularity": 0.8,
                    "spikiness": 0,
                },
            },
            {
                "type": "polygon",
                "params": {
                    "center": Point(0, 100),
                    "avg_radius": 15,
                    "irregularity": 0.8,
                    "spikiness": 0,
                    "num_vertices": 9,
                },
            },
            {
                "type": "polygon",
                "params": {
                    "center": Point(100, 0),
                    "avg_radius": 15,
                    "irregularity": 0.8,
                    "spikiness": 0,
                    "num_vertices": 7,
                },
            },
            {
                "type": "triangle",
                "params": {
                    "center": Point(100, 80),
                    "avg_radius": 15,
                    "irregularity": 0.8,
                    "spikiness": 0,
                },
            },
            {
                "type": "circle",
                "params": {
                    "center": Point(60, 100),
                    "max_dist": 5,
                    "min_radius": 5,
                    "max_radius": 10,
                },
            },
        ],
        "path": [
            (0, 0),
            (10, 15),
            (35, 45),
            (60, 55),
            (80, 80),
            (100, 95),
            (120, 115),
            (130, 155),
        ],
    },
]

################################################################
###################### R-Tree Map Config #######################
################################################################

# Shorter Maps But More Obstacles

R_TREE_MAP_CONFIG = DEFAULT_MAP_CONFIG.copy()

################################################################

R_TREE_MAP_CONFIG[0]["path"] = [
    (0, 0),
    (25, 80),
    (150, 120),
]
R_TREE_MAP_CONFIG[0]["obstacles"].append(
    {
        "type": "circle",
        "params": {
            "center": Point(100, 30),
            "max_dist": 2,
            "min_radius": 5,
            "max_radius": 10,
        },
    }
)
R_TREE_MAP_CONFIG[0]["obstacles"].append(
    {
        "type": "triangle",
        "params": {
            "center": Point(150, 30),
            "avg_radius": 15,
            "irregularity": 0.2,
            "spikiness": 0,
        },
    }
)

################################################################

R_TREE_MAP_CONFIG[1]["path"] = [
    (0, 0),
    (100, 20),
    (150, 80),
    (130, 155),
]

R_TREE_MAP_CONFIG[1]["obstacles"].append(
    {
        "type": "polygon",
        "params": {
            "center": Point(100, 60),
            "avg_radius": 15,
            "irregularity": 0.2,
            "spikiness": 0,
            "num_vertices": 5,
        },
    }
)
R_TREE_MAP_CONFIG[1]["obstacles"].append(
    {
        "type": "triangle",
        "params": {
            "center": Point(150, 30),
            "avg_radius": 15,
            "irregularity": 0.4,
            "spikiness": 0,
        },
    }
)

################################################################

R_TREE_MAP_CONFIG[2]["path"] = [
    (0, 0),
    (35, 45),
    (80, 20),
    (120, 70),
    (130, 155),
]
R_TREE_MAP_CONFIG[2]["obstacles"].append(
    {
        "type": "polygon",
        "params": {
            "center": Point(60, 130),
            "avg_radius": 15,
            "irregularity": 0.4,
            "spikiness": 0,
            "num_vertices": 7,
        },
    }
)
R_TREE_MAP_CONFIG[2]["obstacles"].append(
    {
        "type": "circle",
        "params": {
            "center": Point(40, 0),
            "max_dist": 2,
            "min_radius": 5,
            "max_radius": 10,
        },
    }
)
R_TREE_MAP_CONFIG[2]["obstacles"].append(
    {
        "type": "triangle",
        "params": {
            "center": Point(140, 50),
            "avg_radius": 15,
            "irregularity": 0.6,
            "spikiness": 0,
        },
    }
)

################################################################

R_TREE_MAP_CONFIG[3]["path"] = [
    (0, 0),
    (35, 45),
    (50, 100),
    (80, 140),
    (120, 70),
    (60, 100),
    (100, 140),
]

R_TREE_MAP_CONFIG[3]["obstacles"].append(
    {
        "type": "polygon",
        "params": {
            "center": Point(60, 140),
            "avg_radius": 15,
            "irregularity": 0.7,
            "spikiness": 0,
            "num_vertices": 9,
        },
    }
)
R_TREE_MAP_CONFIG[3]["obstacles"].append(
    {
        "type": "polygon",
        "params": {
            "center": Point(140, 60),
            "avg_radius": 15,
            "irregularity": 0.9,
            "spikiness": 0,
            "num_vertices": 8,
        },
    }
)
R_TREE_MAP_CONFIG[3]["obstacles"].append(
    {
        "type": "polygon",
        "params": {
            "center": Point(0, 150),
            "avg_radius": 15,
            "irregularity": 0.9,
            "spikiness": 0,
            "num_vertices": 9,
        },
    }
)
################################################################
R_TREE_MAP_CONFIG[4]["obstacles"].append(
    {
        "type": "circle",
        "params": {
            "center": Point(40, 140),
            "max_dist": 6,
            "min_radius": 5,
            "max_radius": 10,
        },
    }
)
R_TREE_MAP_CONFIG[4]["obstacles"].append(
    {
        "type": "circle",
        "params": {
            "center": Point(150, 20),
            "max_dist": 6,
            "min_radius": 7,
            "max_radius": 12,
        },
    }
)
R_TREE_MAP_CONFIG[4]["obstacles"].append(
    {
        "type": "circle",
        "params": {
            "center": Point(0, 150),
            "max_dist": 6,
            "min_radius": 7,
            "max_radius": 12,
        },
    }
)
################################################################
################ Safety Certificates Map Config ################
################################################################

# Longer Paths but Less Obstacles

SAFETY_CERTIFICATES_MAP_CONFIG = DEFAULT_MAP_CONFIG.copy()

################################################################
SAFETY_CERTIFICATES_MAP_CONFIG[0]["path"] = [
    (0, 0),
    (25, 20),
    (30, 45),
    (60, 60),
    (65, 80),
    (90, 75),
    (125, 90),
    (120, 110),
    (100, 140),
    (130, 155),
]
################################################################
SAFETY_CERTIFICATES_MAP_CONFIG[1]["path"] = [
    (0, 0),
    (25, 20),
    (30, 45),
    (60, 60),
    (65, 80),
    (90, 75),
    (125, 90),
    (120, 110),
    (100, 140),
    (130, 155),
]
################################################################
SAFETY_CERTIFICATES_MAP_CONFIG[2]["path"] = [
    (0, 0),
    (10, 15),
    (35, 45),
    (60, 55),
    (65, 70),
    (80, 80),
    (100, 95),
    (120, 115),
    (115, 140),
    (130, 155),
    (160, 160),
]
del SAFETY_CERTIFICATES_MAP_CONFIG[2]["obstacles"][-1]

################################################################
################################################################
################################################################

EXERCISE_MAP_CONFIGS = {
    8: DEFAULT_MAP_CONFIG,
    9: DEFAULT_MAP_CONFIG,
    10: R_TREE_MAP_CONFIG,
    11: DEFAULT_MAP_CONFIG,
    12: SAFETY_CERTIFICATES_MAP_CONFIG,
}
