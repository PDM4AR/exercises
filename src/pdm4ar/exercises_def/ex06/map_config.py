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
################################################################
################################################################


EXERCISE_MAP_CONFIGS = {
    8: DEFAULT_MAP_CONFIG,
    9: DEFAULT_MAP_CONFIG,
    10: DEFAULT_MAP_CONFIG,
    11: DEFAULT_MAP_CONFIG,
    12: DEFAULT_MAP_CONFIG,
}
