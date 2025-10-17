import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dg_commons import DgSampledSequence
import os
from pdm4ar.exercises_def.structures import out_dir


def plot_traj(computed: DgSampledSequence, actual: list = None):
    """
    Example of simple plotting function to help you debug your code.
    Feel free to modify it or create your own plotting functions.
    Note that the plot is overwritten at each call which means that
    only the plot of the last simulation is saved.
    """

    timestamps = list(computed._timestamps)  # sequence.get_sampling_points()
    values = list(computed._values)

    df = pd.DataFrame(values)

    plt.plot(df["x"], df["y"], label="Computed Trajectory")
    for i in range(len(df)):
        plt.arrow(
            df["x"][i],
            df["y"][i],
            np.cos(df["psi"][i]),
            np.sin(df["psi"][i]),
            head_width=0.1,
            head_length=0.1,
            fc="k",
            ec="k",
        )

    if actual is not None:
        actual_positions = np.array([[state.x, state.y] for state in actual])
        actual_orientations = np.array([state.psi for state in actual])
        plt.scatter(actual_positions[:, 0], actual_positions[:, 1], label="Actual Trajectory")
        for i in range(len(actual_positions)):
            plt.arrow(
                actual_positions[i, 0],
                actual_positions[i, 1],
                np.cos(actual_orientations[i]),
                np.sin(actual_orientations[i]),
                head_width=0.1,
                head_length=0.05,
                fc="r",
                ec="r",
            )

    plt.grid(True)
    plt.legend()
    # Save the figure to a file
    output_dir = os.path.join(out_dir("13"), "index.html_resources")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "final_traj.png"), dpi=300)
    plt.close()
