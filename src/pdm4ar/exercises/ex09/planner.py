import numpy as np
from dg_commons.sim.models.rocket import RocketCommands, RocketState
from dg_commons.seq import DgSampledSequence


class RocketPlanner:

    def compute_trajectory(self, ) -> (
            tuple[DgSampledSequence[RocketCommands], DgSampledSequence[RocketState]]
    ):
        # TODO all the planning you want

        mycmds, mystates = self._extract_seq_from_array()
        return mycmds, mystates

    @staticmethod
    def _extract_seq_from_array() -> (
            tuple[DgSampledSequence[RocketCommands], DgSampledSequence[RocketState]]
    ):
        ts = (0, 1, 2, 3, 4)
        # in case my planner returns 3 numpy arrays
        F_l = np.array([0, 1, 2, 3, 4])
        F_r = np.array([0, 1, 2, 3, 4])
        dphi = np.array([0, 0, 0, 0, 0])
        cmds_list = [RocketCommands(l, r, dp) for l, r, dp in zip(F_l, F_r, dphi)]
        mycmds = DgSampledSequence[RocketCommands](timestamps=ts, values=cmds_list)

        # in case my state trajectory is in a 2d array
        npstates = np.random.rand(len(ts), 6)
        states = [RocketState(*v) for v in npstates]
        mystates = DgSampledSequence[RocketState](timestamps=ts, values=states)
        return mycmds, mystates
