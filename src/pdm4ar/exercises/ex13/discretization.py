import numpy as np
import sympy as spy
from numpy.typing import NDArray
from scipy.integrate import odeint

from pdm4ar.exercises.ex13.satellite import SatelliteDyn


class DiscretizationMethod:
    K: int  # number of discretization points
    N_sub: int  # number of substeps to approximate the ode
    range_t: tuple  # range of discretization points

    satellite: SatelliteDyn

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, satellite: SatelliteDyn, K: int, N_sub: int):
        # number of discretization points
        self.K = K

        # tuple containing N_sub ts between [0, 1/(K-1)] used to approximate the ODE
        self.N_sub = N_sub
        self.range_t = tuple(np.linspace(0, 1.0 / (self.K - 1), self.N_sub))

        self.f, self.A, self.B, self.F = satellite.get_dynamics()

        # number of states, inputs and parameters
        self.n_x = satellite.n_x
        self.n_u = satellite.n_u
        self.n_p = satellite.n_p

    def calculate_discretization(self, X: NDArray, U: NDArray, p: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Calculate discretization for given states, inputs and parameter matrices.

        :param X: Matrix of states at all time steps
        :param U: Matrix of inputs at all time steps
        :param p: Vector of parameters
        :return: The 2D flattened matrices for the discrete-time linear dynamics, required by the optimization framework
        """
        pass

    def integrate_nonlinear_piecewise(self, X_l: NDArray, U: NDArray, p: NDArray) -> NDArray:
        """
        Piecewise integration of the continuous-time nonlinear dynamics, simulating the system response to the inputs U that
        produced the discrete-time linear state sequence X_l, starting from each state. Helpful to verify accuracy of linearization.

        :param X_l: Linear state evolution matrix
        :param U: Linear input evolution matrix
        :param p: Vector of parameters
        :return: The piecewise integrated dynamics
        """
        pass

    def integrate_nonlinear_full(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:
        """
        Simulate nonlinear behavior given an initial state and an input over time, integrating the continuous-time nonlinear dynamics
        starting from the initial state x0 under the effect of the input U.

        :param x0: Initial state
        :param U: Linear input evolution matrix
        :param p: Vector of parameters
        :return: The full integrated dynamics
        """
        pass

    def integrate_nonlinear_full_dense(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:
        """
        Simulate nonlinear behavior given an initial state and an input over time, integrating the continuous-time nonlinear dynamics
        starting from the initial state x0 under the effect of the input U.

        :param x0: Initial state
        :param U: Linear input evolution matrix
        :param p: Vector of parameters
        :return: The full integrated dynamics (with added points linked to N_subs)
        """
        pass

    def check_dynamics(self) -> bool:
        """
        Check if the implemented dynamics returns the correct state trajectory.
        """
        pass


class ZeroOrderHold(DiscretizationMethod):
    A_bar: NDArray
    B_bar: NDArray
    F_bar: NDArray
    r_bar: NDArray

    x_ind: slice
    A_bar_ind: slice
    B_bar_ind: slice
    F_bar_ind: slice
    r_bar_ind: slice

    P0: NDArray

    def __init__(self, satellite: SatelliteDyn, K: int, N_sub: int):

        super().__init__(satellite, K, N_sub)

        # x+ = A_bar(x*(k))x(k) + B_bar(x*(k))u(k) + F_bar(x*(k))p + r_bar(k)
        self.A_bar = np.zeros([self.n_x * self.n_x, self.K - 1])
        self.B_bar = np.zeros([self.n_x * self.n_u, self.K - 1])
        self.F_bar = np.zeros([self.n_x * self.n_p, self.K - 1])
        self.r_bar = np.zeros([self.n_x, self.K - 1])

        # vector indices for flat matrices
        x_end = self.n_x
        A_bar_end = self.n_x * (1 + self.n_x)
        B_bar_end = self.n_x * (1 + self.n_x + self.n_u)
        F_bar_end = self.n_x * (1 + self.n_x + self.n_u + self.n_p)
        r_bar_end = self.n_x * (1 + self.n_x + self.n_u + self.n_p + 1)
        self.x_ind = slice(0, x_end)
        self.A_bar_ind = slice(x_end, A_bar_end)
        self.B_bar_ind = slice(A_bar_end, B_bar_end)
        self.F_bar_ind = slice(B_bar_end, F_bar_end)
        self.r_bar_ind = slice(F_bar_end, r_bar_end)

        # integration initial condition
        self.P0 = np.zeros((self.n_x * (1 + self.n_x + self.n_u + self.n_p + 1),))
        self.P0[self.A_bar_ind] = np.eye(self.n_x).reshape(-1)

    def calculate_discretization(self, X: NDArray, U: NDArray, p: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:

        for k in range(self.K - 1):
            self.P0[self.x_ind] = X[:, k]
            P = np.array(odeint(self._ode_dPdt, self.P0, self.range_t, args=(U[:, k], p))[-1, :])

            Phi = P[self.A_bar_ind].reshape((self.n_x, self.n_x))
            self.A_bar[:, k] = Phi.flatten(order="F")
            self.B_bar[:, k] = (Phi @ P[self.B_bar_ind].reshape((self.n_x, self.n_u))).flatten(order="F")
            self.F_bar[:, k] = (Phi @ P[self.F_bar_ind]).reshape((self.n_x, self.n_p)).flatten(order="F")
            self.r_bar[:, k] = Phi @ P[self.r_bar_ind]

        return self.A_bar, self.B_bar, self.F_bar, self.r_bar

    def _ode_dPdt(self, P: NDArray, t: float, u: NDArray, p: NDArray) -> NDArray:

        x = P[self.x_ind]

        # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
        # and pre-multiplying with \Phi_A(\tau_{k+1},\tau_k) after integration
        Phi_A_xi = np.linalg.inv(P[self.A_bar_ind].reshape((self.n_x, self.n_x)))

        A_subs = self.A(x, u, p)
        B_subs = self.B(x, u, p)
        F_subs = self.F(x, u, p)
        f_subs = self.f(x, u, p).reshape(-1)

        dPdt = np.zeros_like(P)
        dPdt[self.x_ind] = f_subs.transpose()
        dPdt[self.A_bar_ind] = (A_subs @ P[self.A_bar_ind].reshape((self.n_x, self.n_x))).reshape(-1)
        dPdt[self.B_bar_ind] = (Phi_A_xi @ B_subs).reshape(-1)
        dPdt[self.F_bar_ind] = (Phi_A_xi @ F_subs).reshape(-1)
        r_t = f_subs - A_subs @ x - B_subs @ u - F_subs @ p
        dPdt[self.r_bar_ind] = Phi_A_xi @ r_t

        return dPdt

    def integrate_nonlinear_piecewise(self, X_l: NDArray, U: NDArray, p: NDArray) -> NDArray:

        X_nl = np.zeros_like(X_l)
        X_nl[:, 0] = X_l[:, 0]

        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(self._dxdt, X_l[:, k], self.range_t, args=(U[:, k], p))[-1, :]

        return X_nl

    def integrate_nonlinear_full(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:

        X_nl = np.zeros([x0.size, self.K])
        X_nl[:, 0] = x0

        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(self._dxdt, X_nl[:, k], self.range_t, args=(U[:, k], p))[-1, :]

        return X_nl

    def integrate_nonlinear_full_dense(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:

        X_nl = np.zeros([x0.size, U.shape[1]])
        X_nl_dense = np.zeros([x0.size, U.shape[1] + (U.shape[1] - 1) * (self.N_sub - 2)])
        X_nl[:, 0] = x0
        X_nl_dense[:, 0] = x0

        for k in range(U.shape[1] - 1):
            x = odeint(self._dxdt, X_nl[:, k], self.range_t, args=(U[:, k], p))
            X_nl[:, k + 1] = x[-1, :]
            X_nl_dense[:, k * (self.N_sub - 1) + 1 : (k + 1) * (self.N_sub - 1) + 1] = np.array(x)[1:, :].T

        return X_nl_dense

    def _dxdt(self, x: NDArray, t: float, u: NDArray, p: NDArray) -> NDArray:
        return np.squeeze(self.f(x, u, p))

    def check_dynamics(self) -> bool:

        threshold = 1e-4
        x0 = np.array([-9.0, -9.0, 0, 0.0, 0.0, 0.0, 0.0, 2.5])
        U = np.array(
            [
                [
                    -7.16632509e-12,
                    2.00000000e00,
                    2.00000000e00,
                    2.00000000e00,
                    -1.05752561e00,
                    -2.00000000e00,
                    -1.83788525e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    1.10504277e00,
                    1.59509275e00,
                    1.99999999e00,
                    2.00000000e00,
                    2.00000000e00,
                    -8.37909487e-01,
                    -1.47698868e00,
                    -8.88497023e-01,
                    1.41185025e00,
                    2.00000000e00,
                    2.00000000e00,
                    2.00000000e00,
                    2.00000000e00,
                    2.00000000e00,
                    -1.99999999e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -1.92901516e-01,
                    1.18207288e00,
                    9.99483945e-01,
                    1.56882470e00,
                    2.00000000e00,
                    2.00000000e00,
                    2.00000000e00,
                    2.00000000e00,
                    2.00000000e00,
                    -1.04638321e00,
                    -1.12215158e00,
                    1.08354082e00,
                    -2.68917508e-11,
                ],
                [
                    -5.92915767e-11,
                    7.85398163e-01,
                    7.85398163e-01,
                    -7.85398159e-01,
                    -7.85398162e-01,
                    -6.80646331e-01,
                    1.88210197e-01,
                    1.89932007e-02,
                    7.85398161e-01,
                    7.85398158e-01,
                    4.77011856e-01,
                    -6.67923030e-01,
                    -7.85398157e-01,
                    -7.85398158e-01,
                    -7.85398157e-01,
                    -7.85398145e-01,
                    -7.85397903e-01,
                    1.18105969e-01,
                    9.92256355e-02,
                    -3.30208731e-01,
                    4.93251872e-01,
                    -1.93461495e-02,
                    6.67885344e-01,
                    -3.13715210e-01,
                    2.24463679e-01,
                    -2.06359332e-01,
                    2.42088868e-01,
                    -1.99712906e-01,
                    6.88723597e-01,
                    -7.85398115e-01,
                    1.51842051e-01,
                    -3.53033477e-01,
                    7.85398160e-01,
                    7.85398160e-01,
                    7.85398154e-01,
                    -6.68006909e-01,
                    7.85045666e-01,
                    7.85398137e-01,
                    7.85398108e-01,
                    -4.02884894e-01,
                    -4.05403275e-01,
                    -6.91265766e-01,
                    -6.81356728e-01,
                    -7.85397496e-01,
                    -3.92733624e-01,
                    7.85398148e-01,
                    7.85398151e-01,
                    5.76815388e-01,
                    7.20848254e-01,
                    -1.21876164e-12,
                ],
            ]
        )
        p = np.array([16.88996566])
        expected_x = np.array(
            [
                [
                    -9.00000000e00,
                    -9.00000000e00,
                    -8.95271189e00,
                    -8.81361391e00,
                    -8.58534401e00,
                    -8.33500326e00,
                    -8.14741502e00,
                    -8.01867557e00,
                    -7.92952384e00,
                    -7.86113312e00,
                    -7.80508979e00,
                    -7.75774918e00,
                    -7.71025235e00,
                    -7.63696985e00,
                    -7.51253399e00,
                    -7.31661922e00,
                    -7.03396868e00,
                    -6.66139712e00,
                    -6.22012202e00,
                    -5.75215278e00,
                    -5.27444133e00,
                    -4.73912843e00,
                    -4.12509059e00,
                    -3.43861762e00,
                    -2.68996939e00,
                    -1.92498376e00,
                    -1.18401946e00,
                    -4.58428772e-01,
                    2.62241815e-01,
                    9.64512923e-01,
                    1.64294586e00,
                    2.28649800e00,
                    2.90734335e00,
                    3.53388022e00,
                    4.14240089e00,
                    4.66126090e00,
                    5.10520356e00,
                    5.49663414e00,
                    5.87692810e00,
                    6.23974057e00,
                    6.55807560e00,
                    6.81850914e00,
                    6.99718036e00,
                    7.08497687e00,
                    7.08335711e00,
                    6.99239137e00,
                    6.80889775e00,
                    6.59744571e00,
                    6.39678638e00,
                    6.20642989e00,
                ],
                [
                    -9.00000000e00,
                    -9.00000000e00,
                    -8.99577118e00,
                    -8.96761988e00,
                    -8.90615696e00,
                    -8.83303284e00,
                    -8.72472558e00,
                    -8.54731295e00,
                    -8.28808087e00,
                    -7.93649261e00,
                    -7.49105644e00,
                    -6.95162386e00,
                    -6.31838120e00,
                    -5.59564596e00,
                    -4.79526745e00,
                    -3.93598442e00,
                    -3.04630519e00,
                    -2.16814413e00,
                    -1.34837819e00,
                    -6.14029063e-01,
                    9.42218365e-02,
                    8.18757542e-01,
                    1.52235100e00,
                    2.16919324e00,
                    2.74722114e00,
                    3.30367494e00,
                    3.90823190e00,
                    4.56530445e00,
                    5.21025469e00,
                    5.77819194e00,
                    6.25613577e00,
                    6.64747441e00,
                    6.94822277e00,
                    7.15723181e00,
                    7.34901466e00,
                    7.54260974e00,
                    7.68277813e00,
                    7.74729658e00,
                    7.76280154e00,
                    7.79367309e00,
                    7.84796505e00,
                    7.91798014e00,
                    8.00546774e00,
                    8.11668098e00,
                    8.25749644e00,
                    8.42876901e00,
                    8.60741995e00,
                    8.78931220e00,
                    9.01920379e00,
                    9.25547464e00,
                ],
                [
                    0.00000000e00,
                    -3.11675288e-24,
                    -1.06825462e-02,
                    -8.45256443e-02,
                    -2.61779857e-01,
                    -4.69652137e-01,
                    -6.81142101e-01,
                    -9.34190175e-01,
                    -1.22781714e00,
                    -1.54964867e00,
                    -1.84613307e00,
                    -2.05911321e00,
                    -2.16314140e00,
                    -2.19602176e00,
                    -2.21905087e00,
                    -2.29569314e00,
                    -2.48550754e00,
                    -2.83980924e00,
                    -3.39233058e00,
                    -4.14498156e00,
                    -4.94077964e00,
                    -5.59906480e00,
                    -6.08545435e00,
                    -6.38937054e00,
                    -6.53273167e00,
                    -6.62899226e00,
                    -6.81814278e00,
                    -7.10440868e00,
                    -7.37018185e00,
                    -7.50280169e00,
                    -7.49739850e00,
                    -7.32972253e00,
                    -6.99434381e00,
                    -6.48693940e00,
                    -5.96197116e00,
                    -5.51465592e00,
                    -5.10226596e00,
                    -4.73957943e00,
                    -4.38963921e00,
                    -4.06108719e00,
                    -3.78851514e00,
                    -3.57077013e00,
                    -3.39809154e00,
                    -3.22568898e00,
                    -2.99705420e00,
                    -2.65855716e00,
                    -2.19148552e00,
                    -1.68889948e00,
                    -1.20014964e00,
                    -7.20835696e-01,
                ],
                [
                    0.00000000e00,
                    -9.88073344e-13,
                    2.72524008e-01,
                    5.17699660e-01,
                    7.19193228e-01,
                    4.79555720e-01,
                    8.76078398e-02,
                    -3.18042244e-01,
                    -7.58510433e-01,
                    -1.15195267e00,
                    -1.41499714e00,
                    -1.56302898e00,
                    -1.72541970e00,
                    -1.95884560e00,
                    -2.20307106e00,
                    -2.36826706e00,
                    -2.33449241e00,
                    -1.88022136e00,
                    -7.41542013e-01,
                    9.67065643e-01,
                    2.36616698e00,
                    2.60893638e00,
                    2.24371193e00,
                    1.88653797e00,
                    1.79154352e00,
                    1.49764722e00,
                    8.78591484e-01,
                    -9.36943232e-03,
                    -6.09563323e-01,
                    -7.33468457e-01,
                    -5.11602682e-01,
                    4.20900485e-02,
                    8.72857541e-01,
                    1.74314241e00,
                    1.73933425e00,
                    1.34434096e00,
                    7.38302600e-01,
                    8.76529605e-02,
                    -3.20170001e-01,
                    -4.91715062e-01,
                    -5.78108022e-01,
                    -4.98487574e-01,
                    -3.00889198e-01,
                    -9.33501974e-02,
                    6.60719813e-02,
                    1.06111581e-01,
                    -2.05211005e-03,
                    -5.18192074e-01,
                    -9.08001006e-01,
                    -7.94298580e-01,
                ],
                [
                    0.00000000e00,
                    1.00968261e-23,
                    3.93590782e-02,
                    1.79817009e-01,
                    4.00878717e-01,
                    5.06374970e-01,
                    5.98271756e-01,
                    6.19394563e-01,
                    5.05315122e-01,
                    2.02806777e-01,
                    -2.45953357e-01,
                    -6.84345036e-01,
                    -9.66266578e-01,
                    -1.07367649e00,
                    -1.09683881e00,
                    -1.17789534e00,
                    -1.46875425e00,
                    -2.01429595e00,
                    -2.52485262e00,
                    -2.20394958e00,
                    -9.35907767e-01,
                    5.74235615e-01,
                    1.55869567e00,
                    1.99277679e00,
                    2.07199626e00,
                    2.30773145e00,
                    2.66816015e00,
                    2.87490107e00,
                    2.66207857e00,
                    2.40544150e00,
                    2.23810102e00,
                    2.08338539e00,
                    1.72503855e00,
                    8.58353815e-01,
                    4.53171669e-02,
                    -6.02897714e-01,
                    -9.90012863e-01,
                    -1.09931027e00,
                    -1.06109903e00,
                    -8.66283726e-01,
                    -6.59850383e-01,
                    -4.76112976e-01,
                    -3.73654780e-01,
                    -3.72147784e-01,
                    -4.70370650e-01,
                    -6.52899106e-01,
                    -8.13711282e-01,
                    -6.46805084e-01,
                    -2.91235107e-01,
                    1.58431006e-01,
                ],
                [
                    0.00000000e00,
                    -2.52420651e-23,
                    -9.27472788e-02,
                    -3.64233077e-01,
                    -6.35718877e-01,
                    -5.86677562e-01,
                    -6.67177187e-01,
                    -7.94364397e-01,
                    -9.08594274e-01,
                    -9.27720167e-01,
                    -7.62422855e-01,
                    -4.56430939e-01,
                    -1.71207440e-01,
                    -5.01358680e-02,
                    -1.14397422e-01,
                    -3.59311060e-01,
                    -7.67036410e-01,
                    -1.30787329e00,
                    -1.89555541e00,
                    -2.46934452e00,
                    -2.14432316e00,
                    -1.68369602e00,
                    -1.13799522e00,
                    -6.43779887e-01,
                    -1.78871221e-01,
                    -3.76948338e-01,
                    -7.24956389e-01,
                    -9.32907632e-01,
                    -6.05052707e-01,
                    -1.86041925e-01,
                    2.41695187e-01,
                    7.26933810e-01,
                    1.22860100e00,
                    1.69258709e00,
                    1.38109233e00,
                    1.24477928e00,
                    1.12201830e00,
                    1.01302559e00,
                    1.02039517e00,
                    8.68922457e-01,
                    7.19795269e-01,
                    5.55611230e-01,
                    4.73457819e-01,
                    5.53630917e-01,
                    8.01919388e-01,
                    1.17519554e00,
                    1.50770351e00,
                    1.42423267e00,
                    1.42442654e00,
                    1.34162165e00,
                ],
                [
                    0.00000000e00,
                    -2.04374019e-11,
                    2.70721388e-01,
                    5.41442776e-01,
                    2.70721389e-01,
                    1.70302836e-09,
                    -2.34614144e-01,
                    -1.69739374e-01,
                    -1.63192547e-01,
                    1.07528840e-01,
                    3.78250226e-01,
                    5.42672958e-01,
                    3.12444447e-01,
                    4.17230614e-02,
                    -2.28998325e-01,
                    -4.99719710e-01,
                    -7.70441092e-01,
                    -1.04116239e00,
                    -1.00045207e00,
                    -9.66249669e-01,
                    -1.08007037e00,
                    -9.10049811e-01,
                    -9.16718297e-01,
                    -6.86502776e-01,
                    -7.94638268e-01,
                    -7.17267169e-01,
                    -7.88397823e-01,
                    -7.04951442e-01,
                    -7.73791118e-01,
                    -5.36392794e-01,
                    -8.07114165e-01,
                    -7.54775246e-01,
                    -8.76463477e-01,
                    -6.05742090e-01,
                    -3.35020703e-01,
                    -6.42993183e-02,
                    -2.94556742e-01,
                    -2.39568574e-02,
                    2.46764521e-01,
                    5.17485890e-01,
                    3.78614216e-01,
                    2.38874474e-01,
                    5.99880787e-04,
                    -2.34259134e-01,
                    -5.04980292e-01,
                    -6.40352893e-01,
                    -3.69631510e-01,
                    -9.89101263e-02,
                    9.99142021e-02,
                    3.48385677e-01,
                ],
                [
                    2.50000000e00,
                    2.50000000e00,
                    2.49310614e00,
                    2.48621227e00,
                    2.47931841e00,
                    2.48296363e00,
                    2.48985749e00,
                    2.49619256e00,
                    2.50308642e00,
                    2.50998028e00,
                    2.51687415e00,
                    2.52376801e00,
                    2.53066187e00,
                    2.53755574e00,
                    2.54444960e00,
                    2.55134346e00,
                    2.55823733e00,
                    2.56513119e00,
                    2.57202506e00,
                    2.57891892e00,
                    2.57510991e00,
                    2.56961174e00,
                    2.56271787e00,
                    2.55582401e00,
                    2.54893015e00,
                    2.55181836e00,
                    2.55690944e00,
                    2.55997203e00,
                    2.55510548e00,
                    2.54821162e00,
                    2.54131775e00,
                    2.53442389e00,
                    2.52753002e00,
                    2.52063616e00,
                    2.52753002e00,
                    2.53442389e00,
                    2.54131775e00,
                    2.54821162e00,
                    2.54887653e00,
                    2.54480201e00,
                    2.54135686e00,
                    2.53594922e00,
                    2.52905536e00,
                    2.52216150e00,
                    2.51526763e00,
                    2.50837377e00,
                    2.50147991e00,
                    2.50508672e00,
                    2.50895470e00,
                    2.50521981e00,
                ],
            ]
        )
        X = self.integrate_nonlinear_full(x0, U, p)

        return np.all(np.abs(X - expected_x) < threshold)


class FirstOrderHold(DiscretizationMethod):
    A_bar: NDArray
    B_plus_bar: NDArray
    B_minus_bar: NDArray
    F_bar: NDArray
    r_bar: NDArray

    x_ind: slice
    A_bar_ind: slice
    B_plus_bar_end: slice
    B_minus_bar_end: slice
    F_bar_ind: slice
    r_bar_ind: slice

    P0: NDArray

    def __init__(self, satellite: SatelliteDyn, K: int, N_sub: int):

        super().__init__(satellite, K, N_sub)

        # x+ = A_bar(x*(k))x(k) + B_plus_bar(x*(k))u(k+1) + B_minus_bar(x*(k))u(k) + F_bar(x*(k))p + r_bar(k)
        self.A_bar = np.zeros([self.n_x * self.n_x, self.K - 1])
        self.B_plus_bar = np.zeros([self.n_x * self.n_u, self.K - 1])
        self.B_minus_bar = np.zeros([self.n_x * self.n_u, self.K - 1])
        self.F_bar = np.zeros([self.n_x * self.n_p, self.K - 1])
        self.r_bar = np.zeros([self.n_x, self.K - 1])

        # vector indices for flat matrices
        x_end = self.n_x
        A_bar_end = self.n_x * (1 + self.n_x)
        B_plus_bar_end = self.n_x * (1 + self.n_x + self.n_u)
        B_minus_bar_end = self.n_x * (1 + self.n_x + self.n_u + self.n_u)
        F_bar_end = self.n_x * (1 + self.n_x + self.n_u + self.n_u + self.n_p)
        r_bar_end = self.n_x * (1 + self.n_x + self.n_u + self.n_u + self.n_p + 1)
        self.x_ind = slice(0, x_end)
        self.A_bar_ind = slice(x_end, A_bar_end)
        self.B_plus_bar_ind = slice(A_bar_end, B_plus_bar_end)
        self.B_minus_bar_ind = slice(B_plus_bar_end, B_minus_bar_end)
        self.F_bar_ind = slice(B_minus_bar_end, F_bar_end)
        self.r_bar_ind = slice(F_bar_end, r_bar_end)

        # integration initial condition
        self.P0 = np.zeros((self.n_x * (1 + self.n_x + self.n_u + self.n_u + self.n_p + 1),))
        self.P0[self.A_bar_ind] = np.eye(self.n_x).reshape(-1)

    def calculate_discretization(
        self, X: NDArray, U: NDArray, p: NDArray
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:

        for k in range(self.K - 1):
            self.P0[self.x_ind] = X[:, k]
            P = np.array(odeint(self._ode_dPdt, self.P0, self.range_t, args=(U[:, k], U[:, k + 1], p))[-1, :])

            # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
            # flatten matrices in column-major (Fortran) order for CVXPY
            Phi = P[self.A_bar_ind].reshape((self.n_x, self.n_x))
            self.A_bar[:, k] = Phi.flatten(order="F")
            self.B_plus_bar[:, k] = (Phi @ P[self.B_plus_bar_ind].reshape((self.n_x, self.n_u))).flatten(order="F")
            self.B_minus_bar[:, k] = (Phi @ P[self.B_minus_bar_ind].reshape((self.n_x, self.n_u))).flatten(order="F")
            self.F_bar[:, k] = (Phi @ P[self.F_bar_ind].reshape((self.n_x, self.n_p))).flatten(order="F")
            self.r_bar[:, k] = Phi @ P[self.r_bar_ind]

        return self.A_bar, self.B_plus_bar, self.B_minus_bar, self.F_bar, self.r_bar

    def _ode_dPdt(self, P: NDArray, t: float, u_t0: NDArray, u_t1: NDArray, p: NDArray) -> NDArray:

        beta = (self.K - 1) * t
        alpha = 1 - beta
        x = P[self.x_ind]
        u = alpha * u_t0 + beta * u_t1

        # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
        # and pre-multiplying with \Phi_A(\tau_{k+1},\tau_k) after integration
        Phi_A_xi = np.linalg.inv(P[self.A_bar_ind].reshape((self.n_x, self.n_x)))

        A_subs = self.A(x, u, p)
        B_subs = self.B(x, u, p)
        F_subs = self.F(x, u, p)
        f_subs = self.f(x, u, p).reshape(-1)

        dPdt = np.zeros_like(P)
        dPdt[self.x_ind] = f_subs.transpose()
        dPdt[self.A_bar_ind] = (A_subs @ P[self.A_bar_ind].reshape((self.n_x, self.n_x))).reshape(-1)
        dPdt[self.B_plus_bar_ind] = (Phi_A_xi @ B_subs).reshape(-1) * beta
        dPdt[self.B_minus_bar_ind] = (Phi_A_xi @ B_subs).reshape(-1) * alpha
        dPdt[self.F_bar_ind] = (Phi_A_xi @ F_subs).reshape(-1)

        r_t = f_subs - A_subs @ x - B_subs @ u - F_subs @ p

        dPdt[self.r_bar_ind] = Phi_A_xi @ r_t

        return dPdt

    def integrate_nonlinear_piecewise(self, X_l: NDArray, U: NDArray, p: NDArray) -> NDArray:

        X_nl = np.zeros_like(X_l)
        X_nl[:, 0] = X_l[:, 0]

        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(self._dxdt, X_l[:, k], self.range_t, args=(U[:, k], U[:, k + 1], p))[-1, :]

        return X_nl

    def integrate_nonlinear_full(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:

        X_nl = np.zeros([x0.size, U.shape[1]])
        X_nl[:, 0] = x0

        for k in range(U.shape[1] - 1):
            X_nl[:, k + 1] = odeint(self._dxdt, X_nl[:, k], self.range_t, args=(U[:, k], U[:, k + 1], p))[-1, :]

        return X_nl

    def integrate_nonlinear_full_dense(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:

        X_nl = np.zeros([x0.size, U.shape[1]])
        X_nl_dense = np.zeros([x0.size, U.shape[1] + (U.shape[1] - 1) * (self.N_sub - 2)])
        X_nl[:, 0] = x0
        X_nl_dense[:, 0] = x0

        for k in range(U.shape[1] - 1):
            x = odeint(self._dxdt, X_nl[:, k], self.range_t, args=(U[:, k], U[:, k + 1], p))
            X_nl[:, k + 1] = x[-1, :]
            X_nl_dense[:, k * (self.N_sub - 1) + 1 : (k + 1) * (self.N_sub - 1) + 1] = np.array(x)[1:, :].T

        return X_nl_dense

    def _dxdt(self, x: NDArray, t: float, u_t0: NDArray, u_t1: NDArray, p: NDArray) -> NDArray:
        u = u_t0 + (self.K - 1) * t * (u_t1 - u_t0)
        return np.squeeze(self.f(x, u, p))

    def check_dynamics(self) -> bool:

        threshold = 1e-4
        x0 = np.array([-9.0, -9.0, 0, 0.0, 0.0, 0.0, 0.0, 2.5])
        U = np.array(
            [
                [
                    -7.16632509e-12,
                    2.00000000e00,
                    2.00000000e00,
                    2.00000000e00,
                    -1.05752561e00,
                    -2.00000000e00,
                    -1.83788525e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    1.10504277e00,
                    1.59509275e00,
                    1.99999999e00,
                    2.00000000e00,
                    2.00000000e00,
                    -8.37909487e-01,
                    -1.47698868e00,
                    -8.88497023e-01,
                    1.41185025e00,
                    2.00000000e00,
                    2.00000000e00,
                    2.00000000e00,
                    2.00000000e00,
                    2.00000000e00,
                    -1.99999999e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -2.00000000e00,
                    -1.92901516e-01,
                    1.18207288e00,
                    9.99483945e-01,
                    1.56882470e00,
                    2.00000000e00,
                    2.00000000e00,
                    2.00000000e00,
                    2.00000000e00,
                    2.00000000e00,
                    -1.04638321e00,
                    -1.12215158e00,
                    1.08354082e00,
                    -2.68917508e-11,
                ],
                [
                    -5.92915767e-11,
                    7.85398163e-01,
                    7.85398163e-01,
                    -7.85398159e-01,
                    -7.85398162e-01,
                    -6.80646331e-01,
                    1.88210197e-01,
                    1.89932007e-02,
                    7.85398161e-01,
                    7.85398158e-01,
                    4.77011856e-01,
                    -6.67923030e-01,
                    -7.85398157e-01,
                    -7.85398158e-01,
                    -7.85398157e-01,
                    -7.85398145e-01,
                    -7.85397903e-01,
                    1.18105969e-01,
                    9.92256355e-02,
                    -3.30208731e-01,
                    4.93251872e-01,
                    -1.93461495e-02,
                    6.67885344e-01,
                    -3.13715210e-01,
                    2.24463679e-01,
                    -2.06359332e-01,
                    2.42088868e-01,
                    -1.99712906e-01,
                    6.88723597e-01,
                    -7.85398115e-01,
                    1.51842051e-01,
                    -3.53033477e-01,
                    7.85398160e-01,
                    7.85398160e-01,
                    7.85398154e-01,
                    -6.68006909e-01,
                    7.85045666e-01,
                    7.85398137e-01,
                    7.85398108e-01,
                    -4.02884894e-01,
                    -4.05403275e-01,
                    -6.91265766e-01,
                    -6.81356728e-01,
                    -7.85397496e-01,
                    -3.92733624e-01,
                    7.85398148e-01,
                    7.85398151e-01,
                    5.76815388e-01,
                    7.20848254e-01,
                    -1.21876164e-12,
                ],
            ]
        )
        p = np.array([16.88996566])
        expected_x = np.array(
            [
                [
                    -9.00000000e00,
                    -8.98417195e00,
                    -8.89032197e00,
                    -8.70664610e00,
                    -8.45485644e00,
                    -8.23410669e00,
                    -8.08053682e00,
                    -7.97816238e00,
                    -7.91028999e00,
                    -7.86360715e00,
                    -7.83419769e00,
                    -7.81816268e00,
                    -7.79898846e00,
                    -7.75021892e00,
                    -7.64842043e00,
                    -7.47338911e00,
                    -7.21009532e00,
                    -6.85940990e00,
                    -6.44657298e00,
                    -6.01587862e00,
                    -5.55799875e00,
                    -5.03023703e00,
                    -4.41728649e00,
                    -3.72451645e00,
                    -2.97584310e00,
                    -2.24184464e00,
                    -1.54316741e00,
                    -8.58310267e-01,
                    -1.67528608e-01,
                    5.27862362e-01,
                    1.22191862e00,
                    1.91502224e00,
                    2.62864244e00,
                    3.37490311e00,
                    4.05976523e00,
                    4.67466618e00,
                    5.25273745e00,
                    5.82856706e00,
                    6.40536207e00,
                    6.94129747e00,
                    7.42496608e00,
                    7.83751830e00,
                    8.16134374e00,
                    8.39209435e00,
                    8.52906882e00,
                    8.57426101e00,
                    8.56098288e00,
                    8.54204757e00,
                    8.50842192e00,
                    8.49995720e00,
                ],
                [
                    -9.00000000e00,
                    -8.99935953e00,
                    -8.98589923e00,
                    -8.94068339e00,
                    -8.87540527e00,
                    -8.79579725e00,
                    -8.66292376e00,
                    -8.45632504e00,
                    -8.16295415e00,
                    -7.77723237e00,
                    -7.29856416e00,
                    -6.72662391e00,
                    -6.06129394e00,
                    -5.30767157e00,
                    -4.47779015e00,
                    -3.59160940e00,
                    -2.68155881e00,
                    -1.79417393e00,
                    -9.72187493e-01,
                    -2.14896383e-01,
                    5.59543219e-01,
                    1.34820687e00,
                    2.11397878e00,
                    2.83268754e00,
                    3.50774789e00,
                    4.19915098e00,
                    4.93781166e00,
                    5.70092510e00,
                    6.41211758e00,
                    7.03487142e00,
                    7.56439418e00,
                    8.00051597e00,
                    8.34584173e00,
                    8.63785688e00,
                    8.92085319e00,
                    9.14632016e00,
                    9.28666764e00,
                    9.34934083e00,
                    9.39796938e00,
                    9.45772657e00,
                    9.51475610e00,
                    9.55862084e00,
                    9.58489800e00,
                    9.59756920e00,
                    9.59998437e00,
                    9.58203397e00,
                    9.52650903e00,
                    9.49780430e00,
                    9.50063471e00,
                    9.50000058e00,
                ],
                [
                    0.00000000e00,
                    -1.60692184e-03,
                    -3.61567368e-02,
                    -1.59337508e-01,
                    -3.54987339e-01,
                    -5.49656844e-01,
                    -7.68262615e-01,
                    -1.02973565e00,
                    -1.32799084e00,
                    -1.63024029e00,
                    -1.87665711e00,
                    -2.02209713e00,
                    -2.07320047e00,
                    -2.08307666e00,
                    -2.11503578e00,
                    -2.23115743e00,
                    -2.48739042e00,
                    -2.92791014e00,
                    -3.56939200e00,
                    -4.35814934e00,
                    -5.07822513e00,
                    -5.64183872e00,
                    -6.02423018e00,
                    -6.23550581e00,
                    -6.32435551e00,
                    -6.45091811e00,
                    -6.68081968e00,
                    -6.96017439e00,
                    -7.14985836e00,
                    -7.20112955e00,
                    -7.10337652e00,
                    -6.83941681e00,
                    -6.40395890e00,
                    -5.85659297e00,
                    -5.37125280e00,
                    -4.93769234e00,
                    -4.54651009e00,
                    -4.18715311e00,
                    -3.83275713e00,
                    -3.52008521e00,
                    -3.26274003e00,
                    -3.05892029e00,
                    -2.88054566e00,
                    -2.67429054e00,
                    -2.38352909e00,
                    -1.96867609e00,
                    -1.46456706e00,
                    -9.67923043e-01,
                    -4.75845095e-01,
                    -9.39078633e-06,
                ],
                [
                    0.00000000e00,
                    1.37542868e-01,
                    4.01217535e-01,
                    6.26821149e-01,
                    6.14813909e-01,
                    3.09090661e-01,
                    -7.71437953e-02,
                    -4.94399693e-01,
                    -9.16367602e-01,
                    -1.25805675e00,
                    -1.47185138e00,
                    -1.63238667e00,
                    -1.84917498e00,
                    -2.11052714e00,
                    -2.34321698e00,
                    -2.46013642e00,
                    -2.31350393e00,
                    -1.62858677e00,
                    -1.96016872e-01,
                    1.62447532e00,
                    2.63846199e00,
                    2.68600602e00,
                    2.38648162e00,
                    2.21211297e00,
                    2.09548659e00,
                    1.69920244e00,
                    9.81775741e-01,
                    1.95170795e-01,
                    -1.78904361e-01,
                    -9.94167759e-02,
                    3.42794840e-01,
                    1.12498788e00,
                    2.01228453e00,
                    2.27975446e00,
                    1.74441862e00,
                    9.06284280e-01,
                    -2.67722215e-03,
                    -7.32700290e-01,
                    -1.15213689e00,
                    -1.31365058e00,
                    -1.28293034e00,
                    -1.07808103e00,
                    -7.91616783e-01,
                    -4.86076618e-01,
                    -1.82038693e-01,
                    1.00595868e-01,
                    1.49589425e-01,
                    -3.49109621e-02,
                    -6.64936729e-02,
                    -7.90980545e-06,
                ],
                [
                    0.00000000e00,
                    9.48395495e-03,
                    9.39976641e-02,
                    2.80610590e-01,
                    4.35942983e-01,
                    5.29728249e-01,
                    6.02570102e-01,
                    5.74539705e-01,
                    3.94972587e-01,
                    3.46072026e-02,
                    -3.99003102e-01,
                    -7.48458529e-01,
                    -9.17612026e-01,
                    -9.48254388e-01,
                    -9.59351094e-01,
                    -1.11192861e00,
                    -1.54253546e00,
                    -2.20047334e00,
                    -2.57490994e00,
                    -1.94461890e00,
                    -5.05577270e-01,
                    8.31801376e-01,
                    1.60094061e00,
                    1.90259622e00,
                    2.04186754e00,
                    2.38784851e00,
                    2.80602678e00,
                    2.94246638e00,
                    2.78869704e00,
                    2.62162740e00,
                    2.42129404e00,
                    2.03048070e00,
                    1.13336171e00,
                    -1.02969380e-01,
                    -1.01059956e00,
                    -1.55450220e00,
                    -1.67710731e00,
                    -1.52714376e00,
                    -1.16372233e00,
                    -7.10565879e-01,
                    -3.08258869e-01,
                    -1.20237034e-02,
                    1.55495432e-01,
                    2.21099883e-01,
                    1.88206861e-01,
                    5.21218489e-02,
                    -6.20624214e-02,
                    -7.52964737e-02,
                    -3.18116619e-02,
                    9.38113225e-06,
                ],
                [
                    0.00000000e00,
                    -2.32934019e-02,
                    -2.07091190e-01,
                    -5.07635966e-01,
                    -5.73810532e-01,
                    -5.77478645e-01,
                    -6.98522828e-01,
                    -8.16420142e-01,
                    -8.98310860e-01,
                    -8.24550444e-01,
                    -5.81829826e-01,
                    -2.65431965e-01,
                    -5.84964063e-02,
                    -2.98291703e-02,
                    -1.85843878e-01,
                    -5.15175860e-01,
                    -9.93835408e-01,
                    -1.56954521e00,
                    -2.15018156e00,
                    -2.27726921e00,
                    -1.87789803e00,
                    -1.37842346e00,
                    -8.48596674e-01,
                    -3.82390579e-01,
                    -2.44788253e-01,
                    -5.14352966e-01,
                    -7.96063002e-01,
                    -7.36249218e-01,
                    -3.48518664e-01,
                    5.25673575e-02,
                    5.23861547e-01,
                    1.01050697e00,
                    1.51015768e00,
                    1.53568187e00,
                    1.30971094e00,
                    1.20821319e00,
                    1.06380226e00,
                    1.04076446e00,
                    9.88899326e-01,
                    8.25984298e-01,
                    6.64677830e-01,
                    5.31461604e-01,
                    5.30657673e-01,
                    6.94246081e-01,
                    1.01352210e00,
                    1.38697342e00,
                    1.46821743e00,
                    1.42746133e00,
                    1.41411171e00,
                    1.36141380e00,
                ],
                [
                    0.00000000e00,
                    1.35360694e-01,
                    4.06082082e-01,
                    4.06082082e-01,
                    1.35360695e-01,
                    -1.17307072e-01,
                    -2.02176759e-01,
                    -1.66465960e-01,
                    -2.78318532e-02,
                    2.42889533e-01,
                    4.60461592e-01,
                    4.27558702e-01,
                    1.77083754e-01,
                    -9.36376318e-02,
                    -3.64359018e-01,
                    -6.35080401e-01,
                    -9.05801741e-01,
                    -1.02080723e00,
                    -9.83350869e-01,
                    -1.02316002e00,
                    -9.95060088e-01,
                    -9.13384054e-01,
                    -8.01610536e-01,
                    -7.40570522e-01,
                    -7.55952719e-01,
                    -7.52832496e-01,
                    -7.46674632e-01,
                    -7.39371280e-01,
                    -6.55091956e-01,
                    -6.71753479e-01,
                    -7.80944705e-01,
                    -8.15619361e-01,
                    -7.41102783e-01,
                    -4.70381396e-01,
                    -1.99660010e-01,
                    -1.79428030e-01,
                    -1.59256799e-01,
                    1.11403832e-01,
                    3.82125206e-01,
                    4.48050053e-01,
                    3.08744345e-01,
                    1.19737177e-01,
                    -1.16829627e-01,
                    -3.69619713e-01,
                    -5.72666593e-01,
                    -5.04992201e-01,
                    -2.34270818e-01,
                    5.02038116e-04,
                    2.24149940e-01,
                    3.48385677e-01,
                ],
                [
                    2.50000000e00,
                    2.49655307e00,
                    2.48965920e00,
                    2.48276534e00,
                    2.48114102e00,
                    2.48641056e00,
                    2.49302502e00,
                    2.49963949e00,
                    2.50653335e00,
                    2.51342722e00,
                    2.52032108e00,
                    2.52721494e00,
                    2.53410881e00,
                    2.54100267e00,
                    2.54789653e00,
                    2.55479040e00,
                    2.56168426e00,
                    2.56857812e00,
                    2.57547199e00,
                    2.57701442e00,
                    2.57236082e00,
                    2.56616480e00,
                    2.55927094e00,
                    2.55237708e00,
                    2.55037425e00,
                    2.55436390e00,
                    2.55844074e00,
                    2.55753875e00,
                    2.55165855e00,
                    2.54476468e00,
                    2.53787082e00,
                    2.53097696e00,
                    2.52408309e00,
                    2.52408309e00,
                    2.53097696e00,
                    2.53787082e00,
                    2.54476468e00,
                    2.54854407e00,
                    2.54683927e00,
                    2.54307943e00,
                    2.53865304e00,
                    2.53250229e00,
                    2.52560843e00,
                    2.51871457e00,
                    2.51182070e00,
                    2.50492684e00,
                    2.50328331e00,
                    2.50702071e00,
                    2.50708725e00,
                    2.50521981e00,
                ],
            ]
        )
        X = self.integrate_nonlinear_full(x0, U, p)

        return np.all(np.abs(X - expected_x) < threshold)
