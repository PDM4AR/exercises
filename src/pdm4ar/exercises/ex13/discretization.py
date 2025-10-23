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
        x0 = np.array([-9.0, -9.0, 0, 0.0, 0.0, 0.0])
        U = np.array(
            [
                [
                    3.94372302e-18,
                    -1.12714278e00,
                    -9.81663394e-01,
                    2.21174790e-01,
                    9.00186398e-01,
                    1.04058043e00,
                    1.02111849e00,
                    1.04449076e00,
                    9.76331112e-01,
                    9.02620053e-01,
                    8.29195213e-01,
                    7.55325056e-01,
                    6.77444300e-01,
                    5.88191430e-01,
                    4.90760355e-01,
                    1.87894377e-01,
                    -3.28981264e-01,
                    -4.59882305e-01,
                    -5.23844672e-01,
                    -5.44266945e-01,
                    -5.31411523e-01,
                    -5.15033472e-01,
                    -4.87321118e-01,
                    -4.47460037e-01,
                    -3.71624857e-01,
                    -3.53279495e-01,
                    -3.70049714e-01,
                    -3.57049007e-01,
                    -3.08504510e-01,
                    -2.50318224e-01,
                    -1.86415758e-01,
                    -1.26791258e-01,
                    -7.14084402e-02,
                    -3.48338488e-02,
                    -4.54874031e-02,
                    -1.84448072e-01,
                    -3.03794591e-01,
                    -3.58487733e-01,
                    -4.20111107e-01,
                    -4.53149995e-01,
                    -4.32280401e-01,
                    -3.19424260e-01,
                    -7.55334276e-03,
                    1.04334452e00,
                    1.21695192e00,
                    1.28560207e00,
                    1.26103526e00,
                    1.13338466e00,
                    6.27924821e-01,
                    2.61179705e-18,
                ],
                [
                    -3.51073503e-18,
                    9.08253133e-01,
                    7.67779346e-01,
                    1.11506653e00,
                    1.00813735e00,
                    8.56821023e-01,
                    4.88703653e-01,
                    5.02491679e-01,
                    3.60892049e-01,
                    2.57755121e-01,
                    1.99958479e-01,
                    1.78654405e-01,
                    1.79269344e-01,
                    1.86067071e-01,
                    1.73262752e-01,
                    -4.12651945e-01,
                    -3.99691783e-01,
                    -4.06087924e-01,
                    -4.15049777e-01,
                    -4.10878746e-01,
                    -3.89275079e-01,
                    -3.74299956e-01,
                    -3.48142390e-01,
                    -3.05652158e-01,
                    -2.20553185e-01,
                    -2.02415212e-01,
                    -2.13186633e-01,
                    -1.90877136e-01,
                    -1.32982106e-01,
                    -6.31808950e-02,
                    1.88526822e-03,
                    7.78651706e-02,
                    1.39315023e-01,
                    1.71671276e-01,
                    1.65351662e-01,
                    5.98806610e-02,
                    -2.35413175e-01,
                    -4.50803898e-01,
                    -6.00903264e-01,
                    -6.86188473e-01,
                    -7.13489097e-01,
                    -6.91809195e-01,
                    -6.29607136e-01,
                    1.58046756e-01,
                    6.32543699e-01,
                    9.93037377e-01,
                    1.24908559e00,
                    1.42087303e00,
                    1.54200818e00,
                    -6.44168970e-18,
                ],
            ]
        )
        p = np.array([19.33735434])
        expected_x = np.array(
            [
                [
                    -9.00000000e00,
                    -9.00000000e00,
                    -9.00852134e00,
                    -9.03382561e00,
                    -9.01815784e00,
                    -8.89406862e00,
                    -8.66777413e00,
                    -8.37971356e00,
                    -8.06735025e00,
                    -7.75759948e00,
                    -7.46946355e00,
                    -7.21349316e00,
                    -6.99484469e00,
                    -6.81530625e00,
                    -6.67405548e00,
                    -6.56760895e00,
                    -6.47189909e00,
                    -6.35416279e00,
                    -6.20169316e00,
                    -6.01278314e00,
                    -5.78844242e00,
                    -5.53139095e00,
                    -5.24459481e00,
                    -4.93066544e00,
                    -4.59237776e00,
                    -4.23358384e00,
                    -3.85701578e00,
                    -3.46214458e00,
                    -3.04808665e00,
                    -2.61605420e00,
                    -2.16915051e00,
                    -1.71152976e00,
                    -1.24843775e00,
                    -7.85943418e-01,
                    -3.29425524e-01,
                    1.19016390e-01,
                    5.67813504e-01,
                    1.04093250e00,
                    1.56523145e00,
                    2.16045869e00,
                    2.83956397e00,
                    3.60621382e00,
                    4.45358119e00,
                    5.36101774e00,
                    6.24870204e00,
                    7.02816605e00,
                    7.65889070e00,
                    8.11301827e00,
                    8.37633678e00,
                    8.45911816e00,
                ],
                [
                    -9.00000000e00,
                    -9.00000000e00,
                    -9.00009017e00,
                    -9.00140290e00,
                    -8.98744112e00,
                    -8.91028724e00,
                    -8.72733648e00,
                    -8.42797733e00,
                    -8.01275867e00,
                    -7.48572268e00,
                    -6.86414850e00,
                    -6.16378012e00,
                    -5.39678581e00,
                    -4.57210512e00,
                    -3.69677202e00,
                    -2.77756918e00,
                    -1.84505276e00,
                    -9.42411295e-01,
                    -9.12174644e-02,
                    6.99905699e-01,
                    1.42636497e00,
                    2.08753054e00,
                    2.68481131e00,
                    3.22066381e00,
                    3.69966351e00,
                    4.13046508e00,
                    4.52026100e00,
                    4.86966781e00,
                    5.17943550e00,
                    5.45513677e00,
                    5.70548926e00,
                    5.93968902e00,
                    6.16663425e00,
                    6.39403317e00,
                    6.62669794e00,
                    6.86523276e00,
                    7.10403275e00,
                    7.33418895e00,
                    7.55303661e00,
                    7.76629677e00,
                    7.98402871e00,
                    8.21707221e00,
                    8.47306848e00,
                    8.75158707e00,
                    9.02045635e00,
                    9.24035985e00,
                    9.39945073e00,
                    9.49897767e00,
                    9.54911680e00,
                    9.56427234e00,
                ],
                [
                    0.00000000e00,
                    -2.32192411e-19,
                    6.33987719e-02,
                    2.44688181e-01,
                    5.08312511e-01,
                    8.03142364e-01,
                    1.09561093e00,
                    1.36577200e00,
                    1.60246710e00,
                    1.80311016e00,
                    1.96449711e00,
                    2.08619818e00,
                    2.17033750e00,
                    2.22099738e00,
                    2.24361463e00,
                    2.24381700e00,
                    2.21542402e00,
                    2.16612265e00,
                    2.11629437e00,
                    2.07153045e00,
                    2.03431007e00,
                    2.00567177e00,
                    1.98584434e00,
                    1.97473565e00,
                    1.97237917e00,
                    1.97914534e00,
                    1.99531625e00,
                    2.02107229e00,
                    2.05689026e00,
                    2.10335137e00,
                    2.16110865e00,
                    2.23056014e00,
                    2.31225151e00,
                    2.40688119e00,
                    2.51450676e00,
                    2.63513182e00,
                    2.76993450e00,
                    2.91447752e00,
                    3.05827501e00,
                    3.19356570e00,
                    3.31596633e00,
                    3.42234913e00,
                    3.50837371e00,
                    3.56342338e00,
                    3.57152188e00,
                    3.53384178e00,
                    3.46884562e00,
                    3.39436441e00,
                    3.32846572e00,
                    3.29999373e00,
                ],
                [
                    0.00000000e00,
                    8.54371651e-20,
                    -4.31738690e-02,
                    -8.48773169e-02,
                    1.60058227e-01,
                    4.57562313e-01,
                    6.74465523e-01,
                    7.72769517e-01,
                    7.98278274e-01,
                    7.62775287e-01,
                    6.91620210e-01,
                    6.01919125e-01,
                    5.03981628e-01,
                    4.04746942e-01,
                    3.10645524e-01,
                    2.28812586e-01,
                    2.56070847e-01,
                    3.39644890e-01,
                    4.31863120e-01,
                    5.24318578e-01,
                    6.11583925e-01,
                    6.90348780e-01,
                    7.62580014e-01,
                    8.28106004e-01,
                    8.86252153e-01,
                    9.32204661e-01,
                    9.76479135e-01,
                    1.02514065e00,
                    1.07384433e00,
                    1.11624693e00,
                    1.14912554e00,
                    1.17039779e00,
                    1.17661204e00,
                    1.16711929e00,
                    1.14616787e00,
                    1.12624050e00,
                    1.14845868e00,
                    1.25002274e00,
                    1.40766829e00,
                    1.60895934e00,
                    1.83216504e00,
                    2.05224091e00,
                    2.24122871e00,
                    2.35713417e00,
                    2.14170423e00,
                    1.80763068e00,
                    1.38710743e00,
                    9.12612420e-01,
                    4.20660386e-01,
                    -1.47505582e-03,
                ],
                [
                    0.00000000e00,
                    -7.10623641e-39,
                    -9.12507770e-04,
                    -7.00038182e-03,
                    8.85240081e-02,
                    3.17122700e-01,
                    6.20649297e-01,
                    9.00922067e-01,
                    1.20439005e00,
                    1.46540473e00,
                    1.68277124e00,
                    1.86481853e00,
                    2.02086975e00,
                    2.15770217e00,
                    2.27805419e00,
                    2.38038165e00,
                    2.34540027e00,
                    2.22842003e00,
                    2.08458885e00,
                    1.92406320e00,
                    1.75702627e00,
                    1.59332604e00,
                    1.43340106e00,
                    1.28213052e00,
                    1.14537462e00,
                    1.03794141e00,
                    9.37629143e-01,
                    8.33342585e-01,
                    7.36823516e-01,
                    6.60734623e-01,
                    6.08346221e-01,
                    5.78803743e-01,
                    5.71419128e-01,
                    5.80868745e-01,
                    5.97879766e-01,
                    6.10592747e-01,
                    6.00123934e-01,
                    5.68735212e-01,
                    5.44145048e-01,
                    5.41179131e-01,
                    5.66822113e-01,
                    6.18125194e-01,
                    6.81952033e-01,
                    7.30619716e-01,
                    6.31694946e-01,
                    4.84855605e-01,
                    3.25959468e-01,
                    1.84325634e-01,
                    7.51780201e-02,
                    3.63146493e-03,
                ],
                [
                    0.00000000e00,
                    -1.17673058e-18,
                    3.21299363e-01,
                    5.97459315e-01,
                    7.38565449e-01,
                    7.55606149e-01,
                    7.26598633e-01,
                    6.42553782e-01,
                    5.56996000e-01,
                    4.59845280e-01,
                    3.58049511e-01,
                    2.58720746e-01,
                    1.67689850e-01,
                    8.90499675e-02,
                    2.55722433e-02,
                    -2.45466435e-02,
                    -1.19346456e-01,
                    -1.30508533e-01,
                    -1.22016769e-01,
                    -1.04842847e-01,
                    -8.37867258e-02,
                    -6.13496419e-02,
                    -3.91340186e-02,
                    -1.71638278e-02,
                    5.22139019e-03,
                    2.90689530e-02,
                    5.28837784e-02,
                    7.76455496e-02,
                    1.03876769e-01,
                    1.31584025e-01,
                    1.61124767e-01,
                    1.90849205e-01,
                    2.23155441e-01,
                    2.56419394e-01,
                    2.89017457e-01,
                    3.22299659e-01,
                    3.60868404e-01,
                    3.71662818e-01,
                    3.57090162e-01,
                    3.28551043e-01,
                    2.91764532e-01,
                    2.47374065e-01,
                    1.88590886e-01,
                    9.03959912e-02,
                    -4.93535312e-02,
                    -1.41605848e-01,
                    -1.87788926e-01,
                    -1.89675252e-01,
                    -1.44293501e-01,
                    1.73900755e-11,
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
        x0 = np.array([-9.0, -9.0, 0, 0.0, 0.0, 0.0])
        U = np.array(
            [
                [
                    3.94372302e-18,
                    -1.12714278e00,
                    -9.81663394e-01,
                    2.21174790e-01,
                    9.00186398e-01,
                    1.04058043e00,
                    1.02111849e00,
                    1.04449076e00,
                    9.76331112e-01,
                    9.02620053e-01,
                    8.29195213e-01,
                    7.55325056e-01,
                    6.77444300e-01,
                    5.88191430e-01,
                    4.90760355e-01,
                    1.87894377e-01,
                    -3.28981264e-01,
                    -4.59882305e-01,
                    -5.23844672e-01,
                    -5.44266945e-01,
                    -5.31411523e-01,
                    -5.15033472e-01,
                    -4.87321118e-01,
                    -4.47460037e-01,
                    -3.71624857e-01,
                    -3.53279495e-01,
                    -3.70049714e-01,
                    -3.57049007e-01,
                    -3.08504510e-01,
                    -2.50318224e-01,
                    -1.86415758e-01,
                    -1.26791258e-01,
                    -7.14084402e-02,
                    -3.48338488e-02,
                    -4.54874031e-02,
                    -1.84448072e-01,
                    -3.03794591e-01,
                    -3.58487733e-01,
                    -4.20111107e-01,
                    -4.53149995e-01,
                    -4.32280401e-01,
                    -3.19424260e-01,
                    -7.55334276e-03,
                    1.04334452e00,
                    1.21695192e00,
                    1.28560207e00,
                    1.26103526e00,
                    1.13338466e00,
                    6.27924821e-01,
                    2.61179705e-18,
                ],
                [
                    -3.51073503e-18,
                    9.08253133e-01,
                    7.67779346e-01,
                    1.11506653e00,
                    1.00813735e00,
                    8.56821023e-01,
                    4.88703653e-01,
                    5.02491679e-01,
                    3.60892049e-01,
                    2.57755121e-01,
                    1.99958479e-01,
                    1.78654405e-01,
                    1.79269344e-01,
                    1.86067071e-01,
                    1.73262752e-01,
                    -4.12651945e-01,
                    -3.99691783e-01,
                    -4.06087924e-01,
                    -4.15049777e-01,
                    -4.10878746e-01,
                    -3.89275079e-01,
                    -3.74299956e-01,
                    -3.48142390e-01,
                    -3.05652158e-01,
                    -2.20553185e-01,
                    -2.02415212e-01,
                    -2.13186633e-01,
                    -1.90877136e-01,
                    -1.32982106e-01,
                    -6.31808950e-02,
                    1.88526822e-03,
                    7.78651706e-02,
                    1.39315023e-01,
                    1.71671276e-01,
                    1.65351662e-01,
                    5.98806610e-02,
                    -2.35413175e-01,
                    -4.50803898e-01,
                    -6.00903264e-01,
                    -6.86188473e-01,
                    -7.13489097e-01,
                    -6.91809195e-01,
                    -6.29607136e-01,
                    1.58046756e-01,
                    6.32543699e-01,
                    9.93037377e-01,
                    1.24908559e00,
                    1.42087303e00,
                    1.54200818e00,
                    -6.44168970e-18,
                ],
            ]
        )
        p = np.array([19.33735434])
        expected_x = np.array(
            [
                [
                    -9.00000000e00,
                    -9.00284078e00,
                    -9.01980575e00,
                    -9.03378318e00,
                    -8.96454264e00,
                    -8.78494349e00,
                    -8.52213372e00,
                    -8.21796030e00,
                    -7.90342493e00,
                    -7.60210944e00,
                    -7.32843304e00,
                    -7.08994395e00,
                    -6.88995168e00,
                    -6.72889541e00,
                    -6.60463116e00,
                    -6.50630392e00,
                    -6.40050773e00,
                    -6.26468280e00,
                    -6.09296258e00,
                    -5.88511394e00,
                    -5.64308530e00,
                    -5.36989814e00,
                    -5.06827067e00,
                    -4.74087930e00,
                    -4.39089817e00,
                    -4.02212007e00,
                    -3.63550284e00,
                    -3.22999595e00,
                    -2.80568334e00,
                    -2.36480930e00,
                    -1.91103016e00,
                    -1.44899338e00,
                    -9.84446926e-01,
                    -5.23288700e-01,
                    -6.95593425e-02,
                    3.78901096e-01,
                    8.38366230e-01,
                    1.33628353e00,
                    1.89568029e00,
                    2.53324951e00,
                    3.25741129e00,
                    4.06665284e00,
                    4.94764772e00,
                    5.85717511e00,
                    6.69566229e00,
                    7.40464925e00,
                    7.94991795e00,
                    8.31026017e00,
                    8.48201466e00,
                    8.51020037e00,
                ],
                [
                    -9.00000000e00,
                    -9.00001203e00,
                    -9.00052682e00,
                    -8.99861520e00,
                    -8.96001688e00,
                    -8.83514792e00,
                    -8.59535600e00,
                    -8.24045816e00,
                    -7.76929689e00,
                    -7.19477289e00,
                    -6.53380904e00,
                    -5.80043502e00,
                    -5.00513399e00,
                    -4.15576277e00,
                    -3.25908668e00,
                    -2.32988194e00,
                    -1.41071025e00,
                    -5.33960985e-01,
                    2.86707738e-01,
                    1.04467584e00,
                    1.73733674e00,
                    2.36538902e00,
                    2.93062110e00,
                    3.43651775e00,
                    3.88940156e00,
                    4.29841466e00,
                    4.66709472e00,
                    4.99537400e00,
                    5.28639008e00,
                    5.54760585e00,
                    5.78812186e00,
                    6.01698971e00,
                    6.24267435e00,
                    6.47156457e00,
                    6.70642786e00,
                    6.94488987e00,
                    7.17881926e00,
                    7.40154755e00,
                    7.61511132e00,
                    7.82774111e00,
                    8.05031417e00,
                    8.29249970e00,
                    8.55846603e00,
                    8.83551844e00,
                    9.08011651e00,
                    9.26853294e00,
                    9.39566163e00,
                    9.46773805e00,
                    9.49750740e00,
                    9.50194210e00,
                ],
                [
                    0.00000000e00,
                    2.11329378e-02,
                    1.44961498e-01,
                    3.71859839e-01,
                    6.55167035e-01,
                    9.50330621e-01,
                    1.23345544e00,
                    1.48693328e00,
                    1.70598361e00,
                    1.88715138e00,
                    2.02861426e00,
                    2.13126156e00,
                    2.19825366e00,
                    2.23439357e00,
                    2.24536404e00,
                    2.23273815e00,
                    2.19114039e00,
                    2.14092925e00,
                    2.09334763e00,
                    2.05222780e00,
                    2.01925304e00,
                    1.99502746e00,
                    1.97956747e00,
                    1.97282124e00,
                    1.97497799e00,
                    1.98644760e00,
                    2.00737993e00,
                    2.03811861e00,
                    2.07920960e00,
                    2.13125850e00,
                    2.19485684e00,
                    2.27034336e00,
                    2.35847239e00,
                    2.45962191e00,
                    2.57372472e00,
                    2.70126474e00,
                    2.84185099e00,
                    2.98685547e00,
                    3.12685888e00,
                    3.25597578e00,
                    3.37061757e00,
                    3.46729457e00,
                    3.53912784e00,
                    3.57206852e00,
                    3.55571569e00,
                    3.50286248e00,
                    3.43166701e00,
                    3.35992257e00,
                    3.30948435e00,
                    3.29999369e00,
                ],
                [
                    0.00000000e00,
                    -2.15944384e-02,
                    -6.41505243e-02,
                    4.14055609e-02,
                    3.18391299e-01,
                    5.78281540e-01,
                    7.33713705e-01,
                    7.95618061e-01,
                    7.88232189e-01,
                    7.32874320e-01,
                    6.51018421e-01,
                    5.56264817e-01,
                    4.57072737e-01,
                    3.60009275e-01,
                    2.71801096e-01,
                    2.44726051e-01,
                    3.00835149e-01,
                    3.88951952e-01,
                    4.81348072e-01,
                    5.71143791e-01,
                    6.54018560e-01,
                    7.29380741e-01,
                    7.98123346e-01,
                    8.59837981e-01,
                    9.11805081e-01,
                    9.56850549e-01,
                    1.00322216e00,
                    1.05185848e00,
                    1.09747183e00,
                    1.13525336e00,
                    1.16252646e00,
                    1.17652638e00,
                    1.17513596e00,
                    1.16006208e00,
                    1.13956814e00,
                    1.14022165e00,
                    1.20155485e00,
                    1.33121073e00,
                    1.51104550e00,
                    1.72374918e00,
                    1.94570721e00,
                    2.15026339e00,
                    2.30235126e00,
                    2.25188371e00,
                    1.97772958e00,
                    1.60092915e00,
                    1.15343234e00,
                    6.69839352e-01,
                    2.12249317e-01,
                    9.74555020e-04,
                ],
                [
                    0.00000000e00,
                    -1.82618752e-04,
                    -3.29393599e-03,
                    2.98195316e-02,
                    1.88142642e-01,
                    4.57241501e-01,
                    7.54040440e-01,
                    1.04837856e00,
                    1.33226902e00,
                    1.57203706e00,
                    1.77175142e00,
                    1.94057591e00,
                    2.08673133e00,
                    2.21506061e00,
                    2.32621341e00,
                    2.36005332e00,
                    2.28456268e00,
                    2.15424712e00,
                    2.00204613e00,
                    1.83818434e00,
                    1.67271711e00,
                    1.51082655e00,
                    1.35516293e00,
                    1.21109650e00,
                    1.08896673e00,
                    9.85062348e-01,
                    8.82712307e-01,
                    7.82272328e-01,
                    6.95977115e-01,
                    6.31792880e-01,
                    5.90930992e-01,
                    5.72647312e-01,
                    5.73912029e-01,
                    5.87346419e-01,
                    6.02203060e-01,
                    6.02426988e-01,
                    5.79607611e-01,
                    5.50296734e-01,
                    5.35639406e-01,
                    5.46726715e-01,
                    5.85534442e-01,
                    6.43924704e-01,
                    7.01426998e-01,
                    6.77916868e-01,
                    5.53666055e-01,
                    3.99197795e-01,
                    2.47972786e-01,
                    1.22747449e-01,
                    3.44189617e-02,
                    -1.52941073e-04,
                ],
                [
                    0.00000000e00,
                    1.60649682e-01,
                    4.59379340e-01,
                    6.68012382e-01,
                    7.47085799e-01,
                    7.41102391e-01,
                    6.84576208e-01,
                    5.99774891e-01,
                    5.08420640e-01,
                    4.08947395e-01,
                    3.08385128e-01,
                    2.13205298e-01,
                    1.28369909e-01,
                    5.73111054e-02,
                    5.12799861e-04,
                    -7.19465499e-02,
                    -1.24927494e-01,
                    -1.26262651e-01,
                    -1.13429808e-01,
                    -9.43147865e-02,
                    -7.25681838e-02,
                    -5.02418302e-02,
                    -2.81489232e-02,
                    -5.97121879e-03,
                    1.71451716e-02,
                    4.09763657e-02,
                    6.52646640e-02,
                    9.07611593e-02,
                    1.17730397e-01,
                    1.46354396e-01,
                    1.75986986e-01,
                    2.07002323e-01,
                    2.39787417e-01,
                    2.72718426e-01,
                    3.05658558e-01,
                    3.41584032e-01,
                    3.66265611e-01,
                    3.64376490e-01,
                    3.42820602e-01,
                    3.10157788e-01,
                    2.69569299e-01,
                    2.17982476e-01,
                    1.39493439e-01,
                    2.05212299e-02,
                    -9.54796895e-02,
                    -1.64697387e-01,
                    -1.88732089e-01,
                    -1.66984377e-01,
                    -7.21467504e-02,
                    1.85046113e-11,
                ],
            ]
        )
        X = self.integrate_nonlinear_full(x0, U, p)

        return np.all(np.abs(X - expected_x) < threshold)
