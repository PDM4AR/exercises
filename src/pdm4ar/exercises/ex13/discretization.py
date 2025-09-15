import numpy as np
import sympy as spy
from numpy.typing import NDArray
from scipy.integrate import odeint

from pdm4ar.exercises.ex13.spaceship import SpaceshipDyn


class DiscretizationMethod:
    K: int  # number of discretization points
    N_sub: int  # number of substeps to approximate the ode
    range_t: tuple  # range of discretization points

    spaceship: SpaceshipDyn

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, spaceship: SpaceshipDyn, K: int, N_sub: int):
        # number of discretization points
        self.K = K

        # tuple containing N_sub ts between [0, 1/(K-1)] used to approximate the ODE
        self.N_sub = N_sub
        self.range_t = tuple(np.linspace(0, 1.0 / (self.K - 1), self.N_sub))

        self.f, self.A, self.B, self.F = spaceship.get_dynamics()

        # number of states, inputs and parameters
        self.n_x = spaceship.n_x
        self.n_u = spaceship.n_u
        self.n_p = spaceship.n_p
    
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

    def __init__(self, spaceship: SpaceshipDyn, K: int, N_sub: int):

        super().__init__(spaceship, K, N_sub)

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
            P = np.array(odeint(self._ode_dPdt,
                                self.P0,
                                self.range_t,
                                args=(U[:, k], p))[-1, :])

            Phi = P[self.A_bar_ind].reshape((self.n_x, self.n_x))
            self.A_bar[:, k] = Phi.flatten(order='F')
            self.B_bar[:, k] = (Phi @ P[self.B_bar_ind].reshape((self.n_x, self.n_u))).flatten(order='F')
            self.F_bar[:, k] = (Phi @ P[self.F_bar_ind]).reshape((self.n_x, self.n_p)).flatten(order='F')
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
            X_nl[:, k + 1] = odeint(self._dxdt,
                                    X_l[:, k],
                                    self.range_t,
                                    args=(U[:, k], p))[-1, :]

        return X_nl

    def integrate_nonlinear_full(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:

        X_nl = np.zeros([x0.size, self.K])
        X_nl[:, 0] = x0

        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(self._dxdt,
                                    X_nl[:, k],
                                    self.range_t,
                                    args=(U[:, k], p))[-1, :]

        return X_nl

    def integrate_nonlinear_full_dense(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:
        
        X_nl = np.zeros([x0.size, U.shape[1]])
        X_nl_dense = np.zeros([x0.size, U.shape[1] + (U.shape[1] - 1) * (self.N_sub - 2)])
        X_nl[:, 0] = x0
        X_nl_dense[:, 0] = x0

        for k in range(U.shape[1] - 1):
            x = odeint(self._dxdt,
                       X_nl[:, k],
                       self.range_t,
                       args=(U[:, k], p))
            X_nl[:, k + 1] = x[-1, :]
            X_nl_dense[:, k * (self.N_sub - 1) + 1:(k + 1) * (self.N_sub - 1) + 1] = np.array(x)[1:, :].T

        return X_nl_dense

    def _dxdt(self, x: NDArray, t: float, u: NDArray, p: NDArray) -> NDArray:
        return np.squeeze(self.f(x, u, p))
    
    def check_dynamics(self) -> bool:

        threshold = 1e-4
        x0 = np.array([ -9.0, -9.0, 0, 0.0, 0.0, 0.0, 0.0, 2.5 ])
        U = np.array([  [-7.16632509e-12,  2.00000000e+00,  2.00000000e+00,  2.00000000e+00, -1.05752561e+00, -2.00000000e+00, -1.83788525e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00,  1.10504277e+00, 1.59509275e+00,  1.99999999e+00,  2.00000000e+00,  2.00000000e+00, -8.37909487e-01, -1.47698868e+00, -8.88497023e-01,  1.41185025e+00, 2.00000000e+00,  2.00000000e+00,  2.00000000e+00,  2.00000000e+00, 2.00000000e+00, -1.99999999e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -1.92901516e-01,  1.18207288e+00,  9.99483945e-01, 1.56882470e+00, 2.00000000e+00,  2.00000000e+00,  2.00000000e+00, 2.00000000e+00,  2.00000000e+00, -1.04638321e+00, -1.12215158e+00, 1.08354082e+00, -2.68917508e-11],
                        [-5.92915767e-11,  7.85398163e-01,  7.85398163e-01, -7.85398159e-01, -7.85398162e-01, -6.80646331e-01,  1.88210197e-01,  1.89932007e-02, 7.85398161e-01,  7.85398158e-01, 4.77011856e-01, -6.67923030e-01, -7.85398157e-01, -7.85398158e-01, -7.85398157e-01, -7.85398145e-01, -7.85397903e-01,  1.18105969e-01,  9.92256355e-02, -3.30208731e-01, 4.93251872e-01, -1.93461495e-02,  6.67885344e-01, -3.13715210e-01, 2.24463679e-01, -2.06359332e-01,  2.42088868e-01, -1.99712906e-01, 6.88723597e-01, -7.85398115e-01,  1.51842051e-01, -3.53033477e-01, 7.85398160e-01,  7.85398160e-01,  7.85398154e-01, -6.68006909e-01, 7.85045666e-01,  7.85398137e-01,  7.85398108e-01, -4.02884894e-01, -4.05403275e-01, -6.91265766e-01, -6.81356728e-01,-7.85397496e-01, -3.92733624e-01, 7.85398148e-01,  7.85398151e-01,  5.76815388e-01, 7.20848254e-01, -1.21876164e-12]])
        p = np.array([16.88996566])
        expected_x = np.array([[-9.00000000e+00, -9.00000000e+00, -8.95271189e+00, -8.81361391e+00, -8.58534401e+00, -8.33500326e+00, -8.14741502e+00, -8.01867557e+00, -7.92952384e+00, -7.86113312e+00, -7.80508979e+00, -7.75774918e+00, -7.71025235e+00, -7.63696985e+00, -7.51253399e+00, -7.31661922e+00, -7.03396868e+00, -6.66139712e+00, -6.22012202e+00, -5.75215278e+00, -5.27444133e+00, -4.73912843e+00, -4.12509059e+00, -3.43861762e+00, -2.68996939e+00, -1.92498376e+00, -1.18401946e+00, -4.58428772e-01, 2.62241815e-01, 9.64512923e-01, 1.64294586e+00, 2.28649800e+00, 2.90734335e+00, 3.53388022e+00, 4.14240089e+00, 4.66126090e+00, 5.10520356e+00, 5.49663414e+00, 5.87692810e+00, 6.23974057e+00, 6.55807560e+00, 6.81850914e+00, 6.99718036e+00, 7.08497687e+00, 7.08335711e+00, 6.99239137e+00, 6.80889775e+00, 6.59744571e+00, 6.39678638e+00, 6.20642989e+00], 
                            [-9.00000000e+00, -9.00000000e+00, -8.99577118e+00, -8.96761988e+00, -8.90615696e+00, -8.83303284e+00, -8.72472558e+00, -8.54731295e+00, -8.28808087e+00, -7.93649261e+00, -7.49105644e+00, -6.95162386e+00, -6.31838120e+00, -5.59564596e+00, -4.79526745e+00, -3.93598442e+00, -3.04630519e+00, -2.16814413e+00, -1.34837819e+00, -6.14029063e-01, 9.42218365e-02, 8.18757542e-01, 1.52235100e+00, 2.16919324e+00, 2.74722114e+00, 3.30367494e+00, 3.90823190e+00, 4.56530445e+00, 5.21025469e+00, 5.77819194e+00, 6.25613577e+00, 6.64747441e+00, 6.94822277e+00, 7.15723181e+00, 7.34901466e+00, 7.54260974e+00, 7.68277813e+00, 7.74729658e+00, 7.76280154e+00, 7.79367309e+00, 7.84796505e+00, 7.91798014e+00, 8.00546774e+00, 8.11668098e+00, 8.25749644e+00, 8.42876901e+00, 8.60741995e+00, 8.78931220e+00, 9.01920379e+00, 9.25547464e+00],
                            [ 0.00000000e+00, -3.11675288e-24, -1.06825462e-02, -8.45256443e-02, -2.61779857e-01, -4.69652137e-01, -6.81142101e-01, -9.34190175e-01, -1.22781714e+00, -1.54964867e+00, -1.84613307e+00, -2.05911321e+00, -2.16314140e+00, -2.19602176e+00, -2.21905087e+00, -2.29569314e+00, -2.48550754e+00, -2.83980924e+00, -3.39233058e+00, -4.14498156e+00, -4.94077964e+00, -5.59906480e+00, -6.08545435e+00, -6.38937054e+00, -6.53273167e+00, -6.62899226e+00, -6.81814278e+00, -7.10440868e+00, -7.37018185e+00, -7.50280169e+00, -7.49739850e+00, -7.32972253e+00, -6.99434381e+00, -6.48693940e+00, -5.96197116e+00, -5.51465592e+00, -5.10226596e+00, -4.73957943e+00, -4.38963921e+00, -4.06108719e+00, -3.78851514e+00, -3.57077013e+00, -3.39809154e+00, -3.22568898e+00, -2.99705420e+00, -2.65855716e+00, -2.19148552e+00, -1.68889948e+00, -1.20014964e+00, -7.20835696e-01],
                            [ 0.00000000e+00, -9.88073344e-13, 2.72524008e-01, 5.17699660e-01, 7.19193228e-01, 4.79555720e-01, 8.76078398e-02, -3.18042244e-01, -7.58510433e-01, -1.15195267e+00, -1.41499714e+00, -1.56302898e+00, -1.72541970e+00, -1.95884560e+00, -2.20307106e+00, -2.36826706e+00, -2.33449241e+00, -1.88022136e+00, -7.41542013e-01, 9.67065643e-01, 2.36616698e+00, 2.60893638e+00, 2.24371193e+00, 1.88653797e+00, 1.79154352e+00, 1.49764722e+00, 8.78591484e-01, -9.36943232e-03, -6.09563323e-01, -7.33468457e-01, -5.11602682e-01, 4.20900485e-02, 8.72857541e-01, 1.74314241e+00, 1.73933425e+00, 1.34434096e+00, 7.38302600e-01, 8.76529605e-02, -3.20170001e-01, -4.91715062e-01, -5.78108022e-01, -4.98487574e-01, -3.00889198e-01, -9.33501974e-02, 6.60719813e-02, 1.06111581e-01, -2.05211005e-03, -5.18192074e-01, -9.08001006e-01, -7.94298580e-01],
                            [ 0.00000000e+00, 1.00968261e-23, 3.93590782e-02, 1.79817009e-01, 4.00878717e-01, 5.06374970e-01, 5.98271756e-01, 6.19394563e-01, 5.05315122e-01, 2.02806777e-01, -2.45953357e-01, -6.84345036e-01, -9.66266578e-01, -1.07367649e+00, -1.09683881e+00, -1.17789534e+00, -1.46875425e+00, -2.01429595e+00, -2.52485262e+00, -2.20394958e+00, -9.35907767e-01, 5.74235615e-01, 1.55869567e+00, 1.99277679e+00, 2.07199626e+00, 2.30773145e+00, 2.66816015e+00, 2.87490107e+00, 2.66207857e+00, 2.40544150e+00, 2.23810102e+00, 2.08338539e+00, 1.72503855e+00, 8.58353815e-01, 4.53171669e-02, -6.02897714e-01, -9.90012863e-01, -1.09931027e+00, -1.06109903e+00, -8.66283726e-01, -6.59850383e-01, -4.76112976e-01, -3.73654780e-01, -3.72147784e-01, -4.70370650e-01, -6.52899106e-01, -8.13711282e-01, -6.46805084e-01, -2.91235107e-01, 1.58431006e-01],
                            [ 0.00000000e+00, -2.52420651e-23, -9.27472788e-02, -3.64233077e-01, -6.35718877e-01, -5.86677562e-01, -6.67177187e-01, -7.94364397e-01, -9.08594274e-01, -9.27720167e-01, -7.62422855e-01, -4.56430939e-01, -1.71207440e-01, -5.01358680e-02, -1.14397422e-01, -3.59311060e-01, -7.67036410e-01, -1.30787329e+00, -1.89555541e+00, -2.46934452e+00, -2.14432316e+00, -1.68369602e+00, -1.13799522e+00, -6.43779887e-01, -1.78871221e-01, -3.76948338e-01, -7.24956389e-01, -9.32907632e-01, -6.05052707e-01, -1.86041925e-01, 2.41695187e-01, 7.26933810e-01, 1.22860100e+00, 1.69258709e+00, 1.38109233e+00, 1.24477928e+00, 1.12201830e+00, 1.01302559e+00, 1.02039517e+00, 8.68922457e-01, 7.19795269e-01, 5.55611230e-01, 4.73457819e-01, 5.53630917e-01, 8.01919388e-01, 1.17519554e+00, 1.50770351e+00, 1.42423267e+00, 1.42442654e+00, 1.34162165e+00],
                            [ 0.00000000e+00, -2.04374019e-11, 2.70721388e-01, 5.41442776e-01, 2.70721389e-01, 1.70302836e-09, -2.34614144e-01, -1.69739374e-01, -1.63192547e-01, 1.07528840e-01, 3.78250226e-01, 5.42672958e-01, 3.12444447e-01, 4.17230614e-02, -2.28998325e-01, -4.99719710e-01, -7.70441092e-01, -1.04116239e+00, -1.00045207e+00, -9.66249669e-01, -1.08007037e+00, -9.10049811e-01, -9.16718297e-01, -6.86502776e-01, -7.94638268e-01, -7.17267169e-01, -7.88397823e-01, -7.04951442e-01, -7.73791118e-01, -5.36392794e-01, -8.07114165e-01, -7.54775246e-01, -8.76463477e-01, -6.05742090e-01, -3.35020703e-01, -6.42993183e-02, -2.94556742e-01, -2.39568574e-02, 2.46764521e-01, 5.17485890e-01, 3.78614216e-01, 2.38874474e-01, 5.99880787e-04, -2.34259134e-01, -5.04980292e-01, -6.40352893e-01, -3.69631510e-01, -9.89101263e-02, 9.99142021e-02, 3.48385677e-01],
                            [ 2.50000000e+00, 2.50000000e+00, 2.49310614e+00, 2.48621227e+00, 2.47931841e+00, 2.48296363e+00, 2.48985749e+00, 2.49619256e+00, 2.50308642e+00, 2.50998028e+00, 2.51687415e+00, 2.52376801e+00, 2.53066187e+00, 2.53755574e+00, 2.54444960e+00, 2.55134346e+00, 2.55823733e+00, 2.56513119e+00, 2.57202506e+00, 2.57891892e+00, 2.57510991e+00, 2.56961174e+00, 2.56271787e+00, 2.55582401e+00, 2.54893015e+00, 2.55181836e+00, 2.55690944e+00, 2.55997203e+00, 2.55510548e+00, 2.54821162e+00, 2.54131775e+00, 2.53442389e+00, 2.52753002e+00, 2.52063616e+00, 2.52753002e+00, 2.53442389e+00, 2.54131775e+00, 2.54821162e+00, 2.54887653e+00, 2.54480201e+00, 2.54135686e+00, 2.53594922e+00, 2.52905536e+00, 2.52216150e+00, 2.51526763e+00, 2.50837377e+00, 2.50147991e+00, 2.50508672e+00, 2.50895470e+00, 2.50521981e+00]])
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

    def __init__(self, spaceship: SpaceshipDyn, K: int, N_sub: int):

        super().__init__(spaceship, K, N_sub)

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

    def calculate_discretization(self, X: NDArray, U: NDArray, p: NDArray) -> tuple[
        NDArray, NDArray, NDArray, NDArray, NDArray]:

        for k in range(self.K - 1):
            self.P0[self.x_ind] = X[:, k]
            P = np.array(odeint(self._ode_dPdt,
                                self.P0,
                                self.range_t,
                                args=(U[:, k], U[:, k + 1], p))[-1, :])

            # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
            # flatten matrices in column-major (Fortran) order for CVXPY
            Phi = P[self.A_bar_ind].reshape((self.n_x, self.n_x))
            self.A_bar[:, k] = Phi.flatten(order='F')
            self.B_plus_bar[:, k] = (Phi @ P[self.B_plus_bar_ind].reshape((self.n_x, self.n_u))).flatten(order='F')
            self.B_minus_bar[:, k] = (Phi @ P[self.B_minus_bar_ind].reshape((self.n_x, self.n_u))).flatten(order='F')
            self.F_bar[:, k] = (Phi @ P[self.F_bar_ind].reshape((self.n_x, self.n_p))).flatten(order='F')
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
            X_nl[:, k + 1] = odeint(self._dxdt,
                                    X_l[:, k],
                                    self.range_t,
                                    args=(U[:, k], U[:, k + 1], p))[-1, :]

        return X_nl

    def integrate_nonlinear_full(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:

        X_nl = np.zeros([x0.size, U.shape[1]])
        X_nl[:, 0] = x0

        for k in range(U.shape[1] - 1):
            X_nl[:, k + 1] = odeint(self._dxdt, X_nl[:, k],
                                    self.range_t,
                                    args=(U[:, k], U[:, k + 1], p))[-1, :]

        return X_nl

    def integrate_nonlinear_full_dense(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:

        X_nl = np.zeros([x0.size, U.shape[1]])
        X_nl_dense = np.zeros([x0.size, U.shape[1] + (U.shape[1] - 1) * (self.N_sub - 2)])
        X_nl[:, 0] = x0
        X_nl_dense[:, 0] = x0

        for k in range(U.shape[1] - 1):
            x = odeint(self._dxdt,
                       X_nl[:, k],
                       self.range_t,
                       args=(U[:, k], U[:, k + 1], p))
            X_nl[:, k + 1] = x[-1, :]
            X_nl_dense[:, k * (self.N_sub - 1) + 1:(k + 1) * (self.N_sub - 1) + 1] = np.array(x)[1:, :].T

        return X_nl_dense

    def _dxdt(self, x: NDArray, t: float, u_t0: NDArray, u_t1: NDArray, p: NDArray) -> NDArray:
        u = u_t0 + (self.K - 1) * t * (u_t1 - u_t0)
        return np.squeeze(self.f(x, u, p))

    def check_dynamics(self) -> bool:

        threshold = 1e-4
        x0 = np.array([ -9.0, -9.0, 0, 0.0, 0.0, 0.0, 0.0, 2.5 ])
        U = np.array([  [-7.16632509e-12,  2.00000000e+00,  2.00000000e+00,  2.00000000e+00, -1.05752561e+00, -2.00000000e+00, -1.83788525e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00,  1.10504277e+00, 1.59509275e+00,  1.99999999e+00,  2.00000000e+00,  2.00000000e+00, -8.37909487e-01, -1.47698868e+00, -8.88497023e-01,  1.41185025e+00, 2.00000000e+00,  2.00000000e+00,  2.00000000e+00,  2.00000000e+00, 2.00000000e+00, -1.99999999e+00, -2.00000000e+00, -2.00000000e+00, -2.00000000e+00, -1.92901516e-01,  1.18207288e+00,  9.99483945e-01, 1.56882470e+00, 2.00000000e+00,  2.00000000e+00,  2.00000000e+00, 2.00000000e+00,  2.00000000e+00, -1.04638321e+00, -1.12215158e+00, 1.08354082e+00, -2.68917508e-11],
                        [-5.92915767e-11,  7.85398163e-01,  7.85398163e-01, -7.85398159e-01, -7.85398162e-01, -6.80646331e-01,  1.88210197e-01,  1.89932007e-02, 7.85398161e-01,  7.85398158e-01, 4.77011856e-01, -6.67923030e-01, -7.85398157e-01, -7.85398158e-01, -7.85398157e-01, -7.85398145e-01, -7.85397903e-01,  1.18105969e-01,  9.92256355e-02, -3.30208731e-01, 4.93251872e-01, -1.93461495e-02,  6.67885344e-01, -3.13715210e-01, 2.24463679e-01, -2.06359332e-01,  2.42088868e-01, -1.99712906e-01, 6.88723597e-01, -7.85398115e-01,  1.51842051e-01, -3.53033477e-01, 7.85398160e-01,  7.85398160e-01,  7.85398154e-01, -6.68006909e-01, 7.85045666e-01,  7.85398137e-01,  7.85398108e-01, -4.02884894e-01, -4.05403275e-01, -6.91265766e-01, -6.81356728e-01,-7.85397496e-01, -3.92733624e-01, 7.85398148e-01,  7.85398151e-01,  5.76815388e-01, 7.20848254e-01, -1.21876164e-12]])
        p = np.array([16.88996566])
        expected_x = np.array([ [-9.00000000e+00, -8.98417195e+00, -8.89032197e+00, -8.70664610e+00, -8.45485644e+00, -8.23410669e+00, -8.08053682e+00, -7.97816238e+00, -7.91028999e+00, -7.86360715e+00, -7.83419769e+00, -7.81816268e+00, -7.79898846e+00, -7.75021892e+00, -7.64842043e+00, -7.47338911e+00, -7.21009532e+00, -6.85940990e+00, -6.44657298e+00, -6.01587862e+00, -5.55799875e+00, -5.03023703e+00, -4.41728649e+00, -3.72451645e+00, -2.97584310e+00, -2.24184464e+00, -1.54316741e+00, -8.58310267e-01, -1.67528608e-01,  5.27862362e-01,  1.22191862e+00,  1.91502224e+00, 2.62864244e+00,  3.37490311e+00,  4.05976523e+00,  4.67466618e+00, 5.25273745e+00,  5.82856706e+00,  6.40536207e+00,  6.94129747e+00, 7.42496608e+00,  7.83751830e+00,  8.16134374e+00,  8.39209435e+00, 8.52906882e+00,  8.57426101e+00,  8.56098288e+00,  8.54204757e+00, 8.50842192e+00,  8.49995720e+00],
                                [-9.00000000e+00, -8.99935953e+00, -8.98589923e+00, -8.94068339e+00, -8.87540527e+00, -8.79579725e+00, -8.66292376e+00, -8.45632504e+00, -8.16295415e+00, -7.77723237e+00, -7.29856416e+00, -6.72662391e+00, -6.06129394e+00, -5.30767157e+00, -4.47779015e+00, -3.59160940e+00, -2.68155881e+00, -1.79417393e+00, -9.72187493e-01, -2.14896383e-01, 5.59543219e-01,  1.34820687e+00,  2.11397878e+00,  2.83268754e+00, 3.50774789e+00,  4.19915098e+00,  4.93781166e+00,  5.70092510e+00, 6.41211758e+00,  7.03487142e+00,  7.56439418e+00,  8.00051597e+00, 8.34584173e+00,  8.63785688e+00,  8.92085319e+00,  9.14632016e+00, 9.28666764e+00,  9.34934083e+00,  9.39796938e+00,  9.45772657e+00, 9.51475610e+00,  9.55862084e+00,  9.58489800e+00,  9.59756920e+00, 9.59998437e+00,  9.58203397e+00,  9.52650903e+00,  9.49780430e+00, 9.50063471e+00,  9.50000058e+00],
                                [ 0.00000000e+00, -1.60692184e-03, -3.61567368e-02, -1.59337508e-01, -3.54987339e-01, -5.49656844e-01, -7.68262615e-01, -1.02973565e+00, -1.32799084e+00, -1.63024029e+00, -1.87665711e+00, -2.02209713e+00, -2.07320047e+00, -2.08307666e+00, -2.11503578e+00, -2.23115743e+00, -2.48739042e+00, -2.92791014e+00, -3.56939200e+00, -4.35814934e+00, -5.07822513e+00, -5.64183872e+00, -6.02423018e+00, -6.23550581e+00, -6.32435551e+00, -6.45091811e+00, -6.68081968e+00, -6.96017439e+00, -7.14985836e+00, -7.20112955e+00, -7.10337652e+00, -6.83941681e+00, -6.40395890e+00, -5.85659297e+00, -5.37125280e+00, -4.93769234e+00, -4.54651009e+00, -4.18715311e+00, -3.83275713e+00, -3.52008521e+00, -3.26274003e+00, -3.05892029e+00, -2.88054566e+00, -2.67429054e+00, -2.38352909e+00, -1.96867609e+00, -1.46456706e+00, -9.67923043e-01, -4.75845095e-01, -9.39078633e-06],
                                [ 0.00000000e+00,  1.37542868e-01,  4.01217535e-01,  6.26821149e-01, 6.14813909e-01,  3.09090661e-01, -7.71437953e-02, -4.94399693e-01, -9.16367602e-01, -1.25805675e+00, -1.47185138e+00, -1.63238667e+00, -1.84917498e+00, -2.11052714e+00, -2.34321698e+00, -2.46013642e+00, -2.31350393e+00, -1.62858677e+00, -1.96016872e-01,  1.62447532e+00, 2.63846199e+00,  2.68600602e+00,  2.38648162e+00,  2.21211297e+00, 2.09548659e+00, 1.69920244e+00,  9.81775741e-01,  1.95170795e-01, -1.78904361e-01, -9.94167759e-02,  3.42794840e-01,  1.12498788e+00, 2.01228453e+00,  2.27975446e+00,  1.74441862e+00,  9.06284280e-01, -2.67722215e-03, -7.32700290e-01, -1.15213689e+00, -1.31365058e+00, -1.28293034e+00, -1.07808103e+00, -7.91616783e-01, -4.86076618e-01, -1.82038693e-01,  1.00595868e-01,  1.49589425e-01, -3.49109621e-02, -6.64936729e-02, -7.90980545e-06],
                                [ 0.00000000e+00,  9.48395495e-03,  9.39976641e-02,  2.80610590e-01, 4.35942983e-01,  5.29728249e-01,  6.02570102e-01,  5.74539705e-01, 3.94972587e-01,  3.46072026e-02, -3.99003102e-01, -7.48458529e-01, -9.17612026e-01, -9.48254388e-01, -9.59351094e-01, -1.11192861e+00, -1.54253546e+00, -2.20047334e+00, -2.57490994e+00, -1.94461890e+00, -5.05577270e-01,  8.31801376e-01,  1.60094061e+00,  1.90259622e+00, 2.04186754e+00,  2.38784851e+00,  2.80602678e+00,  2.94246638e+00, 2.78869704e+00,  2.62162740e+00,  2.42129404e+00,  2.03048070e+00, 1.13336171e+00, -1.02969380e-01, -1.01059956e+00, -1.55450220e+00, -1.67710731e+00, -1.52714376e+00, -1.16372233e+00, -7.10565879e-01, -3.08258869e-01, -1.20237034e-02,  1.55495432e-01,  2.21099883e-01, 1.88206861e-01,  5.21218489e-02, -6.20624214e-02, -7.52964737e-02, -3.18116619e-02,  9.38113225e-06],
                                [ 0.00000000e+00, -2.32934019e-02, -2.07091190e-01, -5.07635966e-01, -5.73810532e-01, -5.77478645e-01, -6.98522828e-01, -8.16420142e-01, -8.98310860e-01, -8.24550444e-01, -5.81829826e-01, -2.65431965e-01, -5.84964063e-02, -2.98291703e-02, -1.85843878e-01, -5.15175860e-01, -9.93835408e-01, -1.56954521e+00, -2.15018156e+00, -2.27726921e+00, -1.87789803e+00, -1.37842346e+00, -8.48596674e-01, -3.82390579e-01, -2.44788253e-01, -5.14352966e-01, -7.96063002e-01, -7.36249218e-01, -3.48518664e-01,  5.25673575e-02,  5.23861547e-01,  1.01050697e+00, 1.51015768e+00,  1.53568187e+00,  1.30971094e+00,  1.20821319e+00, 1.06380226e+00,  1.04076446e+00,  9.88899326e-01,  8.25984298e-01, 6.64677830e-01,  5.31461604e-01,  5.30657673e-01,  6.94246081e-01, 1.01352210e+00,  1.38697342e+00,  1.46821743e+00,  1.42746133e+00, 1.41411171e+00,  1.36141380e+00],
                                [ 0.00000000e+00,  1.35360694e-01,  4.06082082e-01,  4.06082082e-01, 1.35360695e-01, -1.17307072e-01, -2.02176759e-01, -1.66465960e-01, -2.78318532e-02,  2.42889533e-01,  4.60461592e-01,  4.27558702e-01, 1.77083754e-01, -9.36376318e-02, -3.64359018e-01, -6.35080401e-01, -9.05801741e-01, -1.02080723e+00, -9.83350869e-01, -1.02316002e+00, -9.95060088e-01, -9.13384054e-01, -8.01610536e-01, -7.40570522e-01, -7.55952719e-01, -7.52832496e-01, -7.46674632e-01, -7.39371280e-01, -6.55091956e-01, -6.71753479e-01, -7.80944705e-01, -8.15619361e-01, -7.41102783e-01, -4.70381396e-01, -1.99660010e-01, -1.79428030e-01, -1.59256799e-01,  1.11403832e-01,  3.82125206e-01,  4.48050053e-01, 3.08744345e-01,  1.19737177e-01, -1.16829627e-01, -3.69619713e-01, -5.72666593e-01, -5.04992201e-01, -2.34270818e-01,  5.02038116e-04, 2.24149940e-01,  3.48385677e-01],
                                [ 2.50000000e+00,  2.49655307e+00,  2.48965920e+00,  2.48276534e+00, 2.48114102e+00,  2.48641056e+00,  2.49302502e+00,  2.49963949e+00, 2.50653335e+00,  2.51342722e+00,  2.52032108e+00,  2.52721494e+00, 2.53410881e+00,  2.54100267e+00,  2.54789653e+00,  2.55479040e+00, 2.56168426e+00,  2.56857812e+00,  2.57547199e+00,  2.57701442e+00, 2.57236082e+00,  2.56616480e+00,  2.55927094e+00,  2.55237708e+00, 2.55037425e+00,  2.55436390e+00,  2.55844074e+00,  2.55753875e+00, 2.55165855e+00,  2.54476468e+00,  2.53787082e+00,  2.53097696e+00, 2.52408309e+00,  2.52408309e+00,  2.53097696e+00,  2.53787082e+00, 2.54476468e+00, 2.54854407e+00,  2.54683927e+00,  2.54307943e+00, 2.53865304e+00,  2.53250229e+00,  2.52560843e+00,  2.51871457e+00, 2.51182070e+00,  2.50492684e+00,  2.50328331e+00,  2.50702071e+00, 2.50708725e+00,  2.50521981e+00]]
)
        X = self.integrate_nonlinear_full(x0, U, p)

        return np.all(np.abs(X - expected_x) < threshold)