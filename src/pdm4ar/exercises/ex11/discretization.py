import numpy as np
import sympy as spy
from scipy.integrate import odeint

from typing import Any
from numpy.typing import NDArray

from pdm4ar.exercises.ex11.rocket import Rocket

class DiscretizationMethod:

    K: int              # number of discretization points
    N_sub: int          # number of substeps to approximate the ode
    range_t: tuple      # range of discretization points

    rocket: Rocket

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, rocket: Rocket, K: int, N_sub: int):
        
        # number of discretization points
        self.K = K

        # tuple containing N_sub ts between [0, 1/(K-1)] used to approximate the ODE
        self.N_sub = N_sub
        self.range_t = tuple(np.linspace(0, 1. / (self.K - 1), self.N_sub))

        self.f, self.A, self.B, self.F = rocket.get_dynamics()
        
        # number of states, inputs and parameters
        self.n_x = rocket.n_x
        self.n_u = rocket.n_u
        self.n_p = rocket.n_p

    def integrate_nonlinear_piecewise(self, X_l: NDArray, U: NDArray, p: NDArray) -> NDArray:
        pass

    def integrate_nonlinear_full(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:
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

    def __init__(self, rocket: Rocket, K: int, N_sub: int):

        super().__init__(rocket, K, N_sub)

        # x+ = A_bar(x*(k))x(k) + B_bar(x*(k))u(k) + F_bar(x*(k))p + r_bar(k)
        self.A_bar = np.zeros([self.n_x * self.n_x, self.K-1])
        self.B_bar = np.zeros([self.n_x * self.n_u, self.K-1])
        self.F_bar = np.zeros([self.n_x * self.n_p, self.K-1])
        self.r_bar = np.zeros([self.n_x, self.K-1])

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
        """
        Calculate discretization for given states, inputs and parameter matrices.

        :param X: Matrix of states at all time steps
        :param U: Matrix of inputs at all time steps
        :param p: Vector of parameters
        :return: The discretization matrices
        """

        for k in range(self.K - 1):
            self.P0[self.x_ind] = X[:, k]
            P = np.array(odeint(self._ode_dPdt, 
                                self.P0, 
                                self.range_t, 
                                args=(U[:, k], p))[-1, :])
            
            Phi = P[self.A_bar_ind].reshape((self.n_x, self.n_x))
            self.A_bar[:, k] = Phi.flatten(order='F')
            self.B_bar[:, k] = (Phi@P[self.B_bar_ind].reshape((self.n_x, self.n_u))).flatten(order='F')
            self.F_bar[:, k] = (Phi@P[self.F_bar_ind]).reshape((self.n_x, self.n_p)).flatten(order='F')
            self.r_bar[:, k] = Phi@P[self.r_bar_ind]

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
        dPdt[self.A_bar_ind] = (A_subs@P[self.A_bar_ind].reshape((self.n_x, self.n_x))).reshape(-1)
        dPdt[self.B_bar_ind] = (Phi_A_xi@B_subs).reshape(-1)
        dPdt[self.F_bar_ind] = (Phi_A_xi@F_subs).reshape(-1)
        r_t = f_subs-A_subs@x-B_subs@u-F_subs@p
        dPdt[self.r_bar_ind] = Phi_A_xi@r_t

        return dPdt

    def integrate_nonlinear_piecewise(self, X_l: NDArray, U: NDArray, p: NDArray) -> NDArray:
        """
        Piecewise integration to verfify accuracy of linearization.

        :param X_l: Linear state evolution matrix
        :param U: Linear input evolution matrix
        :param p: Vector of parameters
        :return: The piecewise integrated dynamics
        """

        X_nl = np.zeros_like(X_l)
        X_nl[:, 0] = X_l[:, 0]

        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(self._dxdt, 
                                    X_l[:, k],
                                    self.range_t,
                                    args=(U[:, k], p))[-1, :]

        return X_nl

    def integrate_nonlinear_full(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:
        """
        Simulate nonlinear behavior given an initial state and an input over time.

        :param x0: Initial state
        :param U: Linear input evolution matrix
        :param p: Vector of parameters
        :return: The full integrated dynamics
        """

        X_nl = np.zeros([x0.size, self.K])
        X_nl[:, 0] = x0

        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(self._dxdt, 
                                    X_nl[:, k],
                                    self.range_t,
                                    args=(U[:, k], p))[-1, :]

        return X_nl
    
    def integrate_nonlinear_full_dense(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:
        """
        Simulate nonlinear behavior given an initial state and an input over time.

        :param x0: Initial state
        :param U: Linear input evolution matrix
        :param p: Vector of parameters
        :return: The full integrated dynamics (with added points linked to N_subs)
        """

        X_nl = np.zeros([x0.size, U.shape[1]])
        X_nl_dense = np.zeros([x0.size, U.shape[1]+(U.shape[1]-1)*(self.N_sub-2)])
        X_nl[:, 0] = x0
        X_nl_dense[:, 0] = x0

        for k in range(U.shape[1] - 1):
            x = odeint(self._dxdt, 
                       X_nl[:, k],
                       self.range_t,
                       args=(U[:, k], p))
            X_nl[:, k+1] = x[-1, :]
            X_nl_dense[:, k*(self.N_sub-1)+1:(k+1)*(self.N_sub-1)+1] = np.array(x)[1:, :].T

        return X_nl_dense

    def _dxdt(self, x: NDArray, t: float, u: NDArray, p: NDArray) -> NDArray:
        return np.squeeze(self.f(x, u, p))
    
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
    
    def __init__(self, rocket: Rocket, K: int, N_sub: int):
        
        super().__init__(rocket, K, N_sub)

        # x+ = A_bar(x*(k))x(k) + B_plus_bar(x*(k))u(k+1) + B_minus_bar(x*(k))u(k) + F_bar(x*(k))p + r_bar(k)
        self.A_bar = np.zeros([self.n_x * self.n_x, self.K-1])
        self.B_plus_bar = np.zeros([self.n_x * self.n_u, self.K-1])
        self.B_minus_bar = np.zeros([self.n_x * self.n_u, self.K-1])
        self.F_bar = np.zeros([self.n_x * self.n_p, self.K-1])
        self.r_bar = np.zeros([self.n_x, self.K-1])
        
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

    def calculate_discretization(self, X: NDArray, U: NDArray, p: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """
        Calculate discretization for given states, inputs and parameter matrices.

        :param X: Matrix of states at all time steps
        :param U: Matrix of inputs at all time steps
        :param p: Vector of parameters
        :return: The discretization matrices
        """

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
            self.B_plus_bar[:, k] = (Phi@P[self.B_plus_bar_ind].reshape((self.n_x, self.n_u))).flatten(order='F')
            self.B_minus_bar[:, k] = (Phi@P[self.B_minus_bar_ind].reshape((self.n_x, self.n_u))).flatten(order='F')
            self.F_bar[:, k] = (Phi@P[self.F_bar_ind].reshape((self.n_x, self.n_p))).flatten(order='F')
            self.r_bar[:, k] = Phi@P[self.r_bar_ind]

        return self.A_bar, self.B_plus_bar, self.B_minus_bar, self.F_bar, self.r_bar
    
    def _ode_dPdt(self, P: NDArray, t: float, u_t0: NDArray, u_t1: NDArray, p: NDArray) -> NDArray:

        beta = (self.K-1)*t
        alpha = 1 - beta
        x = P[self.x_ind]
        u = alpha*u_t0 + beta*u_t1

        # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
        # and pre-multiplying with \Phi_A(\tau_{k+1},\tau_k) after integration
        Phi_A_xi = np.linalg.inv(P[self.A_bar_ind].reshape((self.n_x, self.n_x)))
        
        A_subs = self.A(x, u, p)
        B_subs = self.B(x, u, p)
        F_subs = self.F(x, u, p)
        f_subs = self.f(x, u, p).reshape(-1)
        
        dPdt = np.zeros_like(P)
        dPdt[self.x_ind] = f_subs.transpose()
        dPdt[self.A_bar_ind] = (A_subs@P[self.A_bar_ind].reshape((self.n_x, self.n_x))).reshape(-1)
        dPdt[self.B_plus_bar_ind] = (Phi_A_xi@B_subs).reshape(-1) * beta
        dPdt[self.B_minus_bar_ind] = (Phi_A_xi@B_subs).reshape(-1) * alpha
        dPdt[self.F_bar_ind] = (Phi_A_xi@F_subs).reshape(-1)

        r_t = f_subs-A_subs@x-B_subs@u-F_subs@p

        dPdt[self.r_bar_ind] = Phi_A_xi@r_t

        return dPdt

    def integrate_nonlinear_piecewise(self, X_l: NDArray, U: NDArray, p: NDArray) -> NDArray:
        """
        Piecewise integration to verify accuracy of linearization.

        :param X_l: Linear state evolution matrix
        :param U: Linear input evolution matrix
        :param p: Vector of parameters
        :return: The piecewise integrated dynamics
        """

        X_nl = np.zeros_like(X_l)
        X_nl[:, 0] = X_l[:, 0]

        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(self._dxdt, 
                                    X_l[:, k],
                                    self.range_t,
                                    args=(U[:, k], U[:, k + 1], p))[-1, :]

        return X_nl

    def integrate_nonlinear_full(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:
        """
        Simulate nonlinear behavior given an initial state and an input over time.

        :param x0: Initial state
        :param U: Linear input evolution matrix
        :param p: Vector of parameters
        :return: The full integrated dynamics
        """

        X_nl = np.zeros([x0.size, U.shape[1]])
        X_nl[:, 0] = x0

        for k in range(U.shape[1] - 1):
            X_nl[:, k+1] = odeint(self._dxdt, X_nl[:, k],
                                  self.range_t,
                                  args=(U[:, k], U[:, k + 1], p))[-1, :]

        return X_nl
    
    def integrate_nonlinear_full_dense(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:
        """
        Simulate nonlinear behavior given an initial state and an input over time.

        :param x0: Initial state
        :param U: Linear input evolution matrix
        :param p: Vector of parameters
        :return: The full integrated dynamics (with added points linked to N_subs)
        """

        X_nl = np.zeros([x0.size, U.shape[1]])
        X_nl_dense = np.zeros([x0.size, U.shape[1]+(U.shape[1]-1)*(self.N_sub-2)])
        X_nl[:, 0] = x0
        X_nl_dense[:, 0] = x0

        for k in range(U.shape[1] - 1):
            x = odeint(self._dxdt, 
                       X_nl[:, k],
                       self.range_t,
                       args=(U[:, k], U[:, k + 1], p))
            X_nl[:, k+1] = x[-1, :]
            X_nl_dense[:, k*(self.N_sub-1)+1:(k+1)*(self.N_sub-1)+1] = np.array(x)[1:, :].T

        return X_nl_dense

    def _dxdt(self, x: NDArray, t: float, u_t0: NDArray, u_t1: NDArray, p: NDArray) -> NDArray:
        u = u_t0 + (self.K-1)*t*(u_t1 - u_t0)
        return np.squeeze(self.f(x, u, p))