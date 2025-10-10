import sympy as spy

from dg_commons.sim.models.satellite_structures import SatelliteGeometry, SatelliteParameters


class SatelliteDyn:
    sg: SatelliteGeometry
    sp: SatelliteParameters

    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, sg: SatelliteGeometry, sp: SatelliteParameters):
        self.sg = sg
        self.sp = sp

        self.x = spy.Matrix(spy.symbols("x y psi vx vy dpsi", real=True))  # states
        self.u = spy.Matrix(spy.symbols("thrust_l thrust_r", real=True))  # inputs
        self.p = spy.Matrix([spy.symbols("t_f", positive=True)])  # final time

        self.n_x = self.x.shape[0]  # number of states
        self.n_u = self.u.shape[0]  # number of inputs
        self.n_p = self.p.shape[0]

    def get_dynamics(self) -> tuple[spy.Function, spy.Function, spy.Function, spy.Function]:
        """
        Define dynamics and extract jacobians.
        Get dynamics for SCvx.
        extract the state from self.x the following way:
        0x 1y 2psi 3vx 4vy 5dpsi
        """
        # TODO Modify dynamics
        f = spy.zeros(self.n_x, 1)  # replace this line by computing the dynamics like in following example
        "f[0] = ..."
        "f[1] = ..."
        "..."
        "f[5] = ..."

        # HINT for the dynamics: SymPy is a library for symbolic mathematics.
        # Here you’ll need symbolic math, not numerical math.
        # That means you should use SymPy functions (like sympy.sin, sympy.exp, …)
        # together with symbols, instead of numerical functions from numpy or math.

        # Jacobians and matrices of the system (don't need to change)

        # Dynamics:
        f = spy.zeros(self.n_x, 1)

        x, y, psi, vx, vy, dpsi = self.x
        thrust_l, thrust_r = self.u
        lr, m = self.sg.l_r, self.sp.m_v  # Geometry and parameter constants
        I = self.sg.Iz

        # NOTE: the comments "type: ignore" are here to avoid Pylance errors, because Pylance is unable to evaluate sympy's symbolic expressions

        # 1. Position dynamics
        f[0] = vx  # type: ignore # dx/dt
        f[1] = vy  # type: ignore # dy/dt

        # 2. Orientation dynamics
        f[2] = dpsi  # dpsi/dt

        # 3. Velocity dynamics
        f[3] = (1 / m) * spy.cos(psi) * (thrust_l + thrust_r)  # type: ignore # dvx/dt
        f[4] = (1 / m) * spy.sin(psi) * (thrust_l + thrust_r)  # type: ignore # dvy/dt

        # 4. Angular velocity dynamics
        f[5] = (lr / I) * (thrust_r - thrust_l)  # type: ignore # ddpsi/dt

        f = self.p[0] * f

        A = f.jacobian(self.x)
        B = f.jacobian(self.u)
        F = f.jacobian(self.p)

        f_func = spy.lambdify((self.x, self.u, self.p), f, "numpy")
        A_func = spy.lambdify((self.x, self.u, self.p), A, "numpy")
        B_func = spy.lambdify((self.x, self.u, self.p), B, "numpy")
        F_func = spy.lambdify((self.x, self.u, self.p), F, "numpy")

        return f_func, A_func, B_func, F_func
