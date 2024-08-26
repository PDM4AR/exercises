import sympy as spy

from dg_commons.sim.models.rocket_structures import RocketGeometry, RocketParameters

class Rocket():

    sg: RocketGeometry
    sp: RocketParameters

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

    def __init__(self, sg: RocketGeometry, sp: RocketParameters):

        self.sg = sg
        self.sp = sp

        self.x = spy.Matrix(spy.symbols('x y psi vx vy dpsi phi m', real=True))             # states
        self.u = spy.Matrix(spy.symbols('F_l F_r dphi', real=True))                         # inputs
        self.p = spy.Matrix([spy.symbols('t_f', positive=True)])                            # final time    
        
        self.n_x = self.x.shape[0]                                                          # number of states                                        
        self.n_u = self.u.shape[0]                                                          # number of inputs
        self.n_p = self.p.shape[0]     

    def get_dynamics(self) -> tuple[spy.Function, spy.Function, spy.Function, spy.Function]:
        """
        Define dynamics and extract jacobians.
        Get dynamics for SCvx.
        """
        # Dynamics
        f = spy.zeros(self.n_x, 1)

        A = f.jacobian(self.x)
        B = f.jacobian(self.u)
        F = f.jacobian(self.p)

        f_func = spy.lambdify((self.x, self.u, self.p), f, 'numpy')
        A_func = spy.lambdify((self.x, self.u, self.p), A, 'numpy')
        B_func = spy.lambdify((self.x, self.u, self.p), B, 'numpy')
        F_func = spy.lambdify((self.x, self.u, self.p), F, 'numpy')

        return f_func, A_func, B_func, F_func