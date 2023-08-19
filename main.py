import pygame as pg
import numpy as np
import time
import math
import scipy
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib import animation 
import casadi as ca
# import keyboard
import readchar
import threading
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver, AcadosOcpSolver, AcadosOcp


def create_model(agent):
    name = "particle_ode"
    # Create the CasADi symbols for the function definition

    # states
    x = ca.MX.sym('x', agent.N_STATES)

    # controls
    u = ca.MX.sym('u', agent.N_ACTIONS)

    # dynamics model
    xdot = ca.MX.sym('xdot', agent.N_STATES)

    # explicit expression. the actual derivative xdot = f(x,u)
    f_expl = ca.vertcat(*agent.xdot(x,u))
    
    model = AcadosModel()
    model.x = x
    model.xdot = xdot
    
    # this expression is set to 0 and solved. basically just saying that the explicit expression is just = xdot. extra steps
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.u = u
    model.name = name
    return model

def mag(v):
    return math.sqrt(v[0]**2 + v[1]**2)

class Particle():
    def __init__(self, dt=0.25, init_state = np.array([0,0, 0, 0])) -> None:
        m = 1 # kg
        p0 = [0,0]
        v0 = [0,0]
        self.dt = dt
        self._x = init_state
        self.N_STATES = len(self._x.flatten())
        self.N_ACTIONS = 2
        # discrete state space matrices
        self.A = np.array([[1, self.dt],
                           [0, 1]])
        self.B = np.array([[0, self.dt/m]]).T
        
        self.C = np.zeros((len(self._x), len(self._x)))
        self.D = np.zeros((1, 1))

        # Bounds for particle movement
        self.X_MIN, self.X_MAX = -1, 1
        self.Y_MIN, self.Y_MAX = -1, 1

        self.V_MAX = 0.2

        self.u_old = np.array([v0])

    def state_update(self, u, x_old):
        x_next = self.A @ x_old + self.B @ u
        return x_next  

    def xdot(self, x, u):
        return [x[2], x[3], u[0], u[1]]

    def xdot_scipy(self, t, x, u):
        return self.xdot(x,u)


    # Runge-Kutta (RK4) integration method
    def solve_ivp(self, u, x0):
        k1 = self.A @ x0 + self.B @ u
        k2 = self.A @ (x0 + 0.5 * self.dt * k1) + self.B @ u
        k3 = self.A @ (x0 + 0.5 * self.dt * k2) + self.B @ u
        k4 = self.A @ (x0 + self.dt * k3) + self.B @ u

        x_next = x0 + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x_next

    def step(self, u):
        
        # self._x = self.state_update(u, self._x)
        sol = scipy.integrate.solve_ivp(fun = self.xdot_scipy, t_span = (0,self.dt), y0=self._x, vectorized=True, args = (u,))
        self._x = sol['y'][:,-1]
        # self._x = self._x + self.dt*np.array(self.xdot(self._x, u))
        # self.housekeeping()
        
        return self.state

    def housekeeping(self):

        # Reflect off walls]
        if self._x[0][0] <= self.X_MIN or self._x[0][0] >= self.X_MAX:
            self._x[1][0] *= -1
            self._x[0][0] = max(self.X_MIN, min(self.X_MAX, self._x[0][0]))  # Keep particle inside bounds
        
        if self._x[0][1] <= self.Y_MIN or self._x[0][1] >= self.Y_MAX:
            self._x[1][1] *= -1
            self._x[0][1] = max(self.Y_MIN, min(self.Y_MAX, self._x[0][1]))  # Keep particle inside bounds

    @property
    def state(self):
        return self._x

class Sensor:
    def __init__(self) -> None:
        self.measurement = np.array([])

    def add_noise(self, mu, sigma):
        self.measurement += mu + np.random.randn(self.measurement.shape) * sigma

class Ranger(Sensor):
    def __init__(self, max_dist, map) -> None:
        self.max_dist = max_dist
        self.map = map # list of obstacles
        super().__init__()

    # def measure(self, pos):
    #     for obs in map:
            
    #         self.measurement = mag(pos-)

class Crazyflie(Particle):
    def __init__(self, dt = 0.015, init_state = np.array([0,0,0,1,0,0,0,0,0,0,0,0,0])) -> None:
    
        super().__init__(dt=dt,init_state = init_state)
    

        # parameters: # taken from webots mass tab
        self.g0  = 9.8066     # [m.s^2] accerelation of gravity
        self.mq  = 0.033      # [kg] total mass (with one marker)
        
        
        self.Ixx = 3.5e-5   # [kg.m^2] Inertia moment around x-axis
        self.Iyy = 3.5e-5   # [kg.m^2] Inertia moment around y-axis
        self.Izz = 6.25e-5   # [kg.m^2] Inertia moment around z-axis
        self.Cd  = 7.9379e-06 # [N/krpm^2] Drag coef
        self.Ct  = 3.25e-4    # [N/krpm^2] Thrust coef
        self.dq  = 65e-3      # [m] distance between motors' center
        self.l   = self.dq/2       # [m] distance between motors' center and the axis of rotation
        
        self.N_ACTIONS = 4 # motor speeds

        self.dx = np.zeros(self.N_STATES)     

    def xdot(self, _x, u):
        
        # x,y,z,q1, q2, q3, q4, vbx, vby, vbz, wx, wy, wz = x
        # a = x[0]
        x = _x[0]
        y = _x[1]
        z = _x[2]
        q1 = _x[3]
        q2 = _x[4]
        q3 = _x[5]
        q4 = _x[6]
        vbx = _x[7]
        vby = _x[8]
        vbz = _x[9]
        wx = _x[10]
        wy = _x[11]
        wz = _x[12]

        w1 = u[0]
        w2 = u[1]
        w3 = u[2]
        w4 = u[3]

        # w1, w2, w3, w4 = u

        # print
        dxq = vbx*(2*q1**2 + 2*q2**2 - 1) - vby*(2*q1*q4 - 2*q2*q3) + vbz*(2*q1*q3 + 2*q2*q4)
        dyq = vby*(2*q1**2 + 2*q3**2 - 1) + vbx*(2*q1*q4 + 2*q2*q3) - vbz*(2*q1*q2 - 2*q3*q4)
        dzq = vbz*(2*q1**2 + 2*q4**2 - 1) - vbx*(2*q1*q3 - 2*q2*q4) + vby*(2*q1*q2 + 2*q3*q4)
        dq1 = - (q2*wx)/2 - (q3*wy)/2 - (q4*wz)/2
        dq2 = (q1*wx)/2 - (q4*wy)/2 + (q3*wz)/2
        dq3 = (q4*wx)/2 + (q1*wy)/2 - (q2*wz)/2
        dq4 = (q2*wy)/2 - (q3*wx)/2 + (q1*wz)/2
        dvbx = vby*wz - vbz*wy + self.g0*(2*q1*q3 - 2*q2*q4)
        dvby = vbz*wx - vbx*wz - self.g0*(2*q1*q2 + 2*q3*q4)
        dvbz = vbx*wy - vby*wx - self.g0*(2*q1**2 + 2*q4**2 - 1) + (self.Ct*(w1**2 + w2**2 + w3**2 + w4**2))/self.mq
        dwx = -(self.Ct*self.l*(w1**2 + w2**2 - w3**2 - w4**2) - self.Iyy*wy*wz + self.Izz*wy*wz)/self.Ixx
        dwy = -(self.Ct*self.l*(w1**2 - w2**2 - w3**2 + w4**2) + self.Ixx*wx*wz - self.Izz*wx*wz)/self.Iyy
        dwz = -(self.Cd*(w1**2 - w2**2 + w3**2 - w4**2) - self.Ixx*wx*wy + self.Iyy*wx*wy)/self.Izz

        # print(dxq,dyq,dzq,dq1, dq2,dq3,dq4,dvbx,dvby,dvbz,dwx,dwy,dwz)
        # self.dx = cp.hstack([dxq,dyq,dzq,dq1,dq2,dq3,dq4,dvbx,dvby,dvbz,dwx,dwy,dwz])
        # print(self.dx)
        # x_next = x + self.dx*self.dt

        # x += dxq*self.dt
        # y += dyq*self.dt
        # z += dzq*self.dt
        # q1 += dq1*self.dt
        # q2 += dq2*self.dt
        # q3 += dq3*self.dt
        # q4 += dq4*self.dt
        # vbx += dvbx*self.dt
        # vby += dvby*self.dt
        # vbz += dvbz*self.dt
        # wx += dwx*self.dt
        # wy += dwy*self.dt
        # wz += dwz*self.dt
        
        # print(x_next)
        return [dxq,dyq,dzq,dq1,dq2,dq3,dq4,dvbx,dvby,dvbz,dwx,dwy,dwz]

class Controller:
    def __init__(self, agent=None) -> None:
        self.agent = agent
        # default
        self.action = np.zeros(self.agent.N_ACTIONS)

    def get_action(self, state_c, state_d):
        # action = np.zeros(4)
        action = self.action
        action = 19.421043305572518*np.ones(self.agent.N_ACTIONS)
        return action

class PIDController(Controller):
    def __init__(self, agent=None) -> None:
        self.p = 1
        self.i = 0
        self.d = 6
        self.err_pos = np.zeros(2)
        
        super().__init__()

    def get_action(self, state_c, state_d):
        err_pos = (state_d[0]-state_c[0])
        err_vel = (state_d[1]-state_c[1])
        self.err_pos += err_pos
        vel = self.p*err_pos + self.d*err_vel + self.i*self.err_pos 
        return vel[np.newaxis,:]

class LQRController(Controller):
    def __init__(self, agent) -> None:
        super().__init__()
        self.A = agent.A
        self.B = agent.B

        self.Q = 1*np.eye(2)
        self.R = 1*np.eye(1)

        self.N = 50

    def estimate_disc_P(self):
        """Recursively estimate the steady state feedback gain from DARE solution"""
        P = self.Q
        A = self.A
        B = self.B
        for _ in reversed(range(self.N)):
            P =  A.T@P@A - (A.T@P@B)@(np.linalg.inv(self.R + B.T@P@B)@(B.T@P@A)) + self.Q

        return P

    def solve_cont_P(self):
        """Solve the continuous time ARE"""
        return scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)

    def get_action(self, state_c, state_d):
        # continuous time LQR control law
        # the relevant state is the error 
        vel = np.linalg.inv(self.R)@self.B.T@self.solve_cont_P()@(state_d-state_c)
        
        return vel 

class KeyController(Controller):
    def __init__(self, agent=None) -> None:
        super().__init__(agent)
        t = threading.Thread(target = self.key_update)
        t.start()
        self.key = ""
        self.hover_action = [19.421043305572518,19.421043305572518,19.421043305572518,19.421043305572518]

    def key_update(self):
        scaling = 10000000
        while True:
            self.key = readchar.readkey()
            # self.hover_action = [19.44,19.44,19.44,19.44]
            if self.key == readchar.key.UP:
                self.hover_action[1] += 0.5/scaling  # Set the linear velocity forward
                self.hover_action[0] += 0.5/scaling
                self.hover_action[2] -= 0.5/scaling
                self.hover_action[3] -= 0.5/scaling
                print("up")
            elif self.key == readchar.key.DOWN:
                self.hover_action[1] -= 0.5/scaling  # Set the linear velocity forward
                self.hover_action[0] -= 0.5/scaling
                self.hover_action[2] += 0.5/scaling
                self.hover_action[3] += 0.5/scaling
                print("down")
            elif self.key == readchar.key.LEFT:
                self.hover_action[1] -= 0.5/scaling  # Set the linear velocity forward
                self.hover_action[2] -= 0.5/scaling
                self.hover_action[0] += 0.5/scaling
                self.hover_action[3] += 0.5/scaling
                print("left")

            elif self.key == readchar.key.RIGHT:
                self.hover_action[1] += 0.5/scaling  # Set the linear velocity forward
                self.hover_action[2] += 0.5/scaling
                self.hover_action[0] -= 0.5/scaling
                self.hover_action[3] -= 0.5/scaling
                print("right")
        # elif key == b'a':
        #     cmd_vel.angular.z = 0.5  # Set the angular velocity left
        # elif key == b'd':
        #     cmd_vel.angular.z = -0.5  #
        
            self.key=""

            time.sleep(0.1)
            self.hover_action = [19.421043305572518,19.421043305572518,19.421043305572518,19.421043305572518]

    def get_action(self, state_c, state_d):
        
        print(self.hover_action)
        return self.hover_action

class MPCController(Controller):
    def __init__(self, agent) -> None:
        self.n = 20 # horizon
        self.A = agent.A
        self.B = agent.B
        self.agent = agent
        self.us = []
        self.x_init = agent.state.flatten()
        
        self.Q = 10*ca.DM(np.eye(3))  # State cost matrix
        self.R = 0.02*ca.DM(np.eye(self.agent.N_ACTIONS))  # Control cost matrix

        super().__init__(agent)

        self.n_state_vars = self.agent.N_STATES*(self.n+1)
        self.n_act_vars = self.agent.N_ACTIONS*self.n 
        n_constraints = self.agent.N_STATES*self.n

        # Preform bounds to save computation

        # bounds on the constraints. lbg = ubg means these are equality constraints. Basically used for  
        self.lbg = [0 for _ in range(n_constraints)]
        self.ubg = [0 for _ in range(n_constraints)]
        
        u_min = 0
        u_max = 22

        # Bounds on the states. Here, the controls, positions and velocities are all bounded between -1,1. This is kinda forced but it speeds up the MPC
        #     x, y, z, q1, q2, q3, q4, vbx, vby, vbz, wx, wy, wz, u1, u2, u3, u4
        lb_state = [-20, -20, -20, -1, -1, -1, -1 , -2, -2, -2, 0, 0, 0]
        ub_state = [20, 20, 20, 1, 1, 1, 1 , 2, 2, 2, ca.inf, ca.inf, ca.inf]
        
        lb_action = [u_min, u_min, u_min, u_min]
        ub_action = [u_max, u_max, u_max, u_max]

        self.lbx = []
        self.ubx = []
        for i in range(self.n+1):
            self.lbx.extend(lb_state)
            self.ubx.extend(ub_state)

        for i in range(self.n):
            self.lbx.extend(lb_action)
            self.ubx.extend(ub_action)
            

        # self.lbx = [*lb for _ in range((self.agent.N_STATES+self.agent.N_ACTIONS)*self.n + self.agent.N_STATES)]
        # self.ubx = [1 for _ in range((self.agent.N_STATES+self.agent.N_ACTIONS)*self.n + self.agent.N_STATES)]

        # sets up a casadi based function as the model. it is almost always faster than pure python funcitons
        self.model = self.setup_model(agent)
        self.setup_problem()

    def get_action(self, state_c, state_d):
        x = cp.Variable((self.agent.N_STATES, self.n+1))
        u = cp.Variable((self.agent.N_ACTIONS, self.n))

        cost = 0
        constraints = []
        constraints += [x[:,0] == self.x_init]

        for k in range(self.n):
            constraints += [x[:,k+1] == cp.hstack(self.agent.state_update(u[:,k], x[:,k]))]
            constraints += [u[0,k]<=self.agent.V_MAX]
            constraints += [u[1,k]<=self.agent.V_MAX]
            constraints += [x[:2,k]>=[-1,-1]]
            constraints += [x[:2,k]<=[1,1]]

            cost += cp.quad_form(x[:3, k] - state_d[:3], self.Q)
            # print(x[:2, k].value)
            # print(u[:, k,np.newaxis].shape)
            cost += cp.quad_form(u[:, k, np.newaxis], self.R)
            
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP)

        # next state in solution is init state for next plan
        self.x_init = x[:,1].value
        self.x_opt = x[:2,:].value
        return u[np.newaxis,:,0].value

    def cost(self, x, u, xt):
        j = (x[:3] - xt).T @ self.Q @ (x[:3] - xt) + u.T @ self.R @ u
        # print(j)
        # ((x[0]-0.5)**2) * self.Q + u*self.R*u
        return j

    def setup_model(self, agent):
        """Create a casadi function for the dynamics model. This is usually faster than normal python functions"""

        # Create the CasADi symbols for the function definition
        x = ca.MX.sym('x', self.agent.N_STATES)
        u = ca.MX.sym('u', self.agent.N_ACTIONS)
        
        x_next = x + self.agent.dt * ca.vertcat(*agent.xdot(x,u))
        print(x_next)
        model = ca.Function('F', [x, u], [x_next], ['x0', 'u'], ['x'])

        return model

    def get_action_casadi(self, state_c, state_d):
        start = time.time()
        # RAW METHOD: faster but not readable, 0.021s onn average for 2nd degree particle sim
        Xk = ca.MX.sym('X0', self.agent.N_STATES, self.n+1) # agent state
        Uk = ca.MX.sym('Uk', self.agent.N_ACTIONS, self.n) # agent controls

        constraints = []
        
        # set initial state the same as upper and lower bound. Has the same effect as an equality constraint
        self.lbx[:self.agent.N_STATES] = self.x_init.copy()
        self.ubx[:self.agent.N_STATES] = self.x_init.copy()

        objective = 0

        for k in range(self.n):

            Fk = self.model(x0=Xk[:,k], u=Uk[:,k]) # predict next state and cost
            Xk_pred = Fk['x']
            objective += self.cost(Xk[:,k], Uk[:,k], state_d)   # quadratic cost of the formulations

            constraints += [Xk[:,k+1] - Xk_pred] # constrain next state to be from the dynamic model

        # reshape matrices to combine into one full state vector
        Xk_rs = ca.reshape(Xk, 1, self.agent.N_STATES*(self.n + 1))
        Uk_rs = ca.reshape(Uk, 1, self.agent.N_ACTIONS*self.n)

        # The Non linear program 
        nlp = {'x': ca.horzcat(Xk_rs, Uk_rs), 'f': objective, 'g':ca.vertcat(*constraints)}
        opts = {'ipopt.tol': 1e-4, 'print_time': 0, 'ipopt.print_level': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(lbg=self.lbg, ubg=self.ubg, lbx=self.lbx, ubx=self.ubx)
        
        action = sol['x'][self.n_state_vars:self.n_state_vars + self.agent.N_ACTIONS] # get only the first control action
        print(action)
        self.x_opt = ca.reshape(sol['x'][:self.n_state_vars], self.agent.N_STATES, self.n+1)
    
        self.x_init = list(self.x_opt[:,1].full()[:,0])
        # print(self.x_init)
        print(time.time()-start)
        return action.full()[0]

    def setup_problem(self):
        self.opti = ca.Opti()
        objective = 0
        
        self.Xk = self.opti.variable(self.agent.N_STATES, self.n+1)
        self.Uk = self.opti.variable(self.agent.N_ACTIONS, self.n)

        self.x0 = [self.opti.parameter() for _ in range(self.agent.N_STATES)]
        self.xt = [self.opti.parameter() for _ in range(3)]

        self.opti.subject_to(self.Xk[:,0] == ca.vertcat(*self.x0))

        for k in range(self.n):
            Fk = self.model(x0=self.Xk[:,k], u=self.Uk[:,k]) # predict next state
            Xk_pred = Fk['x']
            # objective += Fk['j']
            objective += self.cost(self.Xk[:,k], self.Uk[:,k], ca.vertcat(*self.xt))
            
            self.opti.subject_to(self.Xk[:,k+1] == Xk_pred)
            # self.opti.subject_to(self.opti.bounded(-1, self.Uk[:,k], 1))
            # self.opti.subject_to(self.opti.bounded(-1, self.Xk[:,k], 1))

            self.opti.subject_to(self.opti.bounded(0, self.Uk[:,k], 30))
            self.opti.subject_to(self.opti.bounded(-10, self.Xk[:3,k], 10))  # position constraint
            # self.opti.subject_to(self.opti.bounded(-1, self.Xk[7:10,k], 1))
            # self.opti.subject_to(self.Xk[10:,k] >= 0)
        # print(objective)
        p_opts = {"expand":False, "print_time":0}
        s_opts =  {'tol': 1e-4, 'print_level': 0}
        self.opti.minimize(objective)

        self.opti.solver('ipopt', p_opts,  s_opts)

    def get_action_casadi_opti(self, state_c, state_d):
        ## ========= OPTI METHOD: SLOWER but readble,  0.035s onn average for 2nd degree particle sim
        start = time.time()
        # set initial state
        for i, x0_i in enumerate(self.agent.state):
            self.opti.set_value(self.x0[i], x0_i)

        # set objective setpoint
        for i, xt_i in enumerate(state_d[0]):
            self.opti.set_value(self.xt[i], xt_i)
        
        sol = self.opti.solve()

        control_opt = sol.value(self.Uk)[:,0]
        print(control_opt)
        # self.x_init = sol.value(self.Xk)[:,1]
        # print(self.x_init)
        print(time.time()-start)

        return control_opt

    def predict(self):
        self.xs = []
        for i in range(self.n):
            self.xs.append(self.agent.step())

class MPCControllerAcados(Controller):
    def __init__(self, agent=None) -> None:
        super().__init__(agent)

        ocp = AcadosOcp()
        model = create_model(agent)
        ocp.model = model

        nx = ocp.model.x.size()[0]
        nu = ocp.model.u.size()[0]
        N = 50
        Tf = self.agent.dt*N
        
        # weight matrices
        Q_mat = 100*np.diag([50, 50, 5000, 1e-3, 1e-3, 1e-3, 1e-3, 5, 5, 10, 1e-5, 1e-5, 10])
        # Q_mat = 1*np.diag([120, 100, 100, 1e-3, 1e-3, 1e-3, 1e-3, 0.7, 1, 4, 1e-5, 1e-5, 10])
        R_mat = 0.01*np.diag([1e-2, 1e-2, 1e-2, 1e-2])

        setpoint = np.array([0, 0, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.x0 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        u0 = np.array([0, 0, 0, 0])

        # set simulation time

        ocp.dims.N = N
        # set options
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.num_stages = 4 # RK4 integration
        ocp.solver_options.num_steps = 3
        ocp.solver_options.newton_iter = 3 # for implicit integrator
        ocp.solver_options.collocation_type = "GAUSS_RADAU_IIA"


        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        # path cost
        ocp.model.cost_expr_ext_cost = (model.x-setpoint).T @ Q_mat @ (model.x-setpoint) + model.u.T @ R_mat @ model.u
        # terminal cost
        ocp.model.cost_expr_ext_cost_e = (model.x-setpoint).T@ (Q_mat) @ (model.x-setpoint)

        # ocp.model.yref   = np.array([0, 0, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, agent.hov_w, agent.hov_w, agent.hov_w, agent.hov_w])
        # ocp.model.yref_e = np.array([0, 0, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        Fmax = 22
        Fmin = 0
        ocp.constraints.lbu = np.array([Fmin, Fmin, Fmin, Fmin])
        ocp.constraints.ubu = np.array([+Fmax, +Fmax, Fmax, Fmax])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # set options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
        # ocp.solver_options.print_level = 1
        ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP
        ocp.solver_options.print_level = 0 # SQP_RTI, SQP
        ocp.constraints.x0 = self.x0

        ocp.solver_options.tf = Tf

        self.solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')

    def get_action(self, state_c, state_d):
        # acados_integrator.set("x", x0)
        # acados_integrator.set("u", u0)

        start = time.time()
        self.solver.set(0, "lbx", self.agent.state)
        self.solver.set(0, "ubx", self.agent.state)
        
        status = self.solver.solve()
        # print(time.time()-start)

        # acados_integrator.solve()
        
        # getting the state from a separate integrator rather than the perfect MPC prediction
        # x0 = acados_integrator.get("x")

        # the perfect MPC predicted state. not sure which one to use.
        # self.x0 = self.solver.get(1, "x")
        u0 = self.solver.get(0, "u")
        print(u0)
        return u0

class Env():
    def __init__(self, AGENT=Particle, CONTROLLER=MPCController,  gui=True) -> None:
        self.NUM_AGENTS = 1
        
        self.agents = [AGENT() for i in range(self.NUM_AGENTS)]
        self.controllers = [CONTROLLER(self.agents[i]) for i in range(self.NUM_AGENTS)]
        self.WIDTH, self.HEIGHT = 1000, 1000

        # Particle properties
        
        self.PARTICLE_RADIUS = 5
        self.PARTICLE_SPEED = 2
        self.PARTICLE_COLOR = (255,0,0)

        if gui:
            self.setup_window()

        self.gui = gui
        self.running = True

        self.setpoint = np.array([
            [0.5,-0.5],
            [0, 0],
        ])

    def setup_window(self):
    
        # Initialize Pygame
        pg.init()
        self.screen = pg.display.set_mode((self.WIDTH, self.HEIGHT))
        self.coyote = pg.transform.scale(pg.image.load("coyote.png").convert_alpha(), (170,160))
        self.roadrunner = pg.transform.scale(pg.image.load("roadrunner.png").convert_alpha(), (170,130))
        pg.display.set_caption("Particle Simulation")

    def draw_agent(self, x,y, size=5):

        x_pixel = int((x + 1) * self.WIDTH / 2)
        y_pixel = int((1 - y) * self.HEIGHT / 2)
        pg.draw.circle(self.screen, self.PARTICLE_COLOR, (x_pixel, y_pixel), size)

    def step(self):
        for i in range(self.NUM_AGENTS):
            
            control_action = self.controllers[i].get_action(self.agents[i].state, self.setpoint)

            state = self.agents[i].step(control_action)  
            print(state)              
        
    def render(self):
        # Clear the screen
        self.screen.fill((210, 210, 210))

        for agent in self.agents:
            self.draw_agent(agent.state[0], agent.state[1])
            # for x in self.controllers[0].x_opt.T:
            #     self.draw_agent(x[0], x[1], size=2)        

    def run(self):
        clock = pg.time.Clock()
        pos_px = self.setpoint[0]
        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                elif event.type == pg.MOUSEMOTION:
                    pos_px  = pg.mouse.get_pos()
                    self.setpoint[0,0] = (2 * pos_px[0] / self.WIDTH)-1
                    self.setpoint[0,1] = 1-(2 * pos_px[1] /self.HEIGHT)
            
            self.step()
            if self.gui:
                self.render()
            pg.display.flip()
            clock.tick(120)


class DroneEnv(Env):
    def __init__(self, gui=True) -> None:
        
        super().__init__(CONTROLLER=MPCControllerAcados, AGENT=Crazyflie, gui=gui)
        # self._x = np.zeros(13)
        self.setpoint = [[0,5,10]]
        
        self.anim = animation.FuncAnimation(self.fig, self.ani_update, frames=200, interval=35, blit=True)
        # t = threading.Thread(target=self.run)
        # t.start()
        # t.join()

    def setup_window(self):
        # First set up the figure, the axis, and the plot element we want to animate
        self.fig = plt.figure()
        ax = plt.axes(projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(-10, 10))
        self.pos, = ax.plot(0,0,0, 'ro', markersize=8)
        

    def ani_update(self, i):
        # self.pos.set_data( )
        # self.pos.set_data(self.agents[0].state[0], self.agents[0].state[1])
        # self.pos.set_3d_properties(np.array([self.agents[0].state[2], np.newaxis]))
        self.step()
        # print(self.agents[0].state[2])
        self.pos.set_data_3d(self.agents[0].state[0], self.agents[0].state[1], self.agents[0].state[2])
        return self.pos,

    # def render(self):
    #     self.anim

    def run(self):
        while True:
            self.step()
            # if self.gui:
            #     self.render()
            time.sleep(0.001)


if __name__=="__main__":
    env = DroneEnv(gui = True)
    # env.run()
    plt.show()
