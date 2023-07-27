import pygame as pg
import numpy as np
import time
import math
import scipy
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib import animation 



def mag(v):
    return math.sqrt(v[0]**2 + v[1]**2)

class Particle():
    def __init__(self) -> None:

        self.dt = 0.25
        m = 1 # kg
        p0 = [0,0]
        v0 = [0,0]

        self._x = np.array([p0,v0])
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

    # Runge-Kutta (RK4) integration method
    def solve_ivp(self, u, x0):
        k1 = self.A @ x0 + self.B @ u
        k2 = self.A @ (x0 + 0.5 * self.dt * k1) + self.B @ u
        k3 = self.A @ (x0 + 0.5 * self.dt * k2) + self.B @ u
        k4 = self.A @ (x0 + self.dt * k3) + self.B @ u

        x_next = x0 + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x_next

    def step(self, u):
        
        self._x = self.state_update(u, self._x)
        # self._x = scipy.integrate.solve_ivp(fun = self.state_update, t_span = (0,self.dt), y0=self._x, vectorized=True)

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
    def __init__(self, dt = 0.25) -> None:
        super().__init__()

        # parameters: # taken from webots mass tab
        self.g0  = 9.8066     # [m.s^2] accerelation of gravity
        self.mq  = 0.05      # [kg] total mass (with one marker)
        
        
        self.Ixx = 3.5e-5   # [kg.m^2] Inertia moment around x-axis
        self.Iyy = 3.5e-5   # [kg.m^2] Inertia moment around y-axis
        self.Izz = 6.25e-5   # [kg.m^2] Inertia moment around z-axis
        self.Cd  = 7.9379e-06 # [N/krpm^2] Drag coef
        self.Ct  = 3.25e-4    # [N/krpm^2] Thrust coef
        self.dq  = 65e-3      # [m] distance between motors' center
        self.l   = self.dq/2       # [m] distance between motors' center and the axis of rotation

        self.dt = dt
        self._x = np.zeros([0,0,0,1,0,0,0,0,0,0,0,0,0])
        
        self.N_ACTIONS = 4 # motor speeds

        self.dx = np.zeros(self.N_STATES)
        print(self.state)

    def step(self, u):
        # def s_dot_fn(t, x):
        #     return self.state_update(u,x)
        
        # sol = scipy.integrate.solve_ivp(fun=s_dot_fn, t_span=(0,self.dt), y0=self._x)
        # self._x = sol['y'][:,-1]
        # print(self._x)
        self._x = self.state_update(u, self._x)
        return self.state

    def state_update(self, u, x):
        
        _,_,_,q1, q2, q3, q4, vbx, vby, vbz, wx, wy, wz = x
        w1, w2, w3, w4 = u
        
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

        self.dx[:] = dxq,dyq,dzq,dq1,dq2,dq3,dq4,dvbx,dvby,dvbz,dwx,dwy,dwz
        
        x_next = x + self.dx*self.dt
        print(x_next)
        return x_next


class Controller:
    def __init__(self, agent=None) -> None:
        self.agent = agent
        # default
        self.action = np.zeros(self.agent.N_STATES)

    def get_action(self, state_c, state_d):
        # action = np.zeros(4)
        action = self.action
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

class MPCController(Controller):
    def __init__(self, agent) -> None:
        self.n = 20 # horizon
        self.A = agent.A
        self.B = agent.B
        self.agent = agent
        self.us = []
        self.x_init = agent.state.flatten()

        self.Q = 2*np.eye(2)
        self.R = 4*np.eye(2) 

        super().__init__(agent)

    def cost(self):
        pass

    def get_action(self, state_c, state_d):
        x = cp.Variable((self.agent.N_STATES, self.n+1))
        u = cp.Variable((self.agent.N_ACTIONS, self.n))

        cost = 0
        constraints = []
        constraints += [x[:,0] == self.x_init]
# , cp.reshape(x[:,k],shape=(2,2), order='C')
        for k in range(self.n):
            constraints += [x[:,k+1] == cp.vec(self.agent.state_update(u[np.newaxis, :,k], cp.reshape(x[:,k],shape=(2,2), order='C')), order='C')]
            constraints += [u[0,k]<=self.agent.V_MAX]
            constraints += [u[1,k]<=self.agent.V_MAX]
            constraints += [x[:2,k]>=[-1,-1]]
            constraints += [x[:2,k]<=[1,1]]

            cost += cp.quad_form(x[:2, k] - state_d[0], self.Q)
            # print(x[:2, k].value)
            # print(u[:, k,np.newaxis].shape)
            cost += cp.quad_form(u[:, k, np.newaxis], self.R)
            
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP)

        # next state in solution is init state for next plan
        self.x_init = x[:,1].value
        self.xs = x[:2,:].value
        return u[np.newaxis,:,0].value

    def predict(self):
        self.xs = []
        for i in range(self.n):
            self.xs.append(self.agent.step())


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
            # print(state)              
        
    def render(self):
        # Clear the screen
        self.screen.fill((210, 210, 210))

        for agent in self.agents:
            self.draw_agent(agent.state[0][0], agent.state[0][1])
            for x in self.controllers[0].xs.T:
                self.draw_agent(x[0], x[1], size=2)        

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
        
        super().__init__(CONTROLLER=Controller, AGENT=Crazyflie, gui=gui)
        self._x = np.zeros(13)
        self.setpoint = [0.5,0.5,0.5]
        
        self.anim = animation.FuncAnimation(self.fig, self.ani_update,
                               frames=200, interval=20, blit=True)
    def setup_window(self):
        # First set up the figure, the axis, and the plot element we want to animate
        self.fig = plt.figure()
        ax = plt.axes(projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        self.pos, = ax.plot(0,0,0, 'ro', markersize=8)
        

    def ani_update(self, i):
        # self.pos.set_data( )
        # self.pos.set_data(self.agents[0].state[0], self.agents[0].state[1])
        # self.pos.set_3d_properties(np.array([self.agents[0].state[2], np.newaxis]))
        self.step()

        self.pos.set_data_3d(self.agents[0].state[0], self.agents[0].state[1], self.agents[0].state[2])
        return self.pos,

    # def render(self):
    #     self.anim

    def run(self):
        while True:
            self.step()
            # if self.gui:
            #     self.render()


if __name__=="__main__":
    env = Env(gui = True)
    env.run()
    # plt.show()
