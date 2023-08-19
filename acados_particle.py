import casadi as ca
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver, AcadosOcpSolver, AcadosOcp
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import animation
import scipy

np.set_printoptions(precision=5)

class Particle:
    def __init__(self) -> None:
        self.N_STATES = 4
        self.N_ACTIONS = 2 # motor speeds

        self.m = 1

    def xdot(self, x,u):
        # return ca.vertcat(X[1], U) 
        return [x[2], x[3], u[0]/self.m, u[1]/self.m]
    
class Crazyflie():
    def __init__(self) -> None:

        # parameters: # taken from webots mass tab
        self.g0  = 9.8066     # [m.s^2] accerelation of gravity
        self.mq  = 0.033      # [kg] total mass (with one marker)
        
        
        self.Ixx = 3.5e-5   # [kg.m^2] Inertia moment around x-axis
        self.Iyy = 3.5e-5   # [kg.m^2] Inertia moment around y-axis
        self.Izz = 6.25e-5   # [kg.m^2] Inertia moment around z-axis
        self.Cd  = 7.9379e-06 # [N/krpm^2] Drag coef
        self.Ct  = 0.00221472    # [N/krpm^2] Thrust coef
        self.dq  = 65e-3      # [m] distance between motors' center
        self.l   = self.dq/2       # [m] distance between motors' center and the axis of rotation
        self.N_STATES = 12
        self.N_ACTIONS = 4 # motor speeds

        self.dx = np.zeros(self.N_STATES)
        
        self.hov_w = np.sqrt(self.mq*self.g0/4/self.Ct)

        # linear roll pitch model
        self.A = np.zeros((self.N_STATES, self.N_STATES))
        self.A[:self.N_STATES//2, self.N_STATES//2:] = np.eye(self.N_STATES//2)
        self.A[7,3] = -self.g0
        self.A[6,4] = self.g0

        self.B = np.zeros((self.N_STATES, self.N_ACTIONS))
        self.B[8] = (2*self.Ct/self.mq) * np.array([1,1,1,1])
        self.B[11] = (2*self.Cd/self.Izz) * np.array([-1,1,-1,1])
        self.B[10] = (self.dq*self.Ct/self.Iyy) * np.array([-1,1,1,-1])
        self.B[9] = (self.dq*self.Ct/self.Ixx) * np.array([-1,-1,1,1])

        self.B *= self.hov_w 
        print(self.A)
        print(self.B)

        # print(self.state)
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

    def xdot_linear(self, _x, u):
        return self.A @ _x + self.B @ u


# Define the dynamics model using CasADi symbolic variables
def x_dot(X,U):
    # return ca.vertcat(X[1], U) 
    return [X[1], U]

def dynamics(X,U):
    return X + dt*ca.vertcat(*x_dot(X,U))

def cost(X,U):
    j = ((X[0]-0.5)**2) * Q + U*R*U
    # j = ca.mtimes([(x[0]-xt).T, Q, (x[0]-xt)]) + u*R*u#ca.mtimes(u.T, R, u)
    return j


def export_model(agent):
    name = "particle_ode"

    # Create the CasADi symbols for the function definition

    # constants
    m = 2 # mass in kg

    # states
    x = ca.MX.sym('x', agent.N_STATES)

    # controls
    u = ca.MX.sym('u', agent.N_ACTIONS)

    # dynamics model
    xdot = ca.MX.sym('xdot', agent.N_STATES)

    # explicit expression. the actual derivative xdot = f(x,u)
    # f_expl = ca.vertcat(*agent.xdot_linear(x,u))

    f_expl = agent.xdot_linear(x,u)
    
    model = AcadosModel()
    model.x = x
    model.xdot = xdot
    
    # this expression is set to 0 and solved. basically just saying that the explicit expression is just = xdot. extra steps
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.u = u
    model.name = name
    return model


# Simulate the MPC trajectory
num_time_steps = 200


ts = 0.03 # (s) the integration/discretization sampling time


ocp = AcadosOcp()
agent = Crazyflie()
# agent = Particle()
model = export_model(agent)
ocp.model = model


nx = ocp.model.x.size()[0]
nu = ocp.model.u.size()[0]
N = 50
Tf = N*ts


# weight matrices
# Q_mat = 100*np.diag([50, 50, 5000, 1e-3, 1e-3, 1e-3, 1e-3, 5, 5, 10, 1e-5, 1e-5, 10])
# Q_mat = 100*np.diag([2000, 2000, 5000, 1e-3, 1e-3, 1e-3, 5, 5, 10, 1e-5, 1e-5, 10])
Q_mat = 10*np.diag([120, 100, 100, 1e-3, 1e-3, 1e-3, 4,4, 10, 1e-5, 1e-5, 10])
R_mat = 0.1*np.diag([1e-2, 1e-2, 1e-2, 1e-2])


setpoint = np.array([0, 5, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0])

x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# u0 = np.array([19.42,19.42, 19.42, 19.42])

u0 = np.array([0, 0, 0, 0])
# Q_mat = 2*np.diag([1e3, 1e3, 1e-2, 1e-2])
# R_mat = 2*np.diag([1e-2, 1e-2])

# setpoint = np.array([0.8, 0.1,0,0])

# x0 = np.array([0, 0, 0, 0])
# u0 = np.array([0,0])



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
# ocp.

# path cost
ocp.model.cost_expr_ext_cost = (model.x-setpoint).T @ Q_mat @ (model.x-setpoint) + model.u.T @ R_mat @ model.u
# terminal cost
ocp.model.cost_expr_ext_cost_e = (model.x-setpoint).T@ (50*Q_mat) @ (model.x-setpoint)

# ocp.model.yref   = np.array([0, 0, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, agent.hov_w, agent.hov_w, agent.hov_w, agent.hov_w])
# ocp.model.yref_e = np.array([0, 0, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])


Fmax = 10
Fmin = -10
ocp.constraints.lbu = np.array([Fmin, Fmin, Fmin, Fmin])
ocp.constraints.ubu = np.array([+Fmax, +Fmax, Fmax, Fmax])
ocp.constraints.idxbu = np.array([0, 1, 2, 3])



# Fmax = 1
# Fmin = 0
# ocp.constraints.lbu = np.array([-Fmax, -Fmax])
# ocp.constraints.ubu = np.array([+Fmax, +Fmax])
# ocp.constraints.idxbu = np.array([0, 1])

# set options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
# PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
# PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
# ocp.solver_options.print_level = 1
ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP
ocp.solver_options.print_level = 0 # SQP_RTI, SQP
ocp.constraints.x0 = x0
# ocp.solver_options.qp_solver_cond_N = N
ocp.solver_options.tf = Tf

solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')


# SIMULATION

sim = AcadosSim()
# set model_name
sim.model = model

# set simulation time
sim.solver_options.T = ts
# set options
sim.solver_options.integrator_type = 'ERK'
sim.solver_options.num_stages = 4
sim.solver_options.num_steps = 3
sim.solver_options.newton_iter = 3 # for implicit integrator
sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"

# create
acados_integrator = AcadosSimSolver(sim)
acados_integrator.set("u", u0)
acados_integrator.set("x", x0)
simX = np.zeros((N+1, nx))


# animation
fig = plt.figure()
ax = plt.axes(projection='3d', xlim=(-10, 10), ylim=(-10, 10), zlim=(-10, 10))
pos, = ax.plot(0,0,0, 'bo', markersize=8)
ax.plot(setpoint[0], setpoint[1],setpoint[2], 'rx', markersize=5)
x_pred_plots = [ax.plot(0,0,0, 'k.', markersize=4)[0] for _ in range(N)]

def ani_update(i):

    # the perfect MPC predicted state. not sure which one to use.
    status = solver.solve()
    
    u0 = solver.get(0, "u")
    # u0=np.array([15,15,10,10])

    # u0=np.ones(4)*agent.hov_w
    acados_integrator.set("u", u0)
    acados_integrator.solve()
    sim_x = acados_integrator.get("x")
    
    # sol = scipy.integrate.solve_ivp(fun=agent.xdot, y0=sim_x, t_span = (0,0.1), args=(u0,))
    # sim_x = sol['y'][:,-1]
    solver.set(0, "lbx", sim_x)
    solver.set(0, "ubx", sim_x)
    acados_integrator.set("x", sim_x)

    # getting the state from a separate integrator rather than the perfect MPC prediction
    
    pos.set_data_3d(sim_x[0], sim_x[1], sim_x[2])

    for i in range(N):
        x_pred = solver.get(i, "x")
        x_pred_plots[i].set_data_3d(x_pred[0], x_pred[1], x_pred[2])

    # print(sim_x)
    return pos, *x_pred_plots


anim = animation.FuncAnimation(fig, ani_update, frames=200, interval=35, blit=True)

plt.show()
# for i in range(num_time_steps):
# while True:
#     # acados_integrator.set("x", x0)
#     # acados_integrator.set("u", u0)

#     start = time.time()
#     solver.set(0, "lbx", x0)
#     solver.set(0, "ubx", x0)
    
#     status = solver.solve()
#     # print(time.time()-start)

#     # acados_integrator.solve()
    
#     # getting the state from a separate integrator rather than the perfect MPC prediction
#     # x0 = acados_integrator.get("x")

#     # the perfect MPC predicted state. not sure which one to use.
#     x0 = solver.get(1, "x")
#     u0 = solver.get(0, "u")
#     print(u0)
#     print(x0)


# xi = [[0,0]]

# lbg = [0 for _ in range(2*(N))]
# ubg = [0 for _ in range(2*(N))]
# lbx = [-1 for _ in range(3*(N)+2)]
# ubx = [1 for _ in range(3*(N)+2)]
# start = time.time()

# print(time.time()-start)

# # Convert the MPC results to numpy arrays
# states_mpc = np.array(xi)
# # print(states_mpc)
# # Plot the results
# plt.plot(time_points, states_mpc[:, 0], label='Position')
# plt.plot(time_points, states_mpc[:, 1], label='Velocity')
# plt.xlabel('Time')
# plt.legend()
# plt.grid(True)
# plt.show()
