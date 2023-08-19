import numpy as np
from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
import casadi as ca
import time


class Crazyflie():
    def __init__(self, dt = 0.1, init_state = np.array([0,0,0,1,0,0,0,0,0,0,0,0,0])) -> None:
        # parameters: # taken from webots mass tab
        self.g0  = 9.81    # [m.s^2] accerelation of gravity
        self.mq  = 0.05      # [kg] total mass (with one marker)
        
        self.Ixx = 3.5e-5   # [kg.m^2] Inertia moment around x-axis
        self.Iyy = 3.5e-5   # [kg.m^2] Inertia moment around y-axis
        self.Izz = 6.25e-5   # [kg.m^2] Inertia moment around z-axis
        self.Cd  = 7.9379e-06 # [N/krpm^2] Drag coef
        # self.Cd  = 0 # [N/krpm^2] Drag coef
        self.Ct  = 0.00221472    # [N/krpm^2] Thrust coef # calculated from hover velocity in webot
        self.dq  = 65e-3      # [m] distance between motors' center
        self.l   = self.dq/2       # [m] distance between motors' center and the axis of rotation
        self.dt = dt
        self.N_STATES = 12
        self.N_ACTIONS = 4 # motor speeds

        self.dx = np.zeros(self.N_STATES)

        we = np.sqrt(self.mq*self.g0/(4*self.Ct))


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
        
        self.B*=we

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


class MPCControllerAcados():
    def __init__(self, agent=None) -> None:
        ocp = AcadosOcp()
        model = create_model(agent)
        ocp.model = model

        nx = ocp.model.x.size()[0]
        nu = ocp.model.u.size()[0]
        N = 50
        self.N = N
        Tf = N*agent.dt

        # weight matrices
        Q_mat = 1*np.diag([12, 10, 10, 1e-3, 1e-3, 1e-3, 40, 40, 100, 1e-5, 1e-5, 100])
        # Q_mat = 100*np.diag([50, 500, 500, 1e-3, 1e-3, 1e-3, 5, 5, 10, 1e-5, 1e-5, 10])
        R_mat = 0.01*np.diag([1e-2, 1e-2, 1e-2, 1e-2])

        setpoint = np.array([0, 0, 2 , 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
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
        ocp.model.cost_expr_ext_cost_e = (model.x-setpoint).T@ (50*Q_mat) @ (model.x-setpoint)

        # ocp.model.yref   = np.array([0, 0, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, agent.hov_w, agent.hov_w, agent.hov_w, agent.hov_w])
        # ocp.model.yref_e = np.array([0, 0, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        Fmax = 67
        Fmin = -67
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

        # self.preds = list(np.zeros(N, nx))

    def get_action(self, state_c, state_d):
        # acados_integrator.set("x", x0)
        # acados_integrator.set("u", u0)

        start = time.time()
        self.solver.set(0, "lbx", state_c)
        self.solver.set(0, "ubx", state_c)
        
        status = self.solver.solve()
        # print(time.time()-start)

        # acados_integrator.solve()
        
        # getting the state from a separate integrator rather than the perfect MPC prediction
        # x0 = acados_integrator.get("x")

        # the perfect MPC predicted state. not sure which one to use.
        self.x0 = self.solver.get(1, "x")
        self.u0 = self.solver.get(0, "u")
        

        return self.u0




