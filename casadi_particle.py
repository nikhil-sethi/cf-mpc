import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time

# MPC parameters
N = 20  # Prediction horizon
dt = 0.1  # Time step
Q = 0.5*ca.DM(np.diag([1]))  # State cost matrix
R = ca.DM(np.diag([0.1])  ) # Control cost matrix


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


# Create the CasADi symbols for the function definition
x = ca.MX.sym('x', 2)
u = ca.MX.sym('u')

x_next = dynamics(x,u)
j = cost(x,u)

F = ca.Function('F', [x, u], [x_next, j], ['x0', 'u'], ['x', 'j'])


# Simulate the MPC trajectory
num_time_steps = 100
time_points = np.arange(0, num_time_steps+1) * dt
states_mpc = []


xi = [[0,0]]

lbg = [0 for _ in range(2*(N))]
ubg = [0 for _ in range(2*(N))]
lbx = [-1 for _ in range(3*(N)+2)]
ubx = [1 for _ in range(3*(N)+2)]
start = time.time()
for i in range(num_time_steps):

    ## ========= RAW METHOD: faster but unreadable

    Xk = ca.MX.sym('X0', 2, N+1) # init constraint
    Uk = ca.MX.sym('X0', 1, N) # controls

    constraints = []
    
    lbx[:2] = xi[-1].copy()
    ubx[:2] = xi[-1].copy()

    objective = 0

    for k in range(N):

        Fk = F(x0=Xk[:,k], u=Uk[:,k]) # predict next state
        Xk_pred = Fk['x']
        # Xk_pred = dynamics(Xk[:,k], Uk[:,k])
        # objective += cost(Xk[:,k], Uk[:,k])

        constraints += [Xk[:,k+1] - Xk_pred] # constrain next state to be from the dynamic model
        objective += Fk['j']

    nlp = {'x': ca.horzcat(ca.reshape(Xk, 1, 2*(N+1)), Uk), 'f': objective, 'g':ca.vertcat(*constraints)}
    opts = {'ipopt.tol': 1e-4, 'print_time': 0, 'ipopt.print_level': 0}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    sol = solver(lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
    control_opt = sol['x'][2*(N+1)]

    ## ==========


    ## ========= OPTI METHOD: SLOWER but readble

    # opti = ca.Opti()
    # objective = 0

    # Xk = opti.variable(2, N+1)
    # Uk = opti.variable(1, N)

    # opti.subject_to(Xk[:,0] == xi[-1])

    # for k in range(N):
    #     Fk = F(x0=Xk[:,k], u=Uk[:,k]) # predict next state
    #     Xk_pred = Fk['x']
    #     objective += Fk['j']

    #     opti.subject_to(Xk[:,k+1] == Xk_pred)
    #     opti.subject_to(opti.bounded(-1, Uk[:,k], 1))
    #     opti.subject_to(opti.bounded(-1, Xk[:,k], 1))

    # p_opts = {"expand":False, "print_time":0}
    # s_opts =  {'tol': 1e-4, 'print_level': 0}
    # opti.minimize(objective)
    
    # opti.solver('ipopt', p_opts,  s_opts)
    # sol = opti.solve()
    # control_opt = sol.value(Uk)[0]

    x_opt = ca.reshape(sol['x'][:42], 2, N+1)
    
    xi += [list(x_opt[:,1].full()[:,0])]

print(time.time()-start)

# Convert the MPC results to numpy arrays
states_mpc = np.array(xi)
# print(states_mpc)
# Plot the results
plt.plot(time_points, states_mpc[:, 0], label='Position')
plt.plot(time_points, states_mpc[:, 1], label='Velocity')
plt.xlabel('Time')
plt.legend()
plt.grid(True)
plt.show()
