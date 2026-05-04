import numpy as np


def RK4(y_i, dt, tf, rhs):
    N = int(round(tf / dt))
    y = np.zeros((N+1, 6))
    y[0] = y_i

    t = 0.0
    for i in range(N):
        k1 = rhs(t, y[i])
        k2 = rhs(t + 0.5*dt, y[i] + 0.5*dt*k1)
        k3 = rhs(t + 0.5*dt, y[i] + 0.5*dt*k2)
        k4 = rhs(t + dt, y[i] + dt*k3)

        y[i+1] = y[i] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        t += dt

    return y
