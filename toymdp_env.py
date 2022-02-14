import numpy as np
from SICMDP import SICMDP_Env
from numpy import cos
from numpy.linalg import solve


class toymdp_Env(SICMDP_Env):
    def __init__(self, p, gamma, complexity=20):
        # p > 0.5 in general
        name = 'toy mdp env'
        S = 2
        A = 2

        P = np.zeros((S, A, S), dtype=float)
        P[0, 0, 0] = p
        P[0, 0, 1] = 1. - p
        P[0, 1, 0] = 1. - p
        P[0, 1, 1] = p
        P[1, :, 1] = 1.

        r = np.zeros((S, A), dtype=float)
        r[0, :] = 1.
        r[1, 0] = 1e-6
        r[1, 1] = 0.

        mu = np.array([1., 0.], dtype=float)

        tgt_pi = np.array([[0.9, 0.1], [0.9, 0.1]], dtype=float)
        tgt_pi_axis = tgt_pi[:, :, np.newaxis]
        tgt_P_pi = np.sum(P * tgt_pi_axis, axis=1)
        tgt_d_pi = solve((np.eye(S) - gamma * tgt_P_pi).T, mu) * (1 - gamma)
        print('True occ measure: ' + str(tgt_d_pi))

        self.y_star = np.array([np.pi, 3. * np.pi], dtype=float)   # Active position

        # Upper bound is 2, and the value and the derivative at the boundary is 0 (i.e. L smooth, stronger than Lip)
        def f(x):
            return (1 + cos(x)) * cos(complexity * x)

        f_ub = 2.
        self.tgt_u_ub = f_ub * (1. / (1. - gamma)) * tgt_d_pi * tgt_pi[0, :]

        def c(y):
            if np.ndim(y) <= 1 and np.size(y) == 1:
                y = np.sum(y)
                c_array = np.zeros((S, A), dtype=float)
                if y < 2. * np.pi:
                    c_array[0, 0] = f(y - self.y_star[0])
                else:
                    c_array[1, 0] = f(y - self.y_star[1])
                return c_array
            elif y.ndim == 2:
                y_num = y.shape[0]
                c_array = np.zeros((y_num, S, A), dtype=float)
                indicator = (y < 2. * np.pi).astype(float)
                c_array[:, 0, 0] = (indicator * f(y - self.y_star[0])).reshape(-1)
                c_array[:, 1, 0] = ((1. - indicator) * f(y - self.y_star[1])).reshape(-1)
                return c_array
            # Unexpected behavior
            else:
                assert 1 == 0

        def u(y):
            if np.ndim(y) <= 1 and np.size(y) == 1:
                y = np.sum(y)
                if y < 2. * np.pi:
                    return self.tgt_u_ub[0]
                else:
                    return self.tgt_u_ub[1]
            elif y.ndim == 2:
                indicator = (y < 2. * np.pi).astype(float)
                u_array = indicator * self.tgt_u_ub[0] + (1. - indicator) * self.tgt_u_ub[1]
                return u_array.reshape(-1)
            # Unexpected behavior
            else:
                assert 1 == 0

        dim_Y = 1
        lb_Y = np.zeros((dim_Y), dtype=float)
        ub_Y = 13. * np.ones((dim_Y), dtype=float)   # 13 > 4 pi

        super(toymdp_Env, self).__init__(name=name, S=S, A=A, gamma=gamma, P=P, r=r, c=c, u=u, mu=mu, dim_Y=dim_Y,
                                         lb_Y=lb_Y, ub_Y=ub_Y)

        self.true_Obj = self.Obj_pi(tgt_pi)
        return

    # Check whether a policy pi is feasible (P is known)
    def check_pi_feasible_true_P(self, pi, check_fineness, epsilon=1e-10):
        feasible_flag = False
        q_pi = self.q_pi(pi)
        val1 = (1. / (1. - self.gamma)) * np.sum(self.c(self.y_star[0]) * q_pi) - self.u(self.y_star[0])
        val2 = (1. / (1. - self.gamma)) * np.sum(self.c(self.y_star[1]) * q_pi) - self.u(self.y_star[1])
        max_cons_violat = max(val1, val2)
        # Small tolerance (Avoid numerical error)
        if max_cons_violat <= epsilon:
            feasible_flag = True
        return feasible_flag, max_cons_violat


