import numpy as np
from SICMDP import SICMDP_Env
from numpy.linalg import norm, solve


class pollution_Env(SICMDP_Env):
    def __init__(self, coeff=1+5e-1, gamma=0.9):
        name = 'pollution env'
        S = 4
        A = 2

        P = np.zeros((S, A, S), dtype=float)
        # a = 0: Stay here
        # a = 1: Move to the next state
        for s in range(S):
            P[s, 0, s] = 1.
            P[s, 1, (s+1) % S] = 1.

        r = np.zeros((S, A), dtype=float)
        r[:, 0] = 0.
        r[:, 1] = -1.

        self.state_coordinates = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]])

        def f(x):
            # return np.exp(-x * x)
            # return np.exp(-x)
            return 1. / (1. + x)
            # return 1. / (1. + x * x)

        def c(y):
            if np.ndim(y) == 1 and np.size(y) == dim_Y:
                c_s = f(norm(y - self.state_coordinates, axis=1))[:, np.newaxis]
                return np.repeat(c_s, repeats=A, axis=1)
            elif np.ndim(y) == 2:
                y_num = y.shape[0]
                y_axis = y[:, :, np.newaxis].swapaxes(1, 2)
                c_s = f(norm(y_axis - self.state_coordinates, axis=2))[:, :, np.newaxis]
                return np.repeat(c_s, repeats=A, axis=2)
            # Unexpected behavior
            else:
                assert 1 == 0

        # Uniform distribution
        # mu = np.ones((S), dtype=float) / S
        # mu = np.array([.2, .3, .4, .1], dtype=float)
        mu = np.array([.9, .05, .025, .025], dtype=float)
        d_target = np.ones((S), dtype=float) / S
        # mu = np.array([1., 0., 0., 0.], dtype=float)

        def u(y):
            if np.ndim(y) == 1 and np.size(y) == dim_Y:
                return coeff * (1. / (1 - gamma)) * np.sum(c(y)[:, 0] * d_target)
            elif np.ndim(y) == 2:
                c_array = c(y)
                return coeff * (1. / (1 - gamma)) * np.sum(c_array[:, :, 0] * d_target, axis=1)
            # Unexpected behavior
            else:
                assert 1 == 0

        dim_Y = 2
        lb_Y = np.zeros((dim_Y), dtype=float)
        ub_Y = 2. * np.ones((dim_Y), dtype=float)

        super(pollution_Env, self).__init__(name=name, S=S, A=A, gamma=gamma, P=P, r=r, c=c, u=u, mu=mu, dim_Y=dim_Y,
                                            lb_Y=lb_Y, ub_Y=ub_Y)



class random_pollution_Env(SICMDP_Env):
    def __init__(self, S=4, A=4, dim_Y=2, coeff=1+1e-6, gamma=0.9):
        name = 'random pollution env'

        P = np.random.uniform(0, 1, size=(S, A, S))
        P = P / (np.sum(P, axis=2)[:, :, np.newaxis])

        r = np.random.uniform(low=0, high=1, size=(S, A))

        self.state_coordinates = np.random.uniform(0, 2, size=(S, dim_Y))

        def f(x):
            # return np.exp(-x * x)
            # return np.exp(-x)
            return 1. / (1. + x)
            # return 1. / (1. + x * x)

        def c(y):
            if np.ndim(y) == 1 and np.size(y) == dim_Y:
                c_s = f(norm(y - self.state_coordinates, axis=1))[:, np.newaxis]
                return np.repeat(c_s, repeats=A, axis=1)
            elif np.ndim(y) == 2:
                y_num = y.shape[0]
                y_axis = y[:, :, np.newaxis].swapaxes(1, 2)
                c_s = f(norm(y_axis - self.state_coordinates, axis=2))[:, :, np.newaxis]
                return np.repeat(c_s, repeats=A, axis=2)
            # Unexpected behavior
            else:
                assert 1 == 0


        mu = np.ones((S), dtype=float) / S
        d_target = np.ones((S), dtype=float) / S

        def u(y):
            if np.ndim(y) == 1 and np.size(y) == dim_Y:
                return coeff * (1. / (1 - gamma)) * np.sum(c(y)[:, 0] * d_target)
            elif np.ndim(y) == 2:
                c_array = c(y)
                return coeff * (1. / (1 - gamma)) * np.sum(c_array[:, :, 0] * d_target, axis=1)
            # Unexpected behavior
            else:
                assert 1 == 0

        lb_Y = np.zeros((dim_Y), dtype=float)
        ub_Y = 2. * np.ones((dim_Y), dtype=float)

        super(random_pollution_Env, self).__init__(name=name, S=S, A=A, gamma=gamma, P=P, r=r, c=c, u=u, mu=mu,
                                                   dim_Y=dim_Y, lb_Y=lb_Y, ub_Y=ub_Y)


class random_complex_pollution_Env(SICMDP_Env):
    def __init__(self, S=4, A=4, pos_per_state=10, dim_Y=2, coeff=1+1e-6, gamma=0.9, tgt_mode='Action_uniform'):
        name = 'random complex pollution env'

        P = np.random.uniform(0, 1, size=(S, A, S))
        P = P / (np.sum(P, axis=2)[:, :, np.newaxis])

        r = np.random.uniform(low=0, high=1, size=(S, A))

        self.state_coordinates = np.random.uniform(0, 2, size=(S, pos_per_state, dim_Y))

        def f(x):
            # return np.exp(-x * x)
            # return np.exp(-x)
            # return 1. / (1. + x)
            return 1. / (1. + x * x)

        def c(y):
            if np.ndim(y) == 1 and np.size(y) == dim_Y:
                c_s_per_pos = f(norm(y - self.state_coordinates, axis=2))
                c_s = np.sum(c_s_per_pos, axis=1)[:, np.newaxis]
                return np.repeat(c_s, repeats=A, axis=1)
            elif np.ndim(y) == 2:
                y_num = y.shape[0]
                y_axis = y[:, :, np.newaxis, np.newaxis].swapaxes(1, 3)  # (n, 1, 1, dim_Y)
                c_s_per_pos = f(norm(y_axis - self.state_coordinates, axis=3))  # (n, S, pos_per_state)
                c_s = np.sum(c_s_per_pos, axis=2)[:, :, np.newaxis]
                return np.repeat(c_s, repeats=A, axis=2)
            # Unexpected behavior
            else:
                assert 1 == 0

        mu = np.ones((S), dtype=float) / S
        if tgt_mode == 'Action_uniform':
            tgt_pi = np.ones((S, A)) / A
            tgt_pi_axis = tgt_pi[:, :, np.newaxis]
            tgt_P_pi = np.sum(P * tgt_pi_axis, axis=1)
            d_target = solve((np.eye(S) - gamma * tgt_P_pi).T, mu) * (1 - gamma)
        elif tgt_mode == 'Uniform':
            d_target = np.ones((S), dtype=float) / S
        else:
            raise

        def u(y):
            if np.ndim(y) == 1 and np.size(y) == dim_Y:
                return coeff * (1. / (1 - gamma)) * np.sum(c(y)[:, 0] * d_target)
            elif np.ndim(y) == 2:
                c_array = c(y)
                return coeff * (1. / (1 - gamma)) * np.sum(c_array[:, :, 0] * d_target, axis=1)
            # Unexpected behavior
            else:
                assert 1 == 0

        lb_Y = np.zeros((dim_Y), dtype=float)
        ub_Y = 2. * np.ones((dim_Y), dtype=float)

        super(random_complex_pollution_Env, self).__init__(name=name, S=S, A=A, gamma=gamma, P=P, r=r, c=c, u=u, mu=mu,
                                                           dim_Y=dim_Y, lb_Y=lb_Y, ub_Y=ub_Y)
