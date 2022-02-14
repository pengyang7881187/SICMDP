import gym
import time
import warnings
import numpy as np
import gurobipy as gp
from numpy.linalg import solve, norm
from gurobipy import GRB

z_epsilon = 1e-16
np.random.seed(74751)
np.seterr(divide='ignore', invalid='ignore')


# We only consider Y is a rectangular in R^d
class SICMDP_Env(gym.Env):
    def __init__(self, name: str, S: int, A: int, gamma: float, P, r, c, u, mu, dim_Y: int, lb_Y, ub_Y, *args, **kwargs):
        super(SICMDP_Env, self).__init__(*args, **kwargs)
        self.name = name
        self.S = S
        self.A = A
        self.gamma = gamma
        self.P = np.array(P, dtype=np.float64)
        self.r = np.array(r, dtype=np.float64)
        self.c = c  # Given y, c(y) is a np array with shape (S, A)
        self.u = u  # Given y, u(y) is a real number
        self.mu = np.array(mu, dtype=np.float64)
        self.dim_Y = dim_Y
        self.lb_Y = np.array(lb_Y, dtype=np.float64)
        self.ub_Y = np.array(ub_Y, dtype=np.float64)
        self.y0 = (self.lb_Y + self.ub_Y) / 2.
        # self.y0 = self.lb_Y
        # P, R, mu, lb_Y and ub_Y should be valid
        assert self.P.shape == (self.S, self.A, self.S)
        assert np.allclose(np.sum(self.P, axis=2), 1.)
        assert self.mu.shape == (self.S,)
        assert np.allclose(np.sum(self.mu), 1.)
        assert self.r.shape == (self.S, self.A)
        assert self.lb_Y.shape == (self.dim_Y,)
        assert self.ub_Y.shape == (self.dim_Y,)

        self.vals_mask = np.array([0])

        self.grid_fineness = -1  # -1 is a negative flag
        self.grid_c_u_flag = False
        self.check_grid_fineness = -1  # -1 is a negative flag
        self.check_grid_c_u_flag = False
        self.grid = np.zeros((1))
        self.check_grid = np.zeros((1))
        self.grid_c = np.zeros((1))
        self.grid_u = np.zeros((1))
        self.check_grid_c = np.zeros((1))
        self.check_grid_u = np.zeros((1))

    def reset_grid(self):
        self.grid_fineness = -1  # -1 is a negative flag
        self.grid_c_u_flag = False
        self.grid = np.zeros((1))
        self.grid_c = np.zeros((1))
        self.grid_u = np.zeros((1))
        return

    def reset_check_grid(self):
        self.check_grid_fineness = -1  # -1 is a negative flag
        self.check_grid_c_u_flag = False
        self.check_grid = np.zeros((1))
        self.check_grid_c = np.zeros((1))
        self.check_grid_u = np.zeros((1))
        return

    # Check whether an input policy pi is a valid policy
    def check_pi(self, pi):
        assert pi.shape == (self.S, self.A)
        assert np.allclose(np.sum(pi, axis=1), 1.)
        return

    # Check whether an input z is feasible and find new y via discretization (P is not known)
    def check_z_feasible_and_find_new_y(self, z, grid, grid_c, grid_u, epsilon=1e-12):
        feasible_flag = False
        reduce_z = np.sum(z, axis=2)
        vals = (1. / (1. - self.gamma)) * np.sum(grid_c * reduce_z, axis=(1, 2)) - grid_u
        vals = vals + self.vals_mask
        max_y_index = np.argmax(vals)
        max_cons_violat = vals[max_y_index]
        max_y = grid[max_y_index]
        self.vals_mask[max_y_index] = -np.inf
        if max_cons_violat <= epsilon:
            feasible_flag = True
        return feasible_flag, max_y, max_cons_violat

    # Check whether a policy pi is feasible (P is known)
    def check_pi_feasible_true_P(self, pi, check_fineness, epsilon=1e-10):
        grid = self.generate_grid(fineness=check_fineness, check_flag=True)
        grid_c, grid_u = self.generate_grid_c_u(check_flag=True)
        feasible_flag = False
        q_pi = self.q_pi(pi)
        vals = (1. / (1. - self.gamma)) * np.sum(grid_c * q_pi, axis=(1, 2)) - grid_u
        max_cons_violat = np.max(vals)
        # Small tolerance (Avoid numerical error)
        if max_cons_violat <= epsilon:
            feasible_flag = True
        return feasible_flag, max_cons_violat

    def generate_heat_map(self, pi, check_fineness):
        grid = self.generate_grid(fineness=check_fineness, check_flag=True)
        grid_c, grid_u = self.generate_grid_c_u(check_flag=True)
        q_pi = self.q_pi(pi)
        vals = np.maximum((1. / (1. - self.gamma)) * np.sum(grid_c * q_pi, axis=(1, 2)) - grid_u, 0.)
        return vals

    # Generate grids of Y
    def generate_grid(self, fineness=100, check_flag=False):
        if check_flag:
            # The check grid has been calculated
            if self.check_grid_fineness == fineness:
                return self.check_grid
            elif self.check_grid_fineness != -1:
                self.reset_check_grid()
        else:
            # Init the array which avoid selecting the same grid as before
            self.vals_mask = np.zeros(int((fineness+1) ** self.dim_Y))
            # The grid has been calculated
            if self.grid_fineness == fineness:
                return self.grid
            elif self.grid_fineness != -1:
                self.reset_grid()

        grid_num = fineness + 1
        coordinate_lst = []
        index_array = np.indices(list(self.dim_Y * [grid_num]))
        for i in range(self.dim_Y):
            coordinate_lst.append(index_array[i].reshape(-1) * ((self.ub_Y[i] - self.lb_Y[i]) / fineness) + self.lb_Y[i])
        grid = np.stack(list(coordinate_lst), axis=1)

        # Save the result
        if check_flag:
            self.check_grid = grid
            self.check_grid_fineness = fineness
        else:
            self.grid = grid
            self.grid_fineness = fineness
        return grid

    # When you call this method, the grid must have been generated
    def generate_grid_c_u(self, check_flag=False):
        if check_flag:
            if self.check_grid_c_u_flag:
                return self.check_grid_c, self.check_grid_u
            grid = self.check_grid
        else:
            if self.grid_c_u_flag:
                return self.grid_c, self.grid_u
            grid = self.grid

        grid_c = self.c(grid)
        grid_u = self.u(grid)

        # Save the result
        if check_flag:
            self.check_grid_c = grid_c
            self.check_grid_u = grid_u
            self.check_grid_c_u_flag = True
        else:
            self.grid_c = grid_c
            self.grid_u = grid_u
            self.grid_c_u_flag = True
        return grid_c, grid_u

    # (S,)
    def r_pi(self, pi):
        self.check_pi(pi)
        return np.sum(pi * self.r, axis=1)

    # (S,)
    def c_pi_y(self, pi, y):
        self.check_pi(pi)
        return np.sum(pi * self.c(y), axis=1)

    # (S, S)
    def P_pi(self, pi):
        self.check_pi(pi)
        pi_axis = pi[:, :, np.newaxis]
        return np.sum(self.P * pi_axis, axis=1)

    # Value function, (S,)
    def V_pi(self, pi):
        r_pi = self.r_pi(pi)
        P_pi = self.P_pi(pi)
        return solve(np.eye(self.S) - self.gamma * P_pi, r_pi)

    # Obj
    def Obj_pi(self, pi):
        V_pi = self.V_pi(pi)
        return np.dot(V_pi, self.mu)

    # Constrain value function, (S,)
    def C_pi_y(self, pi, y):
        c_pi_y = self.c_pi_y(pi, y)
        P_pi = self.P_pi(pi)
        return solve(np.eye(self.S) - self.gamma * P_pi, c_pi_y)

    # Q value function, (S, A)
    def Q_pi(self, pi):
        V_pi = self.V_pi(pi)
        return self.r + self.gamma * np.inner(self.P, V_pi)

    # State occupancy measure, (S,)
    def d_pi(self, pi):
        P_pi = self.P_pi(pi)
        return solve((np.eye(self.S) - self.gamma * P_pi).T, self.mu) * (1 - self.gamma)

    # State-action occupancy measure, (S, A)
    def q_pi(self, pi):
        d_pi = self.d_pi(pi)
        return pi * d_pi[:, np.newaxis]

    # State-action-state occupancy measure, (S, A, S)
    def z_pi(self, pi):
        q_pi = self.q_pi(pi)
        return self.P * q_pi[:, :, np.newaxis]

    # Recover policy pi from z_pi
    def pi_z(self, z):
        q = np.sum(z, axis=2)
        return q / np.sum(q, axis=1)[:, np.newaxis]

    # Recover P_UC from z
    def P_z(self, z):
        q = np.sum(z, axis=2)
        return z / q[:, :, np.newaxis]

    # Sample from P, used in sample_uniformly and sample_from_nu
    def sample(self, SA_array):
        SAS_array = np.zeros((self.S, self.A, self.S))
        for s in range(self.S):
            for a in range(self.A):
                SAS_array[s, a, :] = np.random.multinomial(n=SA_array[s, a], pvals=self.P[s, a])
        return SAS_array

    # Sample uniformly in (S, A), m = n * S * A, and sample from P
    def sample_uniformly(self, n: int):
        SA_array = n * np.ones((self.S, self.A))
        return SA_array, self.sample(SA_array)

    # Sample m times from nu, which is a distribution on (S, A), and sample from P
    def sample_from_nu(self, m: int, nu):
        assert nu.shape == (self.S, self.A)
        assert np.allclose(np.sum(nu), 1.)
        nu = nu.reshape(-1)
        SA_array = np.random.multinomial(n=m, pvals=nu).reshape(self.S, self.A)
        return SA_array, self.sample(SA_array)

    # Check whether P is in the uncertainty set
    def check_uncertainty_set(self, P_hat, d_delta):
        nan = np.isnan(d_delta)
        ub = (P_hat + d_delta > self.P)
        lb = (P_hat - d_delta < self.P)
        return np.all((ub & lb) | nan)

    def SI_plan(self, iter_upper_bound, fineness, check_fineness, silent_flag=True):
        search_grid = self.generate_grid(fineness=fineness)
        search_grid_c, search_grid_u = self.generate_grid_c_u(check_flag=False)

        time_start = time.time()
        S = self.S
        A = self.A

        Y0 = [self.y0]
        z = np.zeros((S, A, S))
        old_z = np.zeros((S, A, S))

        # Model with gurobi
        model = gp.Model(self.name + ' Extended LSIP')

        model.setParam('OutputFlag', 0)

        z_var = model.addVars(range(S), range(A), range(S), lb=0, name='z')
        # Set objective
        obj = gp.LinExpr()
        for s in range(S):
            for a in range(A):
                obj += (z_var.sum(s, a, '*') * self.r[s, a])
        model.setObjective(obj, GRB.MAXIMIZE)

        ##### Set constraints #####
        # y0
        constr_y0 = gp.LinExpr()
        for s in range(S):
            for a in range(A):
                constr_y0 += (1. / (1. - self.gamma)) * (z_var.sum(s, a, '*') * self.c(self.y0)[s, a])
        model.addConstr(constr_y0 <= self.u(self.y0), name='c_y0')
        # MDP
        for s in range(S):
            for a in range(A):
                for s1 in range(S):
                    name = 'c_' + str(s) + '_' + str(a) + '_' + str(s1) + '_p'
                    model.addConstr(z_var[s, a, s1] - self.P[s, a, s1] * z_var.sum(s, a, '*') == 0, name=name)
        # Valid occupancy measure
        for s in range(S):
            name = 'c_valid_measure_' + str(s)
            model.addConstr(z_var.sum(s, '*', '*') - (1 - self.gamma) * self.mu[s] - self.gamma * z_var.sum('*', '*', s) == 0,
                            name=name)
        #########################
        Y0_size = 1
        for i in range(iter_upper_bound):
            if not silent_flag:
                print('Step ' + str(i + 1) + ':')
            # Solve LP, if infeasible: return, the problem is infeasible
            model.optimize()
            if model.status == GRB.INFEASIBLE or model.status == GRB.INF_OR_UNBD:
                raise
                return False, -1, -1, -1, np.inf, -np.inf
            if model.status != GRB.OPTIMAL:
                print('Status: ' + str(model.status))
                raise
                break
            for s in range(S):
                for a in range(A):
                    for s1 in range(S):
                        z[s, a, s1] = z_var[s, a, s1].X

            # Is z feasible for SICMDP? Feasible: break, z is optimal; Infeasible: continue
            # At the same time, add new y to Y0 and modify the model
            feasible_flag, y, max_cons_violat = self.check_z_feasible_and_find_new_y(z, search_grid, search_grid_c,
                                                                                     search_grid_u)
            Obj = model.ObjVal  # This is not the true Obj for the RL problem, but the LP OBJ
            if not silent_flag:
                print('Obj: %g' % Obj)
                print('Max constraint violation: %g' % max_cons_violat)
            # Policies could be the same, but the chosen Ps from the uncertainty set could be different
            # print('Policy:' + str(env.pi_z(z)))

            if feasible_flag:
                break
            if norm(z - old_z) < z_epsilon:
                if not silent_flag:
                    print('z not change')
                break
            old_z = z.copy()
            Y0.append(y)
            Y0_size += 1
            constr_y = gp.LinExpr()
            for s in range(S):
                for a in range(A):
                    constr_y += (1. / (1. - self.gamma)) * (z_var.sum(s, a, '*') * self.c(y)[s, a])
            constr_name = 'c_y' + str(Y0_size)
            model.addConstr(constr_y <= self.u(y), name=constr_name)
        time_end = time.time()
        if not silent_flag:
            print('Time: ' + str(time_end - time_start))
            print('Checking...')

        pi_hat = self.pi_z(z)
        feasible_flag, max_cons_violat = self.check_pi_feasible_true_P(pi_hat, check_fineness)
        Obj = self.Obj_pi(pi_hat)
        return feasible_flag, pi_hat, z, Y0, max_cons_violat, Obj


def SI_CRL(env, SA_array, SAS_array, delta, iter_upper_bound, fineness, check_fineness, silent_flag=True):
    search_grid = env.generate_grid(fineness=fineness)
    search_grid_c, search_grid_u = env.generate_grid_c_u(check_flag=False)
    P_hat = SAS_array / np.maximum(SA_array[:, :, np.newaxis], 1.)
    d_delta = np.minimum(np.sqrt(2. * P_hat * (1. - P_hat) * np.log(4. / delta) / SAS_array) +
                         4. * np.log(4. / delta) / SAS_array,
                         np.sqrt(0.5 * np.log(2. / delta) / SAS_array))   # minimum(nan, inf) = minimum(inf, nan) = nan
    if not silent_flag:
        print('d_delta nan exist: ' + str(np.any(np.isnan(d_delta))))
        print('d_delta max: ' + str(np.max(d_delta[np.logical_not(np.isnan(d_delta))])))
    if not env.check_uncertainty_set(P_hat, d_delta):
        warnings.warn('Uncertainty set too small')

    time_start = time.time()
    S = env.S
    A = env.A

    Y0 = [env.y0]
    z = np.zeros((S, A, S))
    old_z = np.zeros((S, A, S))

    # Model with gurobi
    model = gp.Model(env.name + ' Extended LSIP')

    model.setParam('OutputFlag', 0)

    z_var = model.addVars(range(S), range(A), range(S), lb=0, name='z')
    # Set objective
    obj = gp.LinExpr()
    for s in range(S):
        for a in range(A):
            obj += (z_var.sum(s, a, '*') * env.r[s, a])
    model.setObjective(obj, GRB.MAXIMIZE)

    ##### Set constraints #####
    # y0
    constr_y0 = gp.LinExpr()
    for s in range(S):
        for a in range(A):
            constr_y0 += (1. / (1. - env.gamma)) * (z_var.sum(s, a, '*') * env.c(env.y0)[s, a])
    model.addConstr(constr_y0 <= env.u(env.y0), name='c_y0')
    # Uncertainty set
    for s in range(S):
        for a in range(A):
            for s1 in range(S):
                if SAS_array[s, a, s1] == 0:
                    continue
                name_1 = 'c_' + str(s) + '_' + str(a) + '_' + str(s1) + '_ub'
                name_2 = 'c_' + str(s) + '_' + str(a) + '_' + str(s1) + '_lb'
                model.addConstr(z_var[s, a, s1] - (P_hat[s, a, s1] + d_delta[s, a, s1]) * z_var.sum(s, a, '*') <= 0,
                                name=name_1)
                model.addConstr(z_var[s, a, s1] - (P_hat[s, a, s1] - d_delta[s, a, s1]) * z_var.sum(s, a, '*') >= 0,
                                name=name_2)
    # Valid occupancy measure
    for s in range(S):
        name = 'c_valid_measure_' + str(s)
        model.addConstr(
            z_var.sum(s, '*', '*') - (1 - env.gamma) * env.mu[s] - env.gamma * z_var.sum('*', '*', s) == 0,
            name=name)
    #########################
    Y0_size = 1
    for i in range(iter_upper_bound):
        if not silent_flag:
            print('Step ' + str(i+1) + ':')
        # Solve LP, if infeasible: return, the problem is infeasible
        model.optimize()
        if model.status == GRB.INFEASIBLE or model.status == GRB.INF_OR_UNBD:
            return False, -1, -1, -1, np.inf, -np.inf
        if model.status != GRB.OPTIMAL:
            print('Status: ' + str(model.status))
            break
        for s in range(S):
            for a in range(A):
                for s1 in range(S):
                    z[s, a, s1] = z_var[s, a, s1].X


        # Is z feasible for SICMDP? Feasible: break, z is optimal; Infeasible: continue
        # At the same time, add new y to Y0 and modify the model
        feasible_flag, y, max_cons_violat = env.check_z_feasible_and_find_new_y(z, search_grid, search_grid_c, search_grid_u)
        Obj = model.ObjVal  # This is not the true Obj for the RL problem, but the LP OBJ
        if not silent_flag:
            print('Obj: %g' % Obj)
            print('Max constraint violation: %g' % max_cons_violat)
        # Policies could be the same, but the chosen Ps from the uncertainty set could be different
        # print('Policy:' + str(env.pi_z(z)))

        if feasible_flag:
            break
        if norm(z - old_z) < z_epsilon:
            if not silent_flag:
                print('z not change')
            break
        old_z = z.copy()
        Y0.append(y)
        Y0_size += 1
        constr_y = gp.LinExpr()
        for s in range(S):
            for a in range(A):
                constr_y += (1. / (1. - env.gamma)) * (z_var.sum(s, a, '*') * env.c(y)[s, a])
        constr_name = 'c_y' + str(Y0_size)
        model.addConstr(constr_y <= env.u(y), name=constr_name)
    time_end = time.time()
    if not silent_flag:
        print('Time: ' + str(time_end - time_start))
        print('Checking...')

    pi_hat = env.pi_z(z)
    feasible_flag, max_cons_violat = env.check_pi_feasible_true_P(pi_hat, check_fineness)
    Obj = env.Obj_pi(pi_hat)
    return feasible_flag, pi_hat, z, Y0, max_cons_violat, Obj


def SI_CRL_baseline(env, SA_array, SAS_array, delta, fineness, check_fineness):
    Y1 = env.generate_grid(fineness)

    P_hat = SAS_array / np.maximum(SA_array[:, :, np.newaxis], 1.)
    d_delta = np.minimum(np.sqrt(2. * P_hat * (1. - P_hat) * np.log(4. / delta) / SAS_array) +
                         4. * np.log(4. / delta) / SAS_array,
                         np.sqrt(0.5 * np.log(2. / delta) / SAS_array))   # minimum(nan, inf) = minimum(inf, nan) = nan
    if not env.check_uncertainty_set(P_hat, d_delta):
        warnings.warn('Uncertainty set too small')

    time_start = time.time()
    S = env.S
    A = env.A

    z = np.zeros((S, A, S))

    # Model with gurobi
    model = gp.Model(env.name + ' Extended LSIP')

    model.setParam('OutputFlag', 0)

    z_var = model.addVars(range(S), range(A), range(S), lb=0, name='z')
    # Set objective
    obj = gp.LinExpr()
    for s in range(S):
        for a in range(A):
            obj += (z_var.sum(s, a, '*') * env.r[s, a])
    model.setObjective(obj, GRB.MAXIMIZE)

    ##### Set constraints #####
    # Y1
    y_num = 0
    for y in Y1:
        constr_y = gp.LinExpr()
        for s in range(S):
            for a in range(A):
                constr_y += (1. / (1. - env.gamma)) * (z_var.sum(s, a, '*') * env.c(y)[s, a])
        constr_name = 'c_y' + str(y_num)
        model.addConstr(constr_y <= env.u(y), name=constr_name)
        y_num += 1

    # Uncertainty set
    for s in range(S):
        for a in range(A):
            for s1 in range(S):
                if SAS_array[s, a, s1] == 0:
                    continue
                name_1 = 'c_' + str(s) + '_' + str(a) + '_' + str(s1) + '_ub'
                name_2 = 'c_' + str(s) + '_' + str(a) + '_' + str(s1) + '_lb'
                model.addConstr(z_var[s, a, s1] - (P_hat[s, a, s1] + d_delta[s, a, s1]) * z_var.sum(s, a, '*') <= 0,
                                name=name_1)
                model.addConstr(z_var[s, a, s1] - (P_hat[s, a, s1] - d_delta[s, a, s1]) * z_var.sum(s, a, '*') >= 0,
                                name=name_2)
    # Valid occupancy measure
    for s in range(S):
        name = 'c_valid_measure_' + str(s)
        model.addConstr(
            z_var.sum(s, '*', '*') - (1 - env.gamma) * env.mu[s] - env.gamma * z_var.sum('*', '*', s) == 0,
            name=name)

    # Solve LP, if infeasible: return, the problem is infeasible
    model.optimize()
    if model.status == GRB.INFEASIBLE or model.status == GRB.INF_OR_UNBD:
        raise
        return False, -1, -1, -1, -1, -1
    if model.status != GRB.OPTIMAL:
        print('Status: ' + model.status)
        raise
    for s in range(S):
        for a in range(A):
            for s1 in range(S):
                z[s, a, s1] = z_var[s, a, s1].X
    time_end = time.time()
    print('Time: ' + str(time_end - time_start))

    print('Checking...')

    pi_hat = env.pi_z(z)
    feasible_flag, max_cons_violat = env.check_pi_feasible_true_P(pi_hat, check_fineness)
    Obj = env.Obj_pi(pi_hat)
    return feasible_flag, pi_hat, z, max_cons_violat, Obj, time_end - time_start



# def SI_CRL_generative_model(env, n, delta, iter_upper_bound, fineness):
#     SA_array, SAS_array = env.sample_uniformly(n)
#     return SI_CRL(env, SA_array, SAS_array, delta, iter_upper_bound, fineness)
#
#
# def SI_CRL_nu(env, m, nu, delta, iter_upper_bound, fineness):
#     SA_array, SAS_array = env.sample_from_nu(m, nu)
#     return SI_CRL(env, SA_array, SAS_array, delta, iter_upper_bound, fineness)
#
#
# def SI_CRL_generative_model_baseline(env, n, delta, fineness):
#     SA_array, SAS_array = env.sample_uniformly(n)
#     return SI_CRL_baseline(env, SA_array, SAS_array, delta, fineness)
#
#
# def SI_CRL_nu_baseline(env, m, nu, delta, fineness):
#     SA_array, SAS_array = env.sample_from_nu(m, nu)
#     return SI_CRL_baseline(env, SA_array, SAS_array, delta, fineness)

