import math
import argparse
from SICMDP import *
from utils import regress, Myprint
from numpy import sqrt
from pollution_env import random_complex_pollution_Env

parser = argparse.ArgumentParser()
parser.add_argument("--gamma", default=0.9, type=int)
parser.add_argument("--dim_Y", default=2, type=int)
parser.add_argument("--S", default=16, type=int)
parser.add_argument("--A", default=4, type=int)
parser.add_argument("--grid_num", default=int(1e+3), type=int)
parser.add_argument("--check_fineness", default=int(6e+2), type=int)
parser.add_argument("--repeat", default=1, type=int)
parser.add_argument("--epsilon", default=0.1, type=float)
parser.add_argument("--delta_prefix", default=0.01, type=float)
parser.add_argument("--pos_per_state", default=1, type=int)
parser.add_argument("--coeff", default=1 + 1e-6, type=float)
parser.add_argument("--baseline_extend", default=10, type=int)

args = parser.parse_args()


def exp_SI_CRL(env, SA_array, SAS_array, delta, iter_upper_bound, fineness, check_fineness, silent_flag=True):
    time_lst = []
    Obj_lst = []
    max_cons_violat_lst = []
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
    # model.setParam('Method', 1)  # 1 means dual simplex

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
        iter_time_start = time.time()
        model.optimize()
        iter_time_end = time.time()
        if model.status == GRB.INFEASIBLE or model.status == GRB.INF_OR_UNBD:
            return False, -1, -1, -1, np.inf, -np.inf, -1, -1, -1
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
        feasible_flag, y, max_cons_violat = env.check_z_feasible_and_find_new_y(z, search_grid, search_grid_c, search_grid_u)
        # Save the result for the experiment
        pi_hat = env.pi_z(z)
        _, exp_max_cons_violat = env.check_pi_feasible_true_P(pi_hat, check_fineness)
        exp_Obj = env.Obj_pi(pi_hat)

        iter_time = iter_time_end - iter_time_start

        time_lst.append(iter_time)
        max_cons_violat_lst.append(exp_max_cons_violat)
        Obj_lst.append(exp_Obj)

        # Obj = model.ObjVal  # This is not the true Obj for the RL problem, but the LP OBJ
        if not silent_flag:
            print('Time: ' + str(iter_time))
            print('Exp Obj: %g' % exp_Obj)
            print('Exp Max constraint violation: %g' % exp_max_cons_violat)
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
        print('Total Time: ' + str(time_end - time_start))
        # print('Checking...')

    pi_hat = env.pi_z(z)
    feasible_flag, max_cons_violat = env.check_pi_feasible_true_P(pi_hat, check_fineness)
    Obj = env.Obj_pi(pi_hat)
    return feasible_flag, pi_hat, z, Y0, max_cons_violat, Obj, time_lst, np.array(Obj_lst), np.array(max_cons_violat_lst)



# def exp_SI_CRL_baseline(env, SA_array, SAS_array, delta, fineness, check_fineness):
#     time_lst = []
#     Obj_lst = []
#     max_cons_violat_lst = []
#
#
#     Y1 = env.generate_grid(fineness)
#
#     P_hat = SAS_array / np.maximum(SA_array[:, :, np.newaxis], 1.)
#     d_delta = np.minimum(np.sqrt(2. * P_hat * (1. - P_hat) * np.log(4. / delta) / SAS_array) +
#                          4. * np.log(4. / delta) / SAS_array,
#                          np.sqrt(0.5 * np.log(2. / delta) / SAS_array))   # minimum(nan, inf) = minimum(inf, nan) = nan
#     if not env.check_uncertainty_set(P_hat, d_delta):
#         warnings.warn('Uncertainty set too small')
#
#     time_start = time.time()
#     S = env.S
#     A = env.A
#
#     z = np.zeros((S, A, S))
#
#     # Model with gurobi
#     model = gp.Model(env.name + ' Extended LSIP')
#
#     model.setParam('OutputFlag', 0)
#
#     z_var = model.addVars(range(S), range(A), range(S), lb=0, name='z')
#     # Set objective
#     obj = gp.LinExpr()
#     for s in range(S):
#         for a in range(A):
#             obj += (z_var.sum(s, a, '*') * env.r[s, a])
#     model.setObjective(obj, GRB.MAXIMIZE)
#
#     ##### Set constraints #####
#     # Y1
#     y_num = 0
#     for y in Y1:
#         constr_y = gp.LinExpr()
#         for s in range(S):
#             for a in range(A):
#                 constr_y += (1. / (1. - env.gamma)) * (z_var.sum(s, a, '*') * env.c(y)[s, a])
#         constr_name = 'c_y' + str(y_num)
#         model.addConstr(constr_y <= env.u(y), name=constr_name)
#         y_num += 1
#
#     # Uncertainty set
#     for s in range(S):
#         for a in range(A):
#             for s1 in range(S):
#                 if SAS_array[s, a, s1] == 0:
#                     continue
#                 name_1 = 'c_' + str(s) + '_' + str(a) + '_' + str(s1) + '_ub'
#                 name_2 = 'c_' + str(s) + '_' + str(a) + '_' + str(s1) + '_lb'
#                 model.addConstr(z_var[s, a, s1] - (P_hat[s, a, s1] + d_delta[s, a, s1]) * z_var.sum(s, a, '*') <= 0,
#                                 name=name_1)
#                 model.addConstr(z_var[s, a, s1] - (P_hat[s, a, s1] - d_delta[s, a, s1]) * z_var.sum(s, a, '*') >= 0,
#                                 name=name_2)
#     # Valid occupancy measure
#     for s in range(S):
#         name = 'c_valid_measure_' + str(s)
#         model.addConstr(
#             z_var.sum(s, '*', '*') - (1 - env.gamma) * env.mu[s] - env.gamma * z_var.sum('*', '*', s) == 0,
#             name=name)
#
#     # Solve LP, if infeasible: return, the problem is infeasible
#     model.optimize()
#     if model.status == GRB.INFEASIBLE or model.status == GRB.INF_OR_UNBD:
#         raise
#         return False, -1, -1, -1
#     if model.status != GRB.OPTIMAL:
#         print('Status: ' + model.status)
#         raise
#     for s in range(S):
#         for a in range(A):
#             for s1 in range(S):
#                 z[s, a, s1] = z_var[s, a, s1].X
#     time_end = time.time()
#     print('Time: ' + str(time_end - time_start))
#
#     print('Checking...')
#
#     pi_hat = env.pi_z(z)
#     feasible_flag, max_cons_violat = env.check_pi_feasible_true_P(pi_hat, check_fineness)
#     Obj = env.Obj_pi(pi_hat)
#     return feasible_flag, pi_hat, z, max_cons_violat, Obj


def iter_baseline_experiment():
    gamma = args.gamma
    dim_Y = args.dim_Y
    S = args.S
    A = args.A
    grid_num = args.grid_num
    repeat_time = args.repeat
    epsilon = args.epsilon
    delta_prefix = args.delta_prefix
    pos_per_state = args.pos_per_state
    coeff = args.coeff
    baseline_extend = args.baseline_extend

    separation_len = 50

    fineness = int(grid_num ** (1. / dim_Y))
    check_fineness = args.check_fineness

    file = open('./iter_baseline.txt', 'w+')

    iter_upper_bound = 100

    delta = delta_prefix / (2. * S * S * A)

    n = math.ceil(64. * S * A * (np.log(S) ** 3.) * np.log(8 * (S ** 4.) * (A ** 3.) / delta) / ((epsilon ** 2.) * ((1 - gamma) ** 3.)))

    exp_time_lst_lst = []
    exp_Obj_lst_lst = []
    exp_max_cons_violat_lst_lst = []
    exp_error_lst_lst = []

    base_time_lst_lst = []
    base_Obj_lst_lst = []
    base_max_cons_violat_lst_lst = []
    base_error_lst_lst = []

    true_Obj_lst = []
    true_max_cons_violat_lst = []

    T_lst = []

    k = 0
    while True:
        Myprint('=' * separation_len, file)
        print('Repeat: ' + str(k))
        if k >= repeat_time:
            break
        env = random_complex_pollution_Env(S=S, A=A, pos_per_state=pos_per_state, dim_Y=dim_Y, coeff=coeff,
                                           gamma=gamma)
        _, _, _, _, true_max_cons_violat, true_Obj = env.SI_plan(iter_upper_bound, check_fineness, check_fineness)

        true_Obj_lst.append(true_Obj)
        true_max_cons_violat_lst.append(true_max_cons_violat)

        SA_array, SAS_array = env.sample_uniformly(n)
        _, pi_hat, z, Y0, max_cons_violat, Obj, exp_time_lst, exp_Obj_lst, exp_max_cons_violat_lst = \
            exp_SI_CRL(env, SA_array, SAS_array, delta, iter_upper_bound, fineness, check_fineness, silent_flag=False)
        T = len(Y0)

        T_lst.append(T)
        exp_time_lst_lst.append(np.cumsum(np.array(exp_time_lst)))
        exp_Obj_lst_lst.append(exp_Obj_lst)
        exp_max_cons_violat_lst_lst.append(exp_max_cons_violat_lst)

        exp_error_lst = np.maximum(np.maximum(true_Obj - exp_Obj_lst, 0.), np.maximum(exp_max_cons_violat_lst - true_max_cons_violat, 0.))
        exp_error_lst_lst.append(exp_error_lst)


        Obj_gap = true_Obj - Obj
        max_cons_violat_gap = max_cons_violat - max(true_max_cons_violat, 0.)

        if max(Obj_gap, max_cons_violat_gap) < epsilon:
            k += 1
        else:
            Myprint('Current k: ' + str(k), file)
            Myprint('Fail to converge with Obj gap: ' + str(Obj_gap) + ' Max vio gap: ' + str(max_cons_violat_gap), file)

        # Baseline
        baseline_fineness_ub = int(np.ceil(T ** (1. / dim_Y)) + baseline_extend)
        print(baseline_fineness_ub)

        base_time_lst = []
        base_Obj_lst = []
        base_max_cons_violat_lst = []
        for baseline_fineness in range(1, baseline_fineness_ub):
            _, base_pi_hat, base_z, base_max_cons_violat, base_Obj, base_time = \
                SI_CRL_baseline(env, SA_array, SAS_array, delta, baseline_fineness, check_fineness)
            base_time_lst.append(base_time)
            base_Obj_lst.append(base_Obj)
            base_max_cons_violat_lst.append(base_max_cons_violat)
        base_Obj_lst = np.array(base_Obj_lst)
        base_max_cons_violat_lst = np.array(base_max_cons_violat_lst)
        base_time_lst = np.array(base_time_lst)

        base_error_lst = np.maximum(np.maximum(true_Obj - base_Obj_lst, 0.), np.maximum(base_max_cons_violat_lst - true_max_cons_violat, 0.))
        base_error_lst_lst.append(base_error_lst)

        base_time_lst_lst.append(base_time_lst)
        base_Obj_lst_lst.append(base_Obj_lst)
        base_max_cons_violat_lst_lst.append(base_max_cons_violat_lst)


    Myprint('T lst: ' + str(T_lst) + '\n', file)
    Myprint('True Obj lst: ' + str(true_Obj_lst) + '\n', file)
    Myprint('True max violat lst: ' + str(true_max_cons_violat_lst) + '\n', file)
    Myprint('Exp time lst: ' + str(exp_time_lst_lst) + '\n', file)
    Myprint('Exp Obj lst: ' + str(exp_Obj_lst_lst) + '\n', file)
    Myprint('Exp max violat lst: ' + str(exp_max_cons_violat_lst_lst) + '\n', file)
    Myprint('Exp error lst: ' + str(exp_error_lst_lst) + '\n', file)
    Myprint('Base time lst: ' + str(base_time_lst_lst) + '\n', file)
    Myprint('Base Obj lst: ' + str(base_Obj_lst_lst) + '\n', file)
    Myprint('Base max violat lst: ' + str(base_max_cons_violat_lst_lst) + '\n', file)
    Myprint('Base error lst: ' + str(base_error_lst_lst) + '\n', file)

    Myprint('=' * separation_len, file)
    file.close()
    return


if __name__ == '__main__':
    np.random.seed(4)
    iter_baseline_experiment()
