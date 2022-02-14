import math
import argparse
from SICMDP import *
from utils import regress, Myprint
from numpy import sqrt
from pollution_env import random_complex_pollution_Env

parser = argparse.ArgumentParser()
parser.add_argument("--gamma", default=0.9, type=int)
parser.add_argument("--A", default=4, type=int)
parser.add_argument("--grid_num", default=int(1e+5), type=int)
parser.add_argument("--repeat", default=30, type=int)
parser.add_argument("--epsilon", default=0.015, type=float)
parser.add_argument("--delta_prefix", default=0.01, type=float)
parser.add_argument("--pos_per_state", default=1, type=int)
parser.add_argument("--coeff", default=1 + 1e-6, type=float)

args = parser.parse_args()

gamma = args.gamma
A = args.A
grid_num = args.grid_num
repeat_time = args.repeat
epsilon = args.epsilon
delta_prefix = args.delta_prefix
pos_per_state = args.pos_per_state
coeff = args.coeff

separation_len = 50

dim_Y_lst = [1, 2, 3]
dim_Y_len = 3
fineness_lst = np.array([grid_num, int(grid_num ** 0.5), int(grid_num ** (1./3.))], dtype=int)
check_fineness_lst = 4 * fineness_lst


def iter_S_experiment():
    f = open('./iter_S.txt', 'w+')
    S_lst = np.arange(2, 17, 2)
    S_lst_len = S_lst.size
    valid_T_lst = np.zeros((dim_Y_len, S_lst_len, repeat_time))

    iter_upper_bound = 1000

    k_lst = np.zeros((dim_Y_len))
    b_lst = np.zeros((dim_Y_len))
    k_log_lst = np.zeros((dim_Y_len))
    b_log_lst = np.zeros((dim_Y_len))

    for i in range(dim_Y_len):
        Myprint('=' * separation_len * 2, f)
        dim_Y = dim_Y_lst[i]
        Myprint('dim Y: ' + str(dim_Y), f)
        fineness = fineness_lst[i]
        check_fineness = check_fineness_lst[i]

        for j in range(S_lst_len):
            S = S_lst[j]
            delta = delta_prefix / (2. * S * S * A)

            n = math.ceil(64. * S * A * (np.log(S) ** 3.) * np.log(8 * (S ** 4.) * (A ** 3.) / delta) / ((epsilon ** 2.) * ((1 - gamma) ** 3.)))

            Myprint('=' * separation_len, f)
            Myprint('S: ' + str(S_lst[j]), f)

            sum_T = 0

            sum_true_Obj = 0.
            sum_true_max_cons_violat = 0.

            sum_Obj_gap = 0.
            sum_max_cons_violat_gap = 0.

            Obj_gap = -1.
            max_cons_violat_gap = -1.

            k = 0
            while True:
                if k >= repeat_time:
                    break
                env = random_complex_pollution_Env(S=S, A=A, pos_per_state=pos_per_state, dim_Y=dim_Y, coeff=coeff,
                                                   gamma=gamma)
                _, _, _, _, true_max_cons_violat, true_Obj = env.SI_plan(iter_upper_bound, check_fineness, check_fineness)

                sum_true_Obj += true_Obj
                sum_true_max_cons_violat += true_max_cons_violat

                SA_array, SAS_array = env.sample_uniformly(n)
                feasible_flag, pi_hat, z, Y0, max_cons_violat, Obj = SI_CRL(env, SA_array, SAS_array, delta,
                                                                            iter_upper_bound, fineness, check_fineness)
                T = len(Y0)
                Obj_gap = true_Obj - Obj
                max_cons_violat_gap = max_cons_violat - max(true_max_cons_violat, 0.)
                if max(Obj_gap, max_cons_violat_gap) < epsilon:
                    valid_T_lst[i, j, k] = np.log2(T)
                    sum_T += T
                    sum_Obj_gap += Obj_gap
                    sum_max_cons_violat_gap += max_cons_violat_gap
                    k += 1
                else:
                    Myprint('Current k: ' + str(k), f)
                    Myprint('Fail to converge with Obj gap: ' + str(Obj_gap) + ' Max vio gap: ' + str(max_cons_violat_gap), f)

            avg_T = sum_T / repeat_time
            avg_Obj_gap = sum_Obj_gap / repeat_time
            avg_max_cons_violat_gap = sum_max_cons_violat_gap / repeat_time

            avg_true_Obj = sum_true_Obj / repeat_time
            avg_true_max_cons_violat = sum_true_max_cons_violat / repeat_time

            Myprint('log_2 avg T: ' + str(np.log2(avg_T)), f)
            Myprint('avg T: ' + str(avg_T), f)
            Myprint('Avg true Obj val: ' + str(avg_true_Obj), f)
            Myprint('Avg true max constraint violation: ' + str(avg_true_max_cons_violat), f)
            Myprint('Avg Obj val gap: ' + str(avg_Obj_gap), f)
            Myprint('Avg max constraint violation gap: ' + str(avg_max_cons_violat_gap), f)
        k, b = regress(S_lst, valid_T_lst[i, :, :])
        k_lst[i] = k
        b_lst[i] = b
        k_log, b_log = regress(np.log2(S_lst), valid_T_lst[i, :, :])
        k_log_lst[i] = k_log
        b_log_lst[i] = b_log
        Myprint(k, f)
        Myprint(b, f)
        Myprint(k_log, f)
        Myprint(b_log, f)
        Myprint('=' * separation_len, f)
    np.save('./exp_iter_S_log2_T', valid_T_lst)
    Myprint(valid_T_lst, f)
    Myprint(k_lst, f)
    Myprint(b_lst, f)
    Myprint(k_log_lst, f)
    Myprint(b_log_lst, f)
    f.close()
    return


if __name__ == '__main__':
    np.random.seed(2)
    iter_S_experiment()
