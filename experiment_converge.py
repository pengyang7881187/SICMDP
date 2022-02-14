import math
import argparse
from numpy import sqrt
from SICMDP import *
from utils import regress
from toymdp_env import toymdp_Env

parser = argparse.ArgumentParser()
parser.add_argument("--repeat", default=30, type=int)
parser.add_argument("--gamma", default=0.9, type=float)
parser.add_argument("--fineness", default=int(1e+5), type=int)
parser.add_argument("--delta_prefix", default=0.01, type=float)
parser.add_argument("--p", default=0.95, type=float)
parser.add_argument("--complexity", default=1, type=int)

args = parser.parse_args()

repeat_time = args.repeat
gamma = args.gamma
fineness = args.fineness
delta_prefix = args.delta_prefix
p = args.p
complexity = args.complexity

separation_len = 50


def gamma_experiment():

    S = 2
    A = 2
    delta = delta_prefix / (2. * S * S * A)
    dim_Y = 1
    n_lst_len = 20
    log_n_init = 4
    n_lst = np.logspace(log_n_init, n_lst_len + log_n_init - 1, n_lst_len, base=2)

    env = toymdp_Env(p, gamma, complexity=complexity)

    error_lst = np.zeros((n_lst_len, repeat_time))

    for i in range(n_lst_len):
        n = n_lst[i]

        iter_upper_bound = 1000

        print('=' * separation_len)
        print('n: ' + str(n))

        sum_Obj_gap = 0.
        sum_max_cons_violat_gap = 0.
        sum_error = 0.

        j = 0
        while True:
            if j >= repeat_time:
                break
            SA_array, SAS_array = env.sample_uniformly(n)
            feasible_flag, pi_hat, z, Y0, max_cons_violat, Obj = SI_CRL(env, SA_array, SAS_array, delta, iter_upper_bound, fineness, 2)
            Obj_gap = env.true_Obj - Obj
            error = max(Obj_gap, max_cons_violat)
            if not np.isinf(max_cons_violat):
                sum_Obj_gap += Obj_gap
                sum_max_cons_violat_gap += max_cons_violat
                sum_error += error
                error_lst[i, j] = error
                j += 1
            else:
                print('Infeasible')

        avg_Obj_gap = sum_Obj_gap / repeat_time
        avg_max_cons_violat_gap = sum_max_cons_violat_gap / repeat_time
        avg_error = sum_error / repeat_time

        print('True Obj val: ' + str(env.true_Obj))
        print('Avg Obj gap: ' + str(avg_Obj_gap))
        print('Avg max constraint violation gap: ' + str(avg_max_cons_violat_gap))
        print('Avg error: ' + str(avg_error))
        print('=' * separation_len)
    k, b = regress(n_lst, np.log2(error_lst))
    np.save('./converge_log2_n', np.log2(n_lst))
    np.save('./converge_avg_error', np.average(error_lst, axis=1))
    print(error_lst)
    print(k)
    print(b)
    return


if __name__ == '__main__':
    np.random.seed(74750)
    gamma_experiment()
