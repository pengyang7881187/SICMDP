import math
import argparse
from numpy import sqrt
from SICMDP import *
from utils import regress
from toymdp_env import toymdp_Env

parser = argparse.ArgumentParser()
parser.add_argument("--repeat", default=30, type=int)
parser.add_argument("--epsilon", default=0.01, type=float)
parser.add_argument("--fineness", default=int(1e+5), type=int)
parser.add_argument("--delta_prefix", default=0.01, type=float)
parser.add_argument("--p", default=0.95, type=float)
parser.add_argument("--complexity", default=1, type=int)
parser.add_argument("--n_init", default=16, type=int)
args = parser.parse_args()

repeat_time = args.repeat
epsilon = args.epsilon
fineness = args.fineness
delta_prefix = args.delta_prefix
p = args.p
complexity = args.complexity
n_init = args.n_init

separation_len = 50


def gamma_experiment():
    S = 2
    A = 2
    delta = delta_prefix / (2. * S * S * A)
    dim_Y = 1
    transform_gamma_lst = np.logspace(2, 26, 25, base=sqrt(2))
    gamma_lst = 1 - ((1. / transform_gamma_lst) ** (1./3.))
    gamma_lst_len = np.size(gamma_lst)
    valid_n_lst = np.zeros((gamma_lst_len, repeat_time))

    for i in range(gamma_lst_len):
        gamma = gamma_lst[i]
        env = toymdp_Env(p, gamma, complexity=complexity)

        ref_n = math.ceil(64. * S * A * (np.log(S) ** 3.) * np.log(8 * (S ** 4.) * (A ** 3.) / delta) / ((epsilon ** 2.) * ((1 - env.gamma) ** 3.)))

        iter_upper_bound = 1000

        print('=' * separation_len)
        print('transform gamma: ' + str(transform_gamma_lst[i]))
        print('gamma: ' + str(gamma_lst[i]))

        sum_n = 0.
        sum_Obj_gap = 0.
        sum_max_cons_violat_gap = 0.

        for j in range(repeat_time):
            success_flag = False
            for k in range(1000):
                n = int(n_init * (sqrt(sqrt(2)) ** k))
                SA_array, SAS_array = env.sample_uniformly(n)
                feasible_flag, pi_hat, z, Y0, max_cons_violat, Obj = SI_CRL(env, SA_array, SAS_array, delta, iter_upper_bound, fineness, 2)
                Obj_gap = env.true_Obj - Obj
                if max(Obj_gap, max_cons_violat) < epsilon:
                    success_flag = True
                    valid_n_lst[i, j] = np.log2(n)
                    sum_n += n
                    sum_Obj_gap += Obj_gap
                    sum_max_cons_violat_gap += max_cons_violat
                    break
                else:
                    pass
                    # print('Fail')
                    # print('True Obj val: ' + str(env.true_Obj))
                    # print('Obj gap: ' + str(Obj_gap))
                    # print('Max contraint violation gap: ' + str(max_cons_violat))
            if not success_flag:
                warnings.warn('Fail to converge with Obj gap: ' + str(Obj_gap) + ' Max vio gap: ' + str(max_cons_violat))


        avg_n = sum_n / repeat_time
        avg_Obj_gap = sum_Obj_gap / repeat_time
        avg_max_cons_violat_gap = sum_max_cons_violat_gap / repeat_time

        print('True Obj val: ' + str(env.true_Obj))
        print('Avg Obj gap: ' + str(avg_Obj_gap))
        print('Avg max constraint violation gap: ' + str(avg_max_cons_violat_gap))
        print('log_2 avg n: ' + str(np.log2(avg_n)))
        print('avg n: ' + str(avg_n))
        print('ref n: ' + str(ref_n))
        print('ratio: ' + str(ref_n / avg_n))
        print('=' * separation_len)
    k, b = regress(transform_gamma_lst, valid_n_lst)
    np.save('./exp_gamma_log2_n', valid_n_lst)  # log2
    print(valid_n_lst)
    print(k)
    print(b)
    return


if __name__ == '__main__':
    np.random.seed(74750)
    gamma_experiment()
