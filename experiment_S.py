import math
import argparse
from SICMDP import *
from utils import regress, Myprint
from pollution_env import random_complex_pollution_Env

parser = argparse.ArgumentParser()
parser.add_argument("--A_for_S_exp", default=4, type=int)
parser.add_argument("--dim_Y", default=2, type=int)
parser.add_argument("--repeat", default=30, type=int)
parser.add_argument("--epsilon", default=0.015, type=float)
parser.add_argument("--fineness", default=int(3e+2), type=int)
parser.add_argument("--delta_prefix", default=0.01, type=float)
parser.add_argument("--n_init", default=int(2. ** 15.), type=int)
parser.add_argument("--pos_per_state", default=1, type=int)
parser.add_argument("--gamma", default=0.9, type=float)
parser.add_argument("--coeff", default=1+1e-6, type=float)

args = parser.parse_args()

A_for_S_exp = args.A_for_S_exp
dim_Y = args.dim_Y
repeat_time = args.repeat
epsilon = args.epsilon
fineness = args.fineness
delta_prefix = args.delta_prefix
n_init = args.n_init
pos_per_state = args.pos_per_state
gamma = args.gamma
coeff = args.coeff

separation_len = 50

check_fineness = 4 * fineness


def generate_discret_n_lst(ub=12, discret=11):
    n_lst = np.logspace(1, ub, ub, base=2).astype(int)
    discret_n_lst = np.array([], dtype=int)
    for i in range(ub-1):
        discret_n_lst = np.append(discret_n_lst, (n_init * np.linspace(n_lst[i], n_lst[i+1], discret)).astype(int))
    return np.unique(discret_n_lst)


def S_experiment():
    file = open('./S_final.txt', 'w+')
    A = A_for_S_exp
    S_lst = np.arange(2, 17, 2)
    S_lst_len = S_lst.size
    valid_n_lst = np.zeros((S_lst_len, repeat_time))

    min_n_index = -1

    n_lst = generate_discret_n_lst(15)
    n_lst_len = len(n_lst)

    for i in range(S_lst_len):
        current_n_lst = []
        S = S_lst[i]
        delta = delta_prefix / (2. * S * S * A)

        iter_upper_bound = 1000

        Myprint('=' * separation_len, file)
        Myprint('S: ' + str(S_lst[i]), file)

        sum_n = 0.
        sum_true_Obj = 0.
        sum_true_max_cons_violat = 0.
        sum_Obj_gap = 0.
        sum_max_cons_violat_gap = 0.

        Obj_gap = -1.
        max_cons_violat_gap = -1.

        ref_n = math.ceil(64. * S * A * (np.log(S) ** 3.) * np.log(8 * (S ** 4.) * (A ** 3.) / delta) / (
                (epsilon ** 2.) * ((1 - gamma) ** 3.)))

        j = 0
        while True:
            if j >= repeat_time:
                break
            env = random_complex_pollution_Env(S=S, A=A, pos_per_state=pos_per_state, dim_Y=dim_Y, coeff=coeff,
                                               gamma=gamma)
            _, _, _, _, true_max_cons_violat, true_Obj = env.SI_plan(iter_upper_bound, check_fineness, check_fineness)

            sum_true_Obj += true_Obj
            sum_true_max_cons_violat += true_max_cons_violat

            success_flag = False
            for n_idx in range(n_lst_len):
                if n_idx < min_n_index - 2:
                    continue
                n = n_lst[n_idx]
                SA_array, SAS_array = env.sample_uniformly(n)
                feasible_flag, pi_hat, z, Y0, max_cons_violat, Obj = SI_CRL(env, SA_array, SAS_array, delta,
                                                                            iter_upper_bound, fineness, check_fineness)
                Obj_gap = true_Obj - Obj
                max_cons_violat_gap = max_cons_violat - max(true_max_cons_violat, 0.)
                if max(Obj_gap, max_cons_violat_gap) < epsilon:
                    success_flag = True
                    valid_n_lst[i, j] = np.log2(n)
                    current_n_lst.append(n_idx)
                    sum_n += n
                    sum_Obj_gap += Obj_gap
                    sum_max_cons_violat_gap += max_cons_violat_gap
                    j += 1
                    break
                else:
                    pass
                    # Myprint('Fail')
                    # Myprint('True Obj val: ' + str(env.true_Obj))
                    # Myprint('Val gap: ' + str(val_gap))
                    # Myprint('Con gap: ' + str(max_cons_violat))
            if not success_flag:
                Myprint('Current j: ' + str(j), file)
                Myprint('Fail to converge with Obj gap: ' + str(Obj_gap) + ' Max vio gap: ' + str(max_cons_violat_gap), file)
        min_n_index = np.min(np.array(current_n_lst, dtype=int))

        avg_n = sum_n / repeat_time
        avg_Obj_gap = sum_Obj_gap / repeat_time
        avg_max_cons_violat_gap = sum_max_cons_violat_gap / repeat_time

        avg_true_Obj = sum_true_Obj / repeat_time
        avg_true_max_cons_violat = sum_true_max_cons_violat / repeat_time

        Myprint('log_2 avg n: ' + str(np.log2(avg_n)), file)
        Myprint('avg n: ' + str(avg_n), file)
        Myprint('ref n: ' + str(ref_n), file)
        Myprint('ratio: ' + str(ref_n / avg_n), file)
        Myprint('Avg true Obj val: ' + str(avg_true_Obj), file)
        Myprint('Avg true max constraint violation: ' + str(avg_true_max_cons_violat), file)
        Myprint('Avg Obj val gap: ' + str(avg_Obj_gap), file)
        Myprint('Avg max constraint violation gap: ' + str(avg_max_cons_violat_gap), file)
        Myprint('=' * separation_len, file)
    k, b = regress(S_lst, valid_n_lst)
    np.save('./exp_S_log2_n', valid_n_lst)  # log
    Myprint(valid_n_lst, file)
    Myprint(k, file)
    Myprint(b, file)
    file.close()
    return


if __name__ == '__main__':
    np.random.seed(2)
    S_experiment()
