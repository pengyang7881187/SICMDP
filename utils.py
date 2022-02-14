import numpy as np


# This function is not effective, it is only for small data
def regress(para_lst, valid_n_lst):
    para_lst_len = valid_n_lst.shape[0]
    repeat_n_len = valid_n_lst.shape[1]
    log_para_lst = np.log2(para_lst)
    x = np.zeros((para_lst_len * repeat_n_len), dtype=float)
    y = np.zeros((para_lst_len * repeat_n_len), dtype=float)
    current_index = 0
    for i in range(para_lst_len):
        for j in range(repeat_n_len):
            x[current_index] = log_para_lst[i]
            y[current_index] = valid_n_lst[i, j]
            current_index += 1
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x = x - x_mean
    y = y - y_mean
    k = np.sum(x * y) / np.sum(x * x)
    b = y_mean - k * x_mean
    return k, b


def Myprint(s, f):
    s_str = str(s)
    f.write(s_str + '\n')
    print(s_str)
    return