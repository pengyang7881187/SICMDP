import math
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from SICMDP import *
from utils import regress, Myprint
from numpy import sqrt
from pollution_env import random_complex_pollution_Env
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)



def heat_experiment():
    S = 8
    A = 4
    dim_Y = 2
    gamma = 0.9
    epsilon = 0.01
    delta_prefix = 0.01
    delta = delta_prefix / (2. * S * S * A)
    pos_per_state = 1
    coeff = 1 + 1e-6

    baseline_fineness = 2
    SICRL_iter_time = 9

    fineness = 1000
    check_fineness = 1000

    n = math.ceil(64. * S * A * (np.log(S) ** 3.) * np.log(8 * (S ** 4.) * (A ** 3.) / delta) / (
                 (epsilon ** 2.) * ((1 - gamma) ** 3.)))
    env = random_complex_pollution_Env(S=S, A=A, pos_per_state=pos_per_state, dim_Y=dim_Y, coeff=coeff,
                                       gamma=gamma)
    np.save('./fallout_pos', env.state_coordinates)
    SA_array, SAS_array = env.sample_uniformly(n)
    feasible_flag, pi_hat, z, Y0, max_cons_violat, Obj = SI_CRL(env, SA_array, SAS_array, delta,
                                                                SICRL_iter_time, fineness, check_fineness)
    assert len(Y0) == SICRL_iter_time + 1
    np.save('./detect_pos', np.array(Y0))
    _, base_pi_hat, base_z, base_max_cons_violat, base_Obj, _ = \
        SI_CRL_baseline(env, SA_array, SAS_array, delta, baseline_fineness, check_fineness)

    exp_heat_map = env.generate_heat_map(pi_hat, check_fineness)
    baseline_heat_map = env.generate_heat_map(base_pi_hat, check_fineness)
    print('exp')
    print(np.min(exp_heat_map))
    print(np.max(exp_heat_map))
    print(np.average(exp_heat_map))
    print('baseline')
    print(np.min(baseline_heat_map))
    print(np.max(baseline_heat_map))
    print(np.average(baseline_heat_map))
    print('error')  # Bigger the (positive) error, better the exp
    print(np.average(baseline_heat_map - exp_heat_map))

    np.save('./heat_exp', exp_heat_map)
    np.save('./heat_base', baseline_heat_map)
    return


# def transform_grid_date():
#     exp_heat_map = np.load('./heat_exp.npy')
#     baseline_heat_map = np.load('./heat_base.npy')
#
#
#     np.save('./transform_heat_exp', exp_heat_map)
#     np.save('./transform_heat_base', baseline_heat_map)
#     return




def draw():
    exp_heat_map = np.load('./heat_exp.npy')
    baseline_heat_map = np.load('./heat_base.npy')

    grid_num = 1001
    exp_heat_map = exp_heat_map.reshape(grid_num, grid_num)
    baseline_heat_map = baseline_heat_map.reshape(grid_num, grid_num)

    # P_img = rescale(P_MEOBj_lst.T[P_ylabels_numerical], gamma=0.3)

    fig, ax = plt.subplots(figsize=(8, 8))

    green = plt.cm.get_cmap('Wistia', 512)
    # new_green = ListedColormap(green(np.linspace(0, 0.8, 256)))

    im = ax.imshow(exp_heat_map, cmap=green)
    # ax.tick_params(axis="x", bottom=False, top=True, labeltop=True, labelbottom=False)

    cbar = plt.colorbar(im)
    # cbar.ax.tick_params(labelsize=0)

    ############ Baseline


    plt.show()

    fig, ax = plt.subplots(figsize=(8, 8))

    # green = plt.cm.get_cmap('Wistia', 512)
    # new_green = ListedColormap(green(np.linspace(0, 0.8, 256)))

    im = ax.imshow(baseline_heat_map, cmap='bone')
    # ax.tick_params(axis="x", bottom=False, top=True, labeltop=True, labelbottom=False)

    cbar = plt.colorbar(im, orientation="horizontal")
    # cbar.ax.tick_params(labelsize=0)

    plt.show()

    return



def transform(x):
    # return sqrt(x)
    # return sqrt(sqrt(x))
    bias = 5e-6
    return np.log(bias + x) - np.log(bias)
    # return x


def new_draw():
    exp_heat_map = np.load('./heat_exp.npy')
    baseline_heat_map = np.load('./heat_base.npy')

    fallout_pos_lst = np.load('./fallout_pos.npy')[:, 0, :] / 2.
    exp_detect_pos_lst = np.load('./detect_pos.npy') / 2.
    base_detect_pos_lst = np.array([[0., 0.], [0., .5], [0., 1.],
                                    [.5, 0.], [.5, .5], [.5, 1.],
                                    [1., 0.], [1., .5], [1., 1.]])

    grid_num = 1001

    subsample = 10

    exp_heat_map = transform(exp_heat_map.reshape(grid_num, grid_num))[::subsample, ::subsample]
    baseline_heat_map = transform(baseline_heat_map.reshape(grid_num, grid_num))[::subsample, ::subsample]

    val_max = np.max(baseline_heat_map)

    fallout_img = mpimg.imread('./fallout.png')
    detect_img = mpimg.imread('./detect.png')

    zoom = 0.06
    dpi = 400
    fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[4, 4, 0.2]), figsize=(16, 8), dpi=dpi)
    # fig, axs = plt.subplots(ncols=3, figsize=(16, 8), dpi=dpi)
    plt.subplots_adjust(wspace=100000)
    ax1 = sns.heatmap(data=exp_heat_map,
                      cmap=plt.get_cmap('winter'),
                      vmax=val_max,
                      xticklabels=False,
                      yticklabels=False,
                      cbar=False,
                      square=True,
                      ax=axs[0])
    ax1.tick_params(left=False, bottom=False)
    for fallout_pos in fallout_pos_lst:
        fallout_imagebox = OffsetImage(fallout_img, zoom=zoom)
        fallout_imagebox.image.axes = ax1
        xy = (fallout_pos[0], fallout_pos[1])
        fallout_ab = AnnotationBbox(fallout_imagebox, xy,
                                    xybox=xy,
                                    xycoords='axes fraction',
                                    pad=0.0)
        ax1.add_artist(fallout_ab)
    for exp_detect_pos in exp_detect_pos_lst:
        detect_imagebox = OffsetImage(detect_img, zoom=zoom)
        detect_imagebox.image.axes = ax1
        xy = (exp_detect_pos[0], exp_detect_pos[1])
        detect_ab = AnnotationBbox(detect_imagebox, xy,
                                   xybox=xy,
                                   xycoords='axes fraction',
                                   pad=0.0)
        ax1.add_artist(detect_ab)

    ax2 = sns.heatmap(data=baseline_heat_map,
                      cmap=plt.get_cmap('winter'),
                      vmax=val_max,
                      xticklabels=False,
                      yticklabels=False,
                      cbar=False,
                      square=True,
                      ax=axs[1])
    for fallout_pos in fallout_pos_lst:
        fallout_imagebox = OffsetImage(fallout_img, zoom=zoom)
        fallout_imagebox.image.axes = ax2
        xy = (fallout_pos[0], fallout_pos[1])
        fallout_ab = AnnotationBbox(fallout_imagebox, xy,
                                    xybox=xy,
                                    xycoords='axes fraction',
                                    pad=0.0)
        ax2.add_artist(fallout_ab)
    for base_detect_pos in base_detect_pos_lst:
        detect_imagebox = OffsetImage(detect_img, zoom=zoom)
        detect_imagebox.image.axes = ax2
        xy = (base_detect_pos[0], base_detect_pos[1])
        detect_ab = AnnotationBbox(detect_imagebox, xy,
                                   xybox=xy,
                                   xycoords='axes fraction',
                                   pad=0.0)
        ax2.add_artist(detect_ab)
    ax2.tick_params(left=False, bottom=False)
    fig.colorbar(axs[1].collections[0], cax=axs[2])

    plt.tight_layout()
    plt.savefig("./figure/heat.pdf", format='pdf', dpi=dpi, bbox_inches='tight')
    print('Save complete')
    plt.show()
    return



def draw_one_by_one():
    exp_heat_map = np.load('./heat_exp.npy')
    baseline_heat_map = np.load('./heat_base.npy')

    fallout_pos_lst = np.load('./fallout_pos.npy')[:, 0, :] / 2.
    exp_detect_pos_lst = np.load('./detect_pos.npy') / 2.
    base_detect_pos_lst = np.array([[0., 0.], [0., .5], [0., 1.],
                                    [.5, 0.], [.5, .5], [.5, 1.],
                                    [1., 0.], [1., .5], [1., 1.]])

    grid_num = 1001

    subsample = 2

    exp_heat_map = transform(exp_heat_map.reshape(grid_num, grid_num))[::subsample, ::subsample]
    baseline_heat_map = transform(baseline_heat_map.reshape(grid_num, grid_num))[::subsample, ::subsample]

    val_max = np.max(baseline_heat_map)

    fallout_img = mpimg.imread('./fallout.png')
    detect_img = mpimg.imread('./detect.png')

    zoom = 0.03
    dpi = 400
    # fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[4, 4, 0.2]), figsize=(16, 8), dpi=dpi)
    fig, axs = plt.subplots(ncols=1, figsize=(4, 4), dpi=dpi)
    # plt.subplots_adjust(wspace=100000)
    ax1 = sns.heatmap(data=exp_heat_map,
                      cmap=plt.get_cmap('winter'),
                      vmax=val_max,
                      xticklabels=False,
                      yticklabels=False,
                      cbar=False,
                      square=True,
                      ax=axs)
    ax1.tick_params(left=False, bottom=False)
    for fallout_pos in fallout_pos_lst:
        fallout_imagebox = OffsetImage(fallout_img, zoom=zoom)
        fallout_imagebox.image.axes = ax1
        xy = (fallout_pos[0], fallout_pos[1])
        fallout_ab = AnnotationBbox(fallout_imagebox, xy,
                                    xybox=xy,
                                    xycoords='axes fraction',
                                    pad=0.0)
        ax1.add_artist(fallout_ab)
    for exp_detect_pos in exp_detect_pos_lst:
        detect_imagebox = OffsetImage(detect_img, zoom=zoom)
        detect_imagebox.image.axes = ax1
        xy = (exp_detect_pos[0], exp_detect_pos[1])
        detect_ab = AnnotationBbox(detect_imagebox, xy,
                                   xybox=xy,
                                   xycoords='axes fraction',
                                   pad=0.0)
        ax1.add_artist(detect_ab)
    plt.tight_layout()
    plt.savefig("./figure/heat_exp.pdf", format='pdf', dpi=dpi, bbox_inches='tight')
    print('Save complete')



    fig, axs = plt.subplots(ncols=2, gridspec_kw=dict(width_ratios=[20, 1]), figsize=(4.5, 4), dpi=dpi)
    ax2 = sns.heatmap(data=baseline_heat_map,
                      cmap=plt.get_cmap('winter'),
                      vmax=val_max,
                      xticklabels=False,
                      yticklabels=False,
                      cbar=False,
                      cbar_kws={"shrink": .80},
                      square=True,
                      ax=axs[0])

    cb = ax2.figure.colorbar(ax2.collections[0], cax=axs[1])
    cb.ax.tick_params(labelsize=20)


    for fallout_pos in fallout_pos_lst:
        fallout_imagebox = OffsetImage(fallout_img, zoom=zoom)
        fallout_imagebox.image.axes = ax2
        xy = (fallout_pos[0], fallout_pos[1])
        fallout_ab = AnnotationBbox(fallout_imagebox, xy,
                                    xybox=xy,
                                    xycoords='axes fraction',
                                    pad=0.0)
        ax2.add_artist(fallout_ab)
    for base_detect_pos in base_detect_pos_lst:
        detect_imagebox = OffsetImage(detect_img, zoom=zoom)
        detect_imagebox.image.axes = ax2
        xy = (base_detect_pos[0], base_detect_pos[1])
        detect_ab = AnnotationBbox(detect_imagebox, xy,
                                   xybox=xy,
                                   xycoords='axes fraction',
                                   pad=0.0)
        ax2.add_artist(detect_ab)
    ax2.tick_params(left=False, bottom=False)
    plt.tight_layout()
    plt.savefig("./figure/heat_base.pdf", format='pdf', dpi=dpi, bbox_inches='tight')
    print('Save complete')
    # plt.show()
    return





if __name__ == '__main__':
    # np.random.seed(4)  # 4->S=16
    np.random.seed(8)
    # heat_experiment()
    # new_draw()
    draw_one_by_one()