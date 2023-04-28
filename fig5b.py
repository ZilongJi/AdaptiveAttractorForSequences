import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import TwoD_fun
import TwoD_gamma
import brainpy.math as bm
from scipy import optimize
import random

ticksize = 14
labelsize = 18
position_color = '#F18D00'
bump_color = '#009FB9'


def f_1(x, A, B):
    return A * x + B


def plot_6d(simulation=[0, 0, 0, 0, 0]):
    def get_mean_var(x, y, interval):
        delta = np.sqrt((x[interval:] - x[:-interval]) ** 2 + (y[interval:] - y[:-interval]) ** 2)
        return [[np.mean(delta), np.std(delta)]]

    def plot_mean_var(mu, gamma, simulation, label, ax, sigma_u=0.5):
        if simulation == 1:
            center_trace = bm.as_numpy(TwoD_fun.get_trace(mu, gamma, 10, 0.2, 1, 1, sigma_u=sigma_u))
            np.save('./data/fig6b_brownian' + str(mu) + '_' + str(gamma) + '.npy', center_trace)

        center_trace = np.load('./data/fig6b_brownian' + str(mu) + '_' + str(gamma) + '.npy')
        center_shuffle = random.shuffle
        ans = []
        ans_shuffle = []
        interval_list = np.array(range(1, 100, 10))
        for interval in interval_list:
            ans += get_mean_var(center_trace[200:-1, 0], center_trace[200:-1, 1], interval)
            ans_shuffle += get_mean_var(center_shuffle[200:-1, 0], center_shuffle[200:-1, 1], interval)
        ans = np.array(ans)
        ans_shuffle = np.array(ans_shuffle)
        x0 = np.log(interval_list)
        y0 = np.log(ans[:, 0])
        A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]
        x1 = np.log(interval_list) + 0.5
        y1 = A1 * x1 + B1
        ax.plot(np.exp(x1), np.exp(y1), '--', color='yellow', alpha=0.5)
        ax.errorbar(interval_list, ans[:, 0], ans[:, 1], label=r'$~\alpha = $' + str(round(A1, 2)), color='blue',
                    capsize=2)

        ax.errorbar(interval_list, ans_shuffle[:, 0], ans_shuffle[:, 1], label=r'$~\alpha = $' + str(round(A1, 2)),
                    color='blue',
                    capsize=2)

    plt.figure(figsize=(4, 3), dpi=300)
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.0001, 10)
    # remove the top and right axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # replace the y ticks from meter to cm
    ax.set_yticks([0.0001, 0.001, 0.01, 0.1, 1, 10])
    ax.set_yticklabels(['$10^{-2}$', '$10^{-1}$', '$10^0$', '$10^1$', '$10^2$', '$10^3$'])

    plot_mean_var(0.9, 2, simulation[0], 0, ax)
    plot_mean_var(0.9, 1.5, simulation[1], 2, ax)
    plot_mean_var(0.9, 1, simulation[2], 1, ax)
    plot_mean_var(0.9, 0.5, simulation[3], 2, ax)
    plot_mean_var(0.9, 0.1, simulation[4], 2, ax)

    plt.legend(fontsize=ticksize / 1.5, frameon=False, loc='upper left')
    plt.xlabel(r'Time-step interval $(\Delta t/\tau)$', fontsize=labelsize)
    plt.ylabel('Mean distance (cm)', fontsize=labelsize)

    plt.tight_layout()
    plt.savefig('./Figures/Fig5b.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_6d(simulation=[1, 1, 1, 1, 1])