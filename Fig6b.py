import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import TwoD_fun
import brainpy.math as bm
from scipy import optimize

xlabel = ['Traveling wave', 'Super-diffusion', 'Brownian-diffusion', 'Stationary']
color_list = ['r','g','b','orange']
ticksize = 14
labelsize = 18
linewidth = 1


def f_1(x, A, B):
    return A*x + B


def plot_6b(simulation=[0, 0, 0, 0]):
    def get_mean_var(x,y,interval):
        delta = np.sqrt((x[interval:]-x[:-interval])**2 + (y[interval:]-y[:-interval])**2)
        return [[np.mean(delta),np.std(delta)]]

    def plot_mean_var(mu, gamma, simulation, label, ax, sigma_u=0.5):
        if simulation == 1:
            center_trace = bm.as_numpy(TwoD_fun.get_trace(mu, gamma, 10, 0.2, 1, 1, sigma_u=sigma_u))
            np.save('./data/fig6b_center_trace' + str(mu) + '_' + str(gamma) + '.npy', center_trace)

        center_trace = np.load('./data/fig6b_center_trace' + str(mu) + '_' + str(gamma) + '.npy')

        ans = []
        interval_list = np.array(range(1,100,10))
        for interval in interval_list:
            ans += get_mean_var(center_trace[200:-1, 0],center_trace[200:-1, 1],interval)
        ans = np.array(ans)

        x0 = np.log(interval_list)
        y0 = np.log(ans[:,0])
        A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]
        x1 = np.log(interval_list) + 0.5
        y1 = A1 * x1 + B1
        ax.plot(np.exp(x1), np.exp(y1), '--', color = color_list[label], alpha = 0.5)

        ax.errorbar(interval_list,ans[:,0],ans[:,1],label = xlabel[label] + r'$~\alpha = $' + str(round(A1,2)), color = color_list[label])

    plt.figure(figsize=(4, 3), dpi=300)
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    #plot_mean_var(-0.5, 0.001, simulation[0],  0, ax, sigma_u=0.05)
    plot_mean_var(0.1, 2, simulation[1],  1, ax)
    plot_mean_var(0.5, 0.1, simulation[2],  2, ax)
    plot_mean_var(0.9, 0.1, simulation[3],  3, ax)

    plt.legend(fontsize = ticksize/2, frameon=False, loc='upper left')
    plt.xlabel(r'Time-step interval $(\Delta t/\tau)$',fontsize = labelsize)
    plt.ylabel('Mean displacement',fontsize = labelsize)
    #plt.xticks([0,50,100],fontsize = ticksize)
    #plt.yticks([0,0.03,0.06],fontsize=ticksize)


    plt.savefig('./Figures/Fig6b.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_6b(simulation=[0,0,0,0])
