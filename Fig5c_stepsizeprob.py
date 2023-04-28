import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import TwoD_fun
import levy
import brainpy as bp
import brainpy.math as bm

fit_color = ['#009FB9', '#F18D00']
legend = ['Brownian-diffusion', 'Super-diffusion']

ticksize = 14
labelsize = 18
linewidth = 2


def plot5_c_inset(simulation=[0, 0]):
    plt.figure(figsize=(4, 3), dpi=300)
    fit_guess = np.array([[2, 0, 0], [1, 1, 3]])

    def plot_hist(label, simulation, mu, gamma):
        if simulation == 1:
            center_trace = bm.as_numpy(TwoD_fun.get_trace(mu, gamma, 100, 0.2, 1, 100))
            stepsize = np.sum(np.square(center_trace[:-1, :] - center_trace[1:, :]), axis=1)
            stepsize = stepsize[199:]
            np.save('./data/stepsize' + str(mu) + '_' + str(gamma) + '.npy', stepsize)

        stepsize = np.load('./data/stepsize' + str(mu) + '_' + str(gamma) + '.npy')
        #plt.hist(stepsize, range(20), density=True, alpha=0.5, color=fit_color[label], label=legend[label],
                 #edgecolor='w')
        ans = levy.fit_levy(stepsize, alpha=fit_guess[label, 0], beta=fit_guess[label, 1],
                            loc=fit_guess[label, 2])  # alpha beta mu sigma
        para = ans[0].get()
        dist = stats.levy_stable
        x = np.linspace(np.min(stepsize), np.max(stepsize), 100)
        plt.loglog(x, dist.pdf(x, para[0], para[1], para[2], para[3]),
                 lw=linewidth, alpha=1, color=fit_color[label],label=legend[label])

        # set upper and right axis invisible
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plot_hist(0, simulation[0], 2, 15)
    plot_hist(1, simulation[1], 1, 0.1)
    plt.xlim([0.8,20])
    plt.xlabel('Step size', fontsize=labelsize)
    plt.ylabel('Probability', fontsize=labelsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    #set y lim
    plt.ylim([0.0001, 1])
    #plt.yticks([0, 0.1, 0.2, 0.3], fontsize=ticksize)
    # add legend
    plt.legend(fontsize=ticksize/1.5, frameon=False)
    #add reference line of y=1 with red dash line
    plt.plot([0.8, 20], [1, 1], '--', color='r', linewidth=1) 

    plt.tight_layout()
    plt.savefig('./Figures/Fig5c_inset.pdf', bbox_inches='tight')
    plt.show()


plot5_c_inset([1, 1])
