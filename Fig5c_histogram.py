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


fit_color = ['#009FB9','#F18D00']
legend = ['Brownian-diffusion', 'Super-diffusion']

ticksize = 14
labelsize = 18
linewidth = 2


def plot5c(simulation=[0, 0]):
    plt.figure(figsize=(4, 3), dpi=300)
    fit_guess = np.array([[2,0,0],[1,1,3]])
    def plot_hist(label, simulation, mu, gamma):
        if simulation == 1:
            center_trace = bm.as_numpy(TwoD_fun.get_trace(mu, gamma , 100, 0.2, 1, 100))
            stepsize = np.sum(np.square(center_trace[:-1, :] - center_trace[1:, :]), axis=1)
            stepsize = stepsize[199:]
            np.save('./data/stepsize' + str(mu) + '_' + str(gamma) + '.npy', stepsize)

        stepsize = np.load('./data/stepsize' + str(mu) + '_' + str(gamma) + '.npy')
        plt.hist(stepsize, range(20),density=True, alpha = 0.5, color = fit_color[label],label = legend[label],edgecolor='w')
        ans = levy.fit_levy(stepsize, alpha = fit_guess[label,0], beta = fit_guess[label,1], loc = fit_guess[label,2])  # alpha beta mu sigma
        para = ans[0].get()
        dist = stats.levy_stable
        x = np.linspace(np.min(stepsize), np.max(stepsize), 100)
        plt.plot(x, dist.pdf(x, para[0], para[1], para[2], para[3]),
                 lw=linewidth, alpha=1,  color = fit_color[label])

        #set upper and right axis invisible
        ax = plt.gca()  
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        
    plot_hist(0, simulation[0], 2, 15)
    plot_hist(1, simulation[1], 1, 0.1)
    plt.xlabel('Step size',fontsize = labelsize)
    plt.ylabel('Probability', fontsize=labelsize)
    plt.xticks(fontsize = ticksize)
    plt.yticks([0,0.1,0.2,0.3],fontsize = ticksize)
    #put the legend on the right upper corner
    plt.legend(fontsize = ticksize,frameon=False, loc='upper right')

    plt.tight_layout()
    plt.savefig('./Figures/Fi5c.pdf', bbox_inches='tight')
    plt.show()

plot5c([0,0])