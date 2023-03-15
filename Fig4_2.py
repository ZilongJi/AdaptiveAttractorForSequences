import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import TwoD_fun
import levy

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'cm'

fit_color = ['red','blue']

def plot4_2(simulation=[0, 0]):
    plt.figure(figsize=(3, 3), dpi=300)
    fit_guess = np.array([[2,0,0],[1,1,3]])
    def plot_hist(label, simulation, mu, gamma):
        if simulation == 1:
            center_trace = TwoD_fun.get_trace(100, mu, gamma, 0.2, 1, 100)
            stepsize = np.sum(np.square(center_trace[:-1, :] - center_trace[1:, :]), axis=1)
            stepsize = stepsize[199:]
            np.save('./data/stepsize' + str(mu) + '_' + str(gamma) + '.npy', stepsize)

        stepsize = np.load('./data/stepsize' + str(mu) + '_' + str(gamma) + '.npy')
        plt.hist(stepsize, density=True, bins= 15,alpha = 0.5, color = fit_color[label])
        ans = levy.fit_levy(stepsize, alpha = fit_guess[label,0], beta = fit_guess[label,1], loc = fit_guess[label,2])  # alpha beta mu sigma
        para = ans[0].get()
        dist = stats.levy_stable
        x = np.linspace(np.min(stepsize), np.max(stepsize), 100)
        plt.plot(x, dist.pdf(x, para[0], para[1], para[2], para[3]),
                 lw=1, alpha=0.6,  color = fit_color[label])

    plot_hist(0, simulation[0], 2, 15)
    plot_hist(1, simulation[1], 1, 0.1)
    plt.savefig('./Figures/Fig4_2.png')
    plt.savefig('./Figures/Fig4_2.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

plot4_2([0,0])