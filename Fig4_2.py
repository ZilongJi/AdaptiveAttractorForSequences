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

plt.rcParams ['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'cm'

#fit_color = ['#7F5994','#E8EB5E']
fit_color = ['#440453','#FDE73A']
#colors = ['#7F5994', '#7D70A6', '#7685AE', '#6F97B0', '#65A8B0', '#61B9AB', '#6ECAA3', '#8FD892', '#B8E475', '#E8EB5E']

legend = ['Brownian motion', 'Lévy flights']

ticksize = 15
charsize = 18
linewidth = 1.5


def plot4_2(simulation=[0, 0]):
    plt.figure(figsize=(4, 4))
    fit_guess = np.array([[2,0,0],[1,1,3]])
    def plot_hist(label, simulation, mu, gamma):
        if simulation == 1:
            center_trace = bm.as_numpy(TwoD_fun.get_trace(mu, gamma , 100, 0.2, 1, 100))
            stepsize = np.sum(np.square(center_trace[:-1, :] - center_trace[1:, :]), axis=1)
            stepsize = stepsize[199:]
            np.save('./data/stepsize' + str(mu) + '_' + str(gamma) + '.npy', stepsize)

        stepsize = np.load('./data/stepsize' + str(mu) + '_' + str(gamma) + '.npy')
        plt.hist(stepsize, range(20),density=True, alpha = 0.7, color = fit_color[label],label = legend[label],edgecolor='w')
        ans = levy.fit_levy(stepsize, alpha = fit_guess[label,0], beta = fit_guess[label,1], loc = fit_guess[label,2])  # alpha beta mu sigma
        para = ans[0].get()
        dist = stats.levy_stable
        x = np.linspace(np.min(stepsize), np.max(stepsize), 100)
        plt.plot(x, dist.pdf(x, para[0], para[1], para[2], para[3]),
                 lw=linewidth, alpha=1,  color = fit_color[label])

    plot_hist(0, simulation[0], 2, 15)
    plot_hist(1, simulation[1], 1, 0.1)
    plt.xlabel('Step size',fontsize = charsize)
    plt.ylabel('Probability', fontsize=charsize)
    plt.xticks(fontsize = ticksize)
    plt.yticks([0,0.1,0.2,0.3],fontsize = ticksize)
    plt.legend(fontsize = ticksize,frameon=False)

    plt.savefig('./Figures/Fig4_2.png', bbox_inches='tight', dpi=300)
    plt.savefig('./Figures/Fig4_2.pdf', bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()

plot4_2([0,0])