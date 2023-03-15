import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import TwoD_fun
import brainpy as bp
import brainpy.math as bm

plt.rcParams ['pdf.fonttype'] = 42
plt.rcParams ['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'cm'

xlabel = ['Brownian motion', 'Lévy flights']


def plot_4_1(simulation = [0 ,0]):
    fig, axs = plt.subplots(1, 2, figsize=(5, 2), dpi=300, sharex = True, sharey = True)
    def linetrace(mu, gamma, simulation, ax, label):
        if simulation == 1:
            center_trace = bm.as_numpy(TwoD_fun.get_trace(mu, gamma, 1000, 0.2, 1, 100))
            np.save('./data/center_trace' + str(mu) + '_' + str(gamma) + '.npy', center_trace)

        center_trace = np.load('./data/center_trace' + str(mu) + '_' + str(gamma) + '.npy')
        x = center_trace[200:-1,0]
        y = center_trace[200:-1,1]
        dydx = np.array((range(x.shape[0]))) / x.shape[0] # first derivative

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(dydx.min(), dydx.max())
        lc = LineCollection(segments, cmap='winter', norm=norm, linewidth=1, alpha=0.8)
        lc.set_array(dydx)
        line = ax.add_collection(lc)
        ax.set_xlabel(xlabel[label])
        if label == 1:
            ax.set_xlim(x.min()-0.1, x.max()+0.1)
            ax.set_ylim(y.min()-0.1, y.max()+0.1)
            ax.set_xticks([])
            ax.set_yticks([])
        if label == 0:
            x = center_trace[200:-1, 0]*15-1.4
            y = center_trace[200:-1, 1]*15-0.8
            dydx = np.array((range(x.shape[0]))) / x.shape[0]  # first derivative

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(dydx.min(), dydx.max())
            lc = LineCollection(segments, cmap='winter', norm=norm, linewidth=1, alpha=0.8)
            lc.set_array(dydx)
            ax.add_collection(lc)


    line = linetrace(0.8, 0.1, simulation[1], axs[1], 0)
    line = linetrace(0.1, 15, simulation[0], axs[0], 1)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.78])
    axcb = fig.colorbar(line,  cax=cbar_ax)
    axcb.set_label(r'Time $t(/\tau)$', fontsize=15)
    axcb.set_ticks([0, 0.5, 1])
    axcb.set_ticklabels(['0', '50', '100'])

    plt.savefig('./Figures/Fig4_1.png',bbox_inches = 'tight')
    plt.savefig('./Figures/Fig4_1.pdf',bbox_inches = 'tight')
    plt.show()


plot_4_1(simulation = [0, 0])
