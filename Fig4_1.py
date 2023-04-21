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

xlabel = ['Traveling wave', 'Super-diffusion','Brownian motion','Stationary']

ticksize = 15
charsize = 18
linewidth = 1.5


def downsample(center,num = 10):
    ans = np.zeros(np.floor(center.shape[0]/num).astype(int)-1)
    for i in range(ans.shape[0]):
        ans[i] = center[num*i]
    return ans

def plot_4_1(simulation = [0, 0 ,0, 0]):
    fig, axs = plt.subplots(2, 2, figsize=(6, 6), sharex = True, sharey = True)
    def linetrace(mu, gamma,  simulation, ax, label, sigma_u=0.5):
        if simulation == 1:
            center_trace = bm.as_numpy(TwoD_fun.get_trace(mu, gamma, 200, 0.2, 1, 1,sigma_u = sigma_u))
            np.save('./data/center_trace' + str(mu) + '_' + str(gamma) + '.npy', center_trace)

        center_trace = np.load('./data/center_trace' + str(mu) + '_' + str(gamma) + '.npy')
        if label == 2:
            x = downsample(center_trace[200:-1,0]*-3 + 0.15)
            y = downsample(center_trace[200:-1,1]*-3 + 0.05)
        else:
            x = downsample(center_trace[200:-1,0]*-1)
            y = downsample(center_trace[200:-1,1]*-1 + 0.05)
        dydx = np.array((range(x.shape[0]))) / x.shape[0] # first derivative

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(dydx.min(), dydx.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=linewidth, alpha=0.8)
        lc.set_array(dydx)
        line = ax.add_collection(lc)
        ax.set_title(xlabel[label],fontsize=charsize)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([0, 1],fontsize=ticksize)
        ax.set_yticklabels([0, 1],fontsize=ticksize)
        # if label == 0:
        #     ax.set_xlim(-10, 10)
        #     ax.set_ylim(-10, 10)
        if label == 1:
            ax.set_xlim(x.min()-0.2, x.max()+0.2)
            ax.set_ylim(y.min()-0.2, y.max()+0.2)
        if label == 2:
            x2 = downsample(center_trace[200:-1, 0] * -10)
            y2 = downsample(center_trace[200:-1, 1] * -10)

            ax2 = fig.add_axes([0.24, 0.23, 0.2, 0.2])
            dydx = np.array((range(x2.shape[0]))) / x2.shape[0]  # first derivative

            points = np.array([x2, y2]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(dydx.min(), dydx.max())
            lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=linewidth, alpha=0.8)
            lc.set_array(dydx)
            ax2.add_collection(lc)
            #plt.plot(np.ones(2),np.ones(2), linewidth=0)

            plt.xticks([-1,-0.4],[-0.2,0],fontsize = ticksize)
            plt.yticks([], [], fontsize=ticksize)
            #plt.yticks([0.7, 1.2], [0.3, 0.4],fontsize = ticksize)

            plt.xlim([-1.1, -0.3])
            plt.ylim([0.6, 1.4])
        if label == 3:
            x2 = downsample(center_trace[200:-1, 0] * -10)
            y2 = downsample(center_trace[200:-1, 1] * -10)

            ax2 = fig.add_axes([0.65, 0.23, 0.2, 0.2])
            dydx = np.array((range(x2.shape[0]))) / x2.shape[0]  # first derivative

            points = np.array([x2, y2]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(dydx.min(), dydx.max())
            lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=linewidth, alpha=0.8)
            lc.set_array(dydx)
            ax2.add_collection(lc)
            #plt.plot(np.ones(2),np.ones(2), linewidth=0)

            plt.xticks([-0.2,0.2],[-0.01,0],fontsize = ticksize)
            plt.yticks([0.7, 1.2], [0.3, 0.4],fontsize = ticksize)

            plt.xlim([-0.3, 0.3])
            plt.ylim([-0.25, 0.35])


    line = linetrace(-0.2, 0, simulation[0], axs[0,0], 0, sigma_u = 0)
    line = linetrace(0.1, 15, simulation[1], axs[0,1], 1)
    line = linetrace(0.8, 0.1, simulation[2], axs[1,0], 2)
    line = linetrace(0.9, 0.01, simulation[3], axs[1, 1], 3)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.78])
    axcb = fig.colorbar(line,  cax=cbar_ax, cmap='viridis')
    axcb.set_label(r'Time $t(/\tau)$', fontsize=charsize)
    axcb.set_ticks([0, 0.5, 1])
    axcb.set_ticklabels(['0', '50', '100'],fontsize = ticksize)

    plt.savefig('./Figures/Fig4_1.png',bbox_inches = 'tight', dpi=300)
    plt.savefig('./Figures/Fig4_1.pdf',bbox_inches = 'tight', dpi=300)
    plt.show()


plot_4_1(simulation = [0, 0, 0, 0])
