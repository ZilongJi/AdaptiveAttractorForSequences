import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import TwoD_fun
import brainpy.math as bm

#xlabel = ['Super-diffusion', 'Brownian-diffusion', 'Stationary', 'Traveling wave']
xlabel = ['Super-diffusion', 'Brownian-diffusion', 'Stationary']

ticksize = 14
labelsize = 18
linewidth = 2

def downsample(center, num=300):
    ans = np.zeros(np.floor(center.shape[0] / num).astype(int) - 1)
    for i in range(ans.shape[0]):
        ans[i] = center[num * i]
    return ans


def plot_6(simulation=[0, 0, 0]):
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3), sharex=False, sharey=False)

    def linetrace(mu, gamma, simulation, ax, label, sigma_u=0.5):
        if simulation == 1:
            center_trace = bm.as_numpy(TwoD_fun.get_trace(mu, gamma, 10, 0.2, 1, 1, sigma_u=sigma_u))
            np.save('./data/fig6_center_trace' + str(mu) + '_' + str(gamma) + '.npy', center_trace)

        center_trace = np.load('./data/fig6_center_trace' + str(mu) + '_' + str(gamma) + '.npy')

        x = downsample(center_trace[200:-1, 0],num = int(center_trace.shape[0]/100))
        y = downsample(center_trace[200:-1, 1],num = int(center_trace.shape[0]/100))
        # if simulation == 1:
        #     print(center_trace.shape)
        #     print(x)
        #     print(y)

        dydx = np.array((range(x.shape[0]))) / x.shape[0]  # first derivative
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(dydx.min(), dydx.max())
        lc = LineCollection(segments, cmap='Blues', norm=norm, linewidth=linewidth, alpha=1)
        lc.set_array(dydx)
        line = ax.add_collection(lc)
        ax.set_title(xlabel[label], fontsize=labelsize)

        # ax.set_xticks([0, 1])
        # ax.set_yticks([0, 1])
        # ax.set_xticklabels([0, 1], fontsize=ticksize)
        # ax.set_yticklabels([0, 1], fontsize=ticksize)

        #set ticks off
        ax.set_xticks([])
        ax.set_yticks([])

        size = 100
        sigma = 0.02
        xx = np.linspace(np.min(x)-0.1,np.max(x)+0.1,size)
        yy = np.linspace(np.min(y)-0.1,np.max(y)+0.1,size)
        X,Y = np.meshgrid(xx,yy)
        Z = np.zeros((size,size))
        for i in range(x.shape[0]):
            Z += np.exp((-(X-x[i])**2-(Y-y[i])**2)/(2*sigma**2))
        ax.contourf(X, Y, Z, alpha=1, levels=5, cmap='inferno')

    line = linetrace(0.1, 1, simulation[0], axs[0], 0)
    line = linetrace(0.5, 0.1, simulation[1], axs[1], 1)
    line = linetrace(0.9, 0.01, simulation[2], axs[2], 2)
    #line = linetrace(-0.3, 0, simulation[3], axs[3], 3, sigma_u=0.05)

    '''
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.75])
    axcb = fig.colorbar(line, cax=cbar_ax, cmap='inferno')
    axcb.set_label(r'Time $t(/\tau)$', fontsize=labelsize)
    axcb.set_ticks([0, 0.5, 1])
    axcb.set_ticklabels(['0', '50', '100'], fontsize=ticksize)
    '''

    #plt.show()
    plt.savefig('./Figures/Fig6a.pdf')


if __name__ == '__main__':
    plot_6(simulation=[0, 0, 0, 0])
