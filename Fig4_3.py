import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import TwoD_fun

plt.rcParams ['pdf.fonttype'] = 42
plt.rcParams ['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'cm'

xlabel = ['Brownian motion', 'Lévy flights']



def plot_4_3(simulation = 0):
    fig = plt.figure(figsize = (3,3),dpi = 300)
    if simulation == 1:
        Alpha = TwoD_fun.get_Alpha(11, 21, True)
        np.save('./data/Alpha.npy', Alpha)
    Alpha = np.load('./data/Alpha.npy')[0:-3,:]
    Alpha = np.concatenate((np.ones((Alpha.shape[0],Alpha.shape[0]-Alpha.shape[1])),Alpha),axis = 1)

    plt.imshow(Alpha, origin='lower')
    plt.plot(np.linspace(7.5,15.5,100),17/2/np.sqrt(2)*np.sqrt(np.linspace(0,9,100))-0.5, linewidth = 1.5)
    plt.plot(np.linspace(7.5,7.5,10),np.linspace(-0.5,16.5,10), linewidth = 1.5)

    plt.xticks(np.array([3,8,13,18])-0.5,[-0.5,0,0.5,1])
    plt.yticks(np.array([0, 7, 14]) - 0.5, [0, 0.5, 1])
    plt.xlabel('Distance-to-boudary $\mu$', fontsize = 10)
    plt.ylabel('Noise-to-strength $\gamma$', fontsize=10)
    plt.ylim([-0.5,16.5])
    #plt.grid(None)
    #plt.rcParams["axes.grid"] = False
    axcb = plt.colorbar(shrink = 0.6)
    axcb.set_label(r'Levy exponent $\alpha$', fontsize=10)
    axcb.set_ticks([1, 1.5, 2])
    plt.plot()
    '''
    mu = np.linspace(0, 1, Alpha.shape[0])
    gamma = np.linspace(0, 1, Alpha.shape[1])
    Mu, Gamma = np.meshgrid(mu, gamma)
    ax = plt.axes(projection = '3d')
    surf = ax.plot_surface(Mu,Gamma,Alpha,cmap = plt.cm.coolwarm)
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\gamma$')
    ax.grid(False)
    ax.view_init(90,-90)
    fig.colorbar(surf)
    plt.show()
    plt.plot(gamma,Alpha[0,:])
    plt.show()
    '''
    plt.savefig('./Figures/Fig4_3.png', bbox_inches='tight')
    plt.savefig('./Figures/Fig4_3.pdf', bbox_inches='tight')
    plt.tight_layout()
    plt.show()

plot_4_3(0)