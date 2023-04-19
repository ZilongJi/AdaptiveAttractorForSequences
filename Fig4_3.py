import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import TwoD_fun

plt.rcParams ['pdf.fonttype'] = 42
plt.rcParams ['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'cm'

xlabel = ['Brownian motion', 'Lévy flights']

ticksize = 15
charsize = 18
linewidth = 1.5

def plot_4_3(simulation = 0):
    fig = plt.figure(figsize = (6,4))
    if simulation == 1:
        Alpha = TwoD_fun.get_Alpha(11, 21, True)
        np.save('./data/Alpha.npy', Alpha)
    Alpha = np.load('./data/Alpha.npy')[0:-3,:,:]
    Alpha = np.mean(Alpha, axis=2)
    Alpha = np.concatenate((np.ones((Alpha.shape[0],Alpha.shape[0]-Alpha.shape[1]))*np.nan,Alpha),axis = 1)
    Alpha[:,7] *= np.nan

    cmap = matplotlib.cm.viridis
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='white')

    plt.imshow(Alpha, origin='lower',cmap = cmap)
    print(Alpha.shape)
    plt.plot(np.linspace(7.5,15.5,100),17/2/np.sqrt(2)*np.sqrt(np.linspace(0,9,100))-0.5, linewidth = 2 ,color = 'black',alpha = 0.5)
    plt.plot(np.linspace(7.5,7.5,10),np.linspace(-0.5,16.5,10), linewidth = 2 , color = 'black',alpha = 0.5)

    plt.plot(np.linspace(10, 10, 10), np.linspace(-0.5, 16.5, 10), linewidth=1.5,linestyle='--',color = 'black',alpha = 0.8)
    plt.plot(np.linspace(7.5, 17.5, 10), np.linspace(13.5, 13.5, 10), linewidth=1.5, linestyle='--', color='black',
             alpha=0.5)



    plt.xticks(np.array([3,8,13,18])-0.5,[-0.5,0,0.5,1],fontsize = ticksize)
    plt.yticks(np.array([0, 7, 14]) - 0.5, [0, 0.5, 1],fontsize = ticksize)
    plt.xlabel('normalized dist-to-boundary $\mu$', fontsize = charsize)
    plt.ylabel('normalized noise amp $\gamma$', fontsize=charsize)
    plt.ylim([-0.5,16.5])
    #plt.grid(None)
    #plt.rcParams["axes.grid"] = False
    axcb = plt.colorbar()
    axcb.set_label(r'Lévy exponent $\alpha$', fontsize=charsize)
    axcb.set_ticks([1, 1.5, 2])
    axcb.set_ticklabels([1, 1.5, 2], fontsize=ticksize)

    plt.scatter([15,9,2],[2,13,2],marker = 'o', c = 'r',edgecolors= 'grey',s=20,alpha = 0.5)
    '''
    mu = np.linspace(-0.8, 1, Alpha.shape[1])
    gamma = np.linspace(0, 1.5, Alpha.shape[0])
    Mu, Gamma = np.meshgrid(mu, gamma)
    plt.contourf(Mu,Gamma,Alpha, alpha = 1,levels = 100)
    plt.plot(np.linspace(0,0.8, 100),1.5*np.sqrt(np.linspace(0,1, 100)), linewidth=1.5)
    plt.plot(np.linspace(0,0, 10), np.linspace(0, 1.5, 10), linewidth=1.5)
    plt.ylim([0,1.5])
    plt.xticks(np.array([-0.5,0,0.5, 1]))
    plt.yticks(np.array([0,0.5,1,1.5]))
    plt.xlabel('Distance-to-boudary $\mu$', fontsize=10)
    plt.ylabel('Noise-to-strength $\gamma$', fontsize=10)
    axcb = plt.colorbar()
    axcb.set_label(r'Levy exponent $\alpha$', fontsize=10)
    axcb.set_ticks([1, 1.5, 2])
    '''

    plt.savefig('./Figures/Fig4_3.png', bbox_inches='tight',dpi = 300)
    plt.savefig('./Figures/Fig4_3.pdf', bbox_inches='tight',dpi = 300)
    plt.tight_layout()
    plt.show()

plot_4_3(0)