import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import TwoD_fun

plt.rcParams ['pdf.fonttype'] = 42
#plt.rcParams ['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'cm'

xlabel = ['Brownian motion', 'Lévy flights']

ticksize = 14
labelsize = 18
linewidth = 1.5
custom_color1 = '#F18D00'
custom_color2 = '#009FB9'


def plot_4_3(simulation = 0):
    fig = plt.figure(figsize = (9,4.5), dpi=300)
    if simulation == 1:
        Alpha = TwoD_fun.get_Alpha(11, 21, True)
        np.save('./data/Alpha.npy', Alpha)
    Alpha = np.load('./data/Alpha.npy')[0:-3,:,:]
    Alpha = np.mean(Alpha, axis=2)
    Alpha = np.concatenate((np.ones((Alpha.shape[0],Alpha.shape[0]-Alpha.shape[1]))*np.nan,Alpha),axis = 1)
    Alpha[:,7] *= np.nan

    #plot the heatmap
    plt.imshow(Alpha, origin='lower',cmap='inferno')

    #add lines
    plt.plot(np.linspace(7.5,15.5,100),17/2/np.sqrt(2)*np.sqrt(np.linspace(0,9,100))-0.5, linewidth=2 ,color=custom_color2, alpha=1)
    plt.plot(np.linspace(7.5,7.5,10),np.linspace(-0.5,16.5,10), linewidth=2, color ='black',alpha=1)

    plt.plot(np.linspace(10, 10, 10), np.linspace(-0.5, 16.5, 10), linewidth=1.5, linestyle='--', color = 'gray', alpha = 1)
    plt.plot(np.linspace(7.5, 17.5, 10), np.linspace(13.5, 13.5, 10), linewidth=1.5, linestyle='--', color='gray',alpha=1)

    #add colorbar
    axcb = plt.colorbar()
    axcb.set_label(r'Lévy exponent $\alpha$', fontsize=labelsize)
    axcb.set_ticks([1, 1.5, 2])
    axcb.set_ticklabels([1, 1.5, 2], fontsize=ticksize)

    #add markers for demonstration
    plt.scatter([16,14,9,2],[0,10,13,0], marker='o', c=custom_color1, edgecolors='k', s=40, alpha=1)

    #decorate the plot
    plt.xticks(np.array([3,8,13,18])-0.5,[-0.5,0,0.5,1],fontsize = ticksize)
    plt.yticks(np.array([0, 7, 14]) - 0.5, [0, 0.5, 1],fontsize = ticksize)
    plt.xlabel('Dist.-to-boundary $\mu$', fontsize = labelsize)
    plt.ylabel('Noise amp. $\gamma$', fontsize=labelsize)
    plt.ylim([-0.5,16.5])

    plt.tight_layout()
    plt.savefig('./Figures/Fig4b.pdf')
    

plot_4_3(0)