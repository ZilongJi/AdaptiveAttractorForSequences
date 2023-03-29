import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import TwoD_fun

plt.rcParams ['pdf.fonttype'] = 42
plt.rcParams ['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'cm'

# Levy exponent v.s. adaptation strength; Levy exponent v.s. noise strength


ticksize = 15
charsize = 18
linewidth = 1.5

color = ['#FDE73A','#440453']
#colors = ['#440453', '#482976', '#3E4A88', '#30688D', '#24828E', '#1B9E8A', '#32B67B', '#6CCC5F', '#B4DD3D', '#FDE73A']

def plot_4_3(simulation = 0):
    fig = plt.figure(figsize = (9,4))

    Alpha = np.load('./data/Alpha.npy')[:,1:,:]
    print(Alpha.shape)
    Alpha_mean = np.mean(Alpha,axis = 2)
    Alpha_std = np.std(Alpha, axis = 2)
    print(Alpha_std)


    mu = np.linspace(0, 1, Alpha.shape[1])
    mu_theory = np.linspace(0, 1, 1000)
    mu_theory_ = np.minimum(2, 1 + 2 * mu_theory) # gamma = 1

    gamma = np.linspace(0, 1.5, Alpha.shape[0])
    gamma_theory = np.linspace(0.01, 1.5, 1000)
    gamma_theory_ = np.minimum(2, 1 + 0.5 / np.square(gamma_theory)) # mu = 0.25

    ax = plt.subplot(1,2,1)
    #plt.errorbar(mu,Alpha_mean[9,:],Alpha_std[9,:], fmt='o',markersize=3, markerfacecolor='white',linestyle = '-', color = 'k')
    plt.plot(mu,Alpha_mean[9,:], marker = 'o', markersize=5, markerfacecolor='white',linestyle='--', color = color[0],linewidth = linewidth)
    plt.fill_between(mu, Alpha_mean[9,:] - Alpha_std[9,:], Alpha_mean[9,:] + Alpha_std[9,:],color= color[0], alpha=0.2)

    plt.plot(mu_theory, mu_theory_, linestyle='-', color =  color[1],linewidth = linewidth)
    plt.ylabel(r'Lévy exponent $\alpha$', fontsize=charsize)
    plt.xlabel(r'Distance-to-boudary $\mu$', fontsize=charsize)
    plt.xticks(np.array([0, 0.5, 1]),fontsize = ticksize)
    plt.yticks(fontsize=ticksize)

    plt.subplot(1, 2, 2,sharey = ax)
    #plt.errorbar(gamma, Alpha_mean[:,3],Alpha_std[:,3],fmt='o', markersize=3, markerfacecolor='white',linestyle = '-',color = 'k', label = 'simulation')
    plt.plot(gamma, Alpha_mean[:,3], marker='o', markersize=5, markerfacecolor='white', linestyle='--', color=color[0],
             linewidth=linewidth,label = 'simulation')
    plt.fill_between(gamma, Alpha_mean[:,3] - Alpha_std[:,3], Alpha_mean[:,3] + Alpha_std[:,3], color=color[0],
                     alpha=0.2)
    plt.plot(gamma_theory, gamma_theory_, linestyle='-', color=color[1],label = 'theory',linewidth = linewidth)
    plt.xlabel(r'Noise-to-strength $\gamma$', fontsize=charsize)
    plt.xticks(np.array([0, 0.5, 1, 1.5]),fontsize = ticksize)
    plt.yticks(fontsize = ticksize)

    plt.legend(fontsize = ticksize)

    plt.savefig('./Figures/Fig4_4.png', bbox_inches='tight',dpi = 300)
    plt.savefig('./Figures/Fig4_4.pdf', bbox_inches='tight',dpi = 300)
    plt.tight_layout()
    plt.show()

plot_4_3(0)