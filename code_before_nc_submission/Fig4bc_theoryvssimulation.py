import numpy as np
import matplotlib.pyplot as plt
import brainpy.math as bm
#set default ramndom seed for reproducibility
bm.random.seed(1)
#set backend to cpu
bm.set_platform('cpu')

# Levy exponent v.s. adaptation strength; Levy exponent v.s. noise strength

ticksize = 14
labelsize = 18
linewidth = 2

color = ['#F18D00','#009FB9']

def plot_4_3(simulation = 0):
    '''
    :param simulation: 0: load simualted data; 1: simulation
    '''
    Alpha = np.load('./data/Alpha.npy')[:,1:,:]
    #print(Alpha.shape)
    Alpha_mean = np.mean(Alpha,axis = 2)
    Alpha_std = np.std(Alpha, axis = 2)
    #print(Alpha_std)


    mu = np.linspace(0, 1, Alpha.shape[1])
    mu_theory = np.linspace(0, 1, 1000)
    mu_theory_ = np.minimum(2, 1 + 2 * mu_theory) # gamma = 1

    gamma = np.linspace(0, 1.5, Alpha.shape[0])
    gamma_theory = np.linspace(0.01, 1.5, 1000)
    gamma_theory_ = np.minimum(2, 1 + 0.5 / np.square(gamma_theory)) # mu = 0.25

    fig = plt.figure(figsize = (4,3), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(mu,Alpha_mean[9,:], marker = 'o', markersize=5, markerfacecolor='white',linestyle='--', color = color[0],linewidth = linewidth,label = 'simulated')
    plt.fill_between(mu, Alpha_mean[9,:] - Alpha_std[9,:], Alpha_mean[9,:] + Alpha_std[9,:],color= color[0], alpha=0.2)

    plt.plot(mu_theory, mu_theory_, linestyle='-', color =  color[1],linewidth = linewidth ,label = 'analytical')
    plt.ylabel(r'Lévy exponent $\alpha$', fontsize=labelsize)
    plt.xlabel(r'Dist.-to-boundary $\mu$', fontsize=labelsize)
    plt.xticks(np.array([0, 0.5, 1]), fontsize = ticksize)
    plt.yticks(np.asarray([1,1.5,2]), fontsize=ticksize)
    plt.legend(fontsize=ticksize)
    #remove right and up axiss
    ax.spines['right'].set_visible(False)   
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('./Figures/Fig4d.pdf')
    
    #create a new figure
    fig = plt.figure(figsize = (4,3), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    #plt.errorbar(gamma, Alpha_mean[:,3],Alpha_std[:,3],fmt='o', markersize=3, markerfacecolor='white',linestyle = '-',color = 'k', label = 'simulated')
    plt.plot(gamma, Alpha_mean[:,3], marker='o', markersize=5, markerfacecolor='white', linestyle='--', color=color[0],
             linewidth=linewidth,label = 'simulated')
    plt.fill_between(gamma, Alpha_mean[:,3] - Alpha_std[:,3], Alpha_mean[:,3] + Alpha_std[:,3], color=color[0],
                     alpha=0.2)
    plt.plot(gamma_theory, gamma_theory_, linestyle='-', color=color[1],label = 'analytical',linewidth = linewidth)
    plt.ylabel(r'Lévy exponent $\alpha$', fontsize=labelsize)
    plt.xlabel(r'Noise amp. $\gamma$', fontsize=labelsize)
    plt.xticks(np.array([0, 0.5, 1, 1.5]),fontsize = ticksize)
    plt.yticks(np.asarray([1,1.5,2]), fontsize = ticksize)

    plt.legend(fontsize = ticksize)
    #remove right and up axiss
    ax.spines['right'].set_visible(False)   
    ax.spines['top'].set_visible(False)   
    
    plt.tight_layout()
    plt.savefig('./Figures/Fig4e.pdf')


plot_4_3(simulation = 0)