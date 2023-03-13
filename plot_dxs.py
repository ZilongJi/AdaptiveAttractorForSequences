import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_4_1():
    sigma_m = np.linspace(0,1,11)[3]
    m_0 = np.linspace(0,1,11)[4]
    center_trace = np.load('./data/center_trace'+str(sigma_m)+str(m_0)+'.npy')
    plt.scatter(center_trace[:, 0], center_trace[:, 1], c=np.linspace(1, 0, center_trace.shape[0]), s=1)
    plt.show()

def plot_4_3():
    Alpha = np.load('./data/Alpha.npy')
    Alpha = Alpha.T
    mu = np.linspace(0,2,10)
    gamma = np.linspace(0,1,10)
    Mu,Gamma = np.meshgrid(mu,gamma)

    fig = plt.figure()
    plt.imshow(Alpha, origin = 'lower')
    plt.colorbar()
    plt.show()
    # ax = plt.axes(projection = '3d')
    # surf = ax.plot_surface(Mu,Gamma,Alpha,cmap = plt.cm.coolwarm)
    # ax.set_xlabel('Mu')
    # ax.set_ylabel('Gamma')
    # #ax.view_init(90,-90)
    # fig.colorbar(surf)
    # #ax.colorbar()
    # plt.show()
    # plt.plot(gamma,Alpha[0,:])
    # plt.show()

plot_phase()