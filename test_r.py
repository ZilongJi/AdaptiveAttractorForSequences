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

ticksize = 15
charsize = 18
linewidth = 1.5


def downsample(center,num = 10):
    ans = np.zeros(np.floor(center.shape[0]/num).astype(int)-1)
    for i in range(ans.shape[0]):
        ans[i] = center[num*i]
    return ans


mu = 0.1
gamma = 15
center_trace = bm.as_numpy(TwoD_fun.get_trace(mu, gamma, 100, 0.2, 1, 100,visulaize= True))

