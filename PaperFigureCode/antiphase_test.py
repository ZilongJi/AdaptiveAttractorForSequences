import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cann import CANN1D

from matplotlib.animation import FuncAnimation
#set backend to cpu
bm.set_platform('cpu') 
replay_dur = 10000      #simulate 200 ms, roughly the length of a SWR
init_dur = 50 #initial 10 ms to let the network settle

def get_results(mbar, sigma_m=0.1, sigma_u=0.02, oscillation_strength=0.5, oscilation_period=400):
    #initialize the network
    replay_cann_1 = CANN1D(tau=1., tau_v=48., num=128, mbar=mbar, A=0.01, sigma_m=sigma_m, sigma_u=sigma_u)
    replay_cann_1.reset_state()
    Iext, length = bp.inputs.section_input(
        values=[replay_cann_1.get_stimulus_by_pos(-3/4*np.pi), 0.], #initial position of the bump
        durations=[init_dur, replay_dur],
        return_length=True
    )
    noise = 0.00* bm.random.randn(len(Iext), replay_cann_1.num) #add independent noise to the input with level 0.02
    
    x_time, x_dim = Iext.shape
    x = np.arange(x_time)
    
    #geterante the oscillatory input, the period is oscilation_period ms, the strength is oscilaltion_strength
    #the shape is the same as Iext, using sin function
    oscillatory_input = oscillation_strength * np.sin(2*np.pi*x/oscilation_period)
    #subtratc the minimum value to make sure the oscillatory_input is always positive
    #oscillatory_input = oscillatory_input - np.min(oscillatory_input)
    #copy each value in oscillatory_input into dimension x_dim, make oscillatory_input the same shape as Iext
    oscillatory_input = np.tile(oscillatory_input, (x_dim, 1)).T
    
    
    Iext = Iext + noise + oscillatory_input
    Iext = bm.as_numpy(Iext)

    #run the network
    replay_runner_1 = bp.DSRunner(replay_cann_1,
                        inputs=('input', Iext, 'iter'),
                        monitors=['center', 'r'],
                        progress_bar=False)

    replay_runner_1.run(length)

    replay_start = int(init_dur/replay_runner_1.dt)

    bump_center = replay_runner_1.mon.center
    #squeeze the array
    bump_center = np.squeeze(bump_center)[replay_start:]

    fr = bm.as_numpy(replay_runner_1.mon.r)
    fr = fr[replay_start:, :]

    #get the mean fr for each time step]
    # fr_thres = np.max(fr)
    # fr = np.where(fr<fr_thres/10, 0, fr)
    ave_fr = np.sum(fr, axis=1)
    # ave_fr = np.mean(fr, axis=1)

    #caulcte the stepsize of the bump_center
    
    stepsize = np.diff(bump_center)
    #remove the effect of periodic boundary, if the stepsize is larger than pi, then minus 2pi, if the stepsize is smaller than -pi, then add 2pi
    stepsize = np.where(stepsize>np.pi, stepsize-2*np.pi, stepsize)
    stepsize = np.where(stepsize<-np.pi, stepsize+2*np.pi, stepsize)
    
    stepsize = np.abs(stepsize)
    #add 0 at the beginning of the array
    stepsize = np.insert(stepsize, 0, 0, axis=0)
    
    return stepsize, ave_fr, bump_center, oscillatory_input, fr

N = 1

all_stepsize = {}
all_fr = {}
all_bump_center = {}

#!!! @Tianhao: change the mbar value to add adaptation to the network 
stepsize, mean_fr, bump_center, oscillatory_input, fr = get_results(mbar=0.99, sigma_m=0.001, sigma_u=0.00,  oscillation_strength=0.01, oscilation_period=400)
#change stepsize from m to cm
stepsize = stepsize * 100
print(mean_fr)
#time all_fr_list by 1000 to get Hz 
mean_fr = mean_fr * 1000
max_fr = np.max(fr)

plt.plot(stepsize[:5000])

fig, ax = plt.subplots()
n_step = 10
data1 = fr[::n_step, :] 
print(data1.shape)
T = data1.shape[0]
N = data1.shape[1]
# 创建初始空白线条
line1, = ax.plot([], [], label='Line 1')
# 设置轴范围
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-max_fr/5, max_fr*1.2)
x = np.linspace(-np.pi, np.pi, N+1)
x = x[:-1]
# 更新线条的函数
def update(frame):
    y = data1[frame].flatten()
    line1.set_data(x, y)
    return line1

# 创建动画
ani = FuncAnimation(fig, update, frames=T, interval=20, blit=False)
# ani_filename = directory + 'test_Population_activities.gif'
# ani.save(ani_filename, writer='Pillow', fps=30)


plt.show()