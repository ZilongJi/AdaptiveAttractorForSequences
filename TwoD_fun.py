import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import levy
import math

bm.set_platform('cpu')

class CANN2D(bp.dyn.NeuGroup):
  def __init__(self, length, tau=1, tau_v=144, m_0=1., k=8.1, a=0.2, A=10., J0=4., sigma_u=0., sigma_m=0.,
               z_min=-bm.pi, z_max=bm.pi, name=None):
    super(CANN2D, self).__init__(size=(length, length), name=name)

    # parameters
    self.length = length
    self.tau = tau  # The synaptic time constant
    self.tau_v = tau_v
    self.m = tau/tau_v*m_0
    self.k = k  # Degree of the rescaled inhibition
    self.a = a  # Half-width of the range of excitatory connections
    self.A = A  # Magnitude of the external input
    self.J0 = J0  # maximum connection value
    self.sigma_m = sigma_m
    self.sigma_u = sigma_u

    # feature space
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.x = bm.linspace(z_min, z_max, length + 1)[0:-1] # The encoded feature values
    self.rho = length / self.z_range  # The neural density
    self.dx = self.z_range / length  # The stimulus density

    # The connections
    self.conn_mat = self.make_conn()
    self.conn_fft = bm.fft.fft2(self.conn_mat)
    self.reset_state()
    self.center = bm.Variable(bm.zeros(2))

  def reset_state(self, batch_size=None):
    # variables
    self.r = bm.Variable(bm.zeros((self.length, self.length)))
    self.u = bm.Variable(bm.zeros((self.length, self.length)))
    self.v = bm.Variable(bm.zeros((self.length, self.length)))
    self.input = bm.Variable(bm.zeros((self.length, self.length)))

  def show_conn(self):
    plt.imshow(bm.as_numpy(self.conn_mat))
    plt.colorbar()
    plt.show()

  def dist(self, d):
    v_size = bm.asarray([self.z_range, self.z_range])
    return bm.where(d > v_size / 2, v_size - d, d)

  def make_conn(self):
    x1, x2 = bm.meshgrid(self.x, self.x)
    value = bm.stack([x1.flatten(), x2.flatten()]).T
    d = self.dist(bm.abs(value[0] - value))
    d = bm.linalg.norm(d, axis=1)
    d = d.reshape((self.length, self.length))
    Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
    return Jxx

  def get_stimulus_by_pos(self, pos):
    assert bm.size(pos) == 2
    x1, x2 = bm.meshgrid(self.x, self.x)
    value = bm.stack([x1.flatten(), x2.flatten()]).T
    d = self.dist(bm.abs(bm.asarray(pos) - value))
    d = bm.linalg.norm(d, axis=1)
    d = d.reshape((self.length, self.length))
    return self.A * bm.exp(-0.25 * bm.square(d / self.a))

  def get_center(self):
    xcenter = bm.sum(self.r, axis=0)
    ycenter = bm.sum(self.r, axis=1)
    exppos = bm.exp(1j * self.x)
    self.center[0] = bm.angle(bm.sum(exppos*xcenter))
    self.center[1] = bm.angle(bm.sum(exppos*ycenter))

  def update(self, tdi):
    r1 = bm.square(self.u)
    r2 = 1.0 + self.k * bm.sum(r1)
    self.r.value = r1 / r2
    r = bm.fft.fft2(self.r)
    interaction = bm.real(bm.fft.ifft2(r * self.conn_fft))
    self.u.value = self.u + (-self.u + self.input + interaction - self.v) / self.tau * bm.get_dt() \
                   + self.sigma_u * bm.random.normal(0, 1, (self.length, self.length)) * bm.sqrt(bm.get_dt() / self.tau)
    self.v.value = self.v + (-self.v + self.m * self.u) / self.tau_v * bm.get_dt() \
                   + self.sigma_m * self.u * bm.random.normal(0, 1, (self.length, self.length)) * bm.sqrt(bm.get_dt() / self.tau_v)
    #self.v.value = self.v + (-self.v + self.m * self.r) / self.tau_v * bm.get_dt() \
                   #+ self.sigma_m * self.u * bm.random.normal(0, 1, (self.length, self.length)) * bm.sqrt(bm.get_dt() / self.tau_v)
    self.get_center()
    self.input[:] = 0.
# m = 1.13 boundary

def get_trace(mu, gamma, duration=10, a=0.2, tau=1, tau_v=20, sigma_u = 0.5, visulaize = False):
  def get_sigma_m(mu, gamma):
    m_0 = 1 - mu
    sigma_m = 2 * math.sqrt(np.pi) * m_0 * tau / tau_v * a * gamma
    print('sigma_m =', sigma_m, ', m_0 = ', m_0)
    return sigma_m, m_0

  sigma_m, m_0 = get_sigma_m(mu, gamma)

  cann = CANN2D(length=100, a=a, tau=tau, tau_v=tau_v, k=0.1, sigma_u=sigma_u, sigma_m=sigma_m, m_0=m_0)
  Iext, length = bp.inputs.section_input(
      values=[cann.get_stimulus_by_pos([0., 0.]), 0.],
      durations=[2., duration],
      return_length=True
  )

  runner = bp.DSRunner(cann,
                       inputs = ['input', Iext, 'iter'],
                       monitors = ['r', 'center'],
                       dt = 0.01,
                       numpy_mon_after_run=False)
  runner.run(length)
  center_trace = runner.mon.center

  if visulaize == True:
    '''
    bp.visualize.animate_2D(values=runner.mon.r.reshape((-1, cann.num)),
                           net_size=(cann.length, cann.length))

    '''
    plt.scatter(center_trace[:,0], center_trace[:,1], c = np.linspace(1,0,center_trace.shape[0]), s = 1)
    #plt.xlim([-np.pi,np.pi])
    #plt.ylim([-np.pi,np.pi])
    plt.show()

  # center_trace = bm.as_numpy(center_trace)
  #np.save('./data/center_trace'+str(mu)+'_'+str(gamma)+'.npy', center_trace)
  return center_trace


def get_alpha(trace,mu = 0,gamma = 0):
  # trace = np.load('./data/center_trace.npy')
  data = np.sum(np.square(trace[:-1, :] - trace[1:, :]), axis=1)
  data = data[199:]
  # data = np.random.choice(data,200)
  data = np.concatenate((data, data * -1), axis=0)
  data = data * 10e6
  # ans = levy.fit_levy(data,beta = 0, location = 0, scale = 2.5)# alpha beta mu sigma
  ans = levy.fit_levy(data, beta=0, mu=0, sigma=2.3)  # alpha beta mu sigma
  # print(ans)
  para = (2- ans[0].get()) / 1.5 + 1
  print(para[0])
  '''
  likelihood = ans[1]
  print(likelihood)
  plt.figure()
  plt.hist(data, density=True, bins='auto')
  dist = stats.levy_stable
  x = np.linspace(np.min(data),
                  np.max(data), 100)
  plt.plot(x, dist.pdf(x, para[0], para[1], para[2], para[3]),
           'r-', lw=5, alpha=0.6, label='levy_stable pdf')
  plt.savefig('./Figures/'+str(round(mu,2))+'_'+str(round(gamma,2))+'.jpg')
  plt.close()
  #plt.show()
  '''
  return para[0]

def get_Alpha(N, M, simulation = True, epoch=10):
  Alpha = np.zeros((M, N, epoch))

  for e in range(epoch):
    bm.random.seed()
    if simulation == True:
      mu_list = np.linspace(0, 1, N).astype(float)
      gamma_list = np.linspace(0, 1.5, M).astype(float)
      mu_list, gamma_list = np.meshgrid(mu_list, gamma_list)
      Trace = bp.running.jax_vectorize_map(get_trace, [mu_list.flatten(), gamma_list.flatten()], clear_buffer = True, num_parallel=N*M)
      Trace = bm.as_numpy(Trace)
      np.save('./data/Trace_'+str(e)+'.npy',Trace)
    Trace = np.load('./data/Trace_'+str(e)+'.npy')

    for i in range(M):
      for j in range(N):
        Alpha[i,j,e] = get_alpha(Trace[N*M-1-i*N-j,:,:])
  return Alpha


if __name__ == '__main__':
  print('###')