# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 19:03:23 2021

@author: Zilong
"""
# 2023
import brainpy as bp
import brainpy.math as bm

class PCNeuron(bp.dyn.NeuGroup):
    def __init__(self, size, tau, noise_strength, lambda_s, lambda_d, x_s, x_d, c, 
                 theta_s, theta_c, W_pc_pv, W_pc_pc, W_pc_sst, seed, **kwargs):
        
        super(PCNeuron, self).__init__(size, name='PC', **kwargs) 
        
        #parameters
        self.size           =   size            # number of neurons
        self.tau            =   tau             # the rate time constant for PC neuron
        self.noise_strength =   noise_strength  # noise strength add to each neuron
        self.lambda_s       =   lambda_s        # the percentage of currents leadking away from soma
        self.lambda_d       =   lambda_d        # the percentage of currents leadking away from the dendrite
        self.x_s            =   x_s             # bottom-up input to the pc
        self.x_d            =   x_d             # top-down input to the pc
        self.c              =   c               # scales the amount of current from dendritic calcium spike
        self.theta_s        =   theta_s         # the rheobase of the PC
        self.theta_c        =   theta_c         # he threshold for the minimal input needed to produce a Ca2+ - spike
        
        self.W_pc_pv        =   W_pc_pv         # the synaptic connection from PVs to PCs (preferably inhibit the perisomatic and the basal dendrites of PCs)
        self.W_pc_pc        =   W_pc_pc         # the synaptic connection from PCs to PCs (preferably excite the apical dendrites of PCs)
        self.W_pc_sst       =   W_pc_sst        # the synaptic connection from SSTs to PCs (preferably inhibit the apical dendrites of PCs)
        
        #variables
        #A           =   bm.random.randn(self.num)
        #A[A<0]      =   0
        #self.r_pc   =   bm.Variable(A) 
        self.r_pc   =   bm.Variable(bm.zeros(self.num)) # firing rate of the somatic compartment of pcs
        self.I_S    =   bm.Variable(bm.zeros(self.num)) # total synaptic input to soma
        self.I_D    =   bm.Variable(bm.zeros(self.num)) # total synaptic input to dendrite
        self.I_0    =   bm.Variable(bm.zeros(self.num)) # current at the dendrite for generating dendritic spike
        self.I_D0   =   bm.Variable(bm.zeros(self.num))# total synaptically  generated input in the dendrites
        self.rng    =   bm.random.RandomState(seed)
        
        #other classes
        self.PV     =   None
        self.SST    =   None
        
        self.integral = bp.odeint(self.derivative, method='exp_auto')
        
        
    def derivative(self, r_pc, t, I_total):    
        I_total_thres = I_total - self.theta_s
        I_total_thres = bm.where(I_total_thres < 0, 0, I_total_thres)
        # I_total_thres[I_total_thres < 0] = 0
        dr_pc = 1. / self.tau * (-r_pc + I_total_thres)
        return dr_pc
    
    def update(self, _t, _dt):
        #1, calculate the somatic inputs:
        I_S     =   self.x_s + bm.dot(self.W_pc_pv,self.PV.r_pv)
        
        #2, calculate the dendritic inputs:
        I_D     =   self.x_d + bm.dot(self.W_pc_pc,self.r_pc) + bm.dot(self.W_pc_sst,self.SST.r_sst)
        
        #3, calculate the current generated by dendritic calcuim spike:
        #total input on the dendrites
        I_0     =   self.lambda_s*I_S+(1-self.lambda_d)*I_D 
        #implement the heaviside function. Same effect as np.heaviside, but for
        #using numba to speed up, we can't use np.heavisde here. We will update 
        #brainpy to support this in the future
        I_0_thres = I_0-self.theta_c
        I_0_thres = bm.where(I_0_thres<=0, 0, 1)
        # I_0_thres[I_0_thres<=0] = 0
        # I_0_thres[I_0_thres>0] = 1 # dendritic calcium event
        #dendritic spike current
        I_D0 = self.c*I_0_thres
        
        #4, calculate the total input coming from the dendrites, should be non-negative
        d_current = I_D + I_D0
        d_current = bm.where(d_current < 0, 0, d_current)
        # d_current[d_current<0] = 0

        #5, calculate the total input to the soma from generating firing rate
        
        I_total = (1-self.lambda_s)*I_S + self.lambda_d*d_current \
            + self.noise_strength*self.rng.randn(self.num)
        
        self.I_S.value  = I_S
        self.I_D[:]  = I_D
        self.I_D0[:] = I_D0
        self.I_0.value  = I_0
        
        self.r_pc.value = self.integral(self.r_pc, _t, I_total, _dt)
        
class PVNeuron(bp.dyn.NeuGroup):
    
    def __init__(self, size, tau, noise_strength, x_i, W_pv_pc, W_pv_pv, W_pv_sst, W_pv_vip, seed, **kwargs):    
        
        super(PVNeuron, self).__init__(size, name='PV', **kwargs)
        
        #parameters
        self.size           =   size              # number of neurons
        self.tau            =   tau               # the GABAa time constant for PVs
        self.noise_strength =   noise_strength    # noise strength add to each neuron
        self.x_i            =   x_i               # external input to PVs
        self.W_pv_pc        =   W_pv_pc           # the synaptic connection from PCs to PVs
        self.W_pv_pv        =   W_pv_pv           # the synaptic connection from PVs to PVs
        self.W_pv_sst       =   W_pv_sst          # the synaptic connection from SSTs to PVs
        self.W_pv_vip       =   W_pv_vip          # the synaptic connection from VIPs to PVs
        
        #variables
        self.r_pv   =   bm.Variable(bm.zeros(self.num))
        self.rng    =   bm.random.RandomState(seed)
        
        #other classes
        self.PC     =   None
        self.SST    =   None
        self.VIP    =   None
        
        self.integral = bp.odeint(lambda r_pv, t,  I: 1./self.tau*(-r_pv + I),
                                  method='exp_auto')
    
    '''
    def derivative(self, r_pv, t, I_total):
        dr_pv = 1./self.tau*(-r_pv + self.x_i + I_total)
        return dr_pv
    '''

    
    def update(self, _t, _dt):
        
        #calculate input from different cell type
        pc_input    =   bm.dot(self.W_pv_pc, self.PC.r_pc)
        pv_input    =   bm.dot(self.W_pv_pv, self.r_pv)
        sst_input   =   bm.dot(self.W_pv_sst, self.SST.r_sst)
        vip_input   =   bm.dot(self.W_pv_vip, self.VIP.r_vip)
        
        I_total     =   self.x_i + pc_input + pv_input + sst_input + vip_input \
                        + self.noise_strength*self.rng.randn(self.num)
        
        r_pv        =   self.integral(self.r_pv, _t, I_total, _dt)
        
        r_pv = bm.where(r_pv < 0, 0, r_pv)
        # r_pv[r_pv<0] = 0

        self.r_pv.value = r_pv
        
class SSTNeuron(bp.dyn.NeuGroup):
    
    def __init__(self, size, tau, noise_strength, x_i, W_sst_pc, W_sst_pv, W_sst_sst, 
                 W_sst_vip, seed, **kwargs):

        super(SSTNeuron, self).__init__(size, name='SST', **kwargs)
        
        #parameters
        self.size           =   size              # number of neurons
        self.tau            =   tau                 # the GABAa time constant for SSTs
        self.noise_strength =   noise_strength      # noise strength add to each neuron
        self.x_i            =   x_i                 # external input to SSTs
        self.W_sst_pc       =   W_sst_pc            # the synaptic connection from PCs to SSTs
        self.W_sst_pv       =   W_sst_pv            # the synaptic connection from PVs to SSTs
        self.W_sst_sst      =   W_sst_sst           # the synaptic connection from SSTs to SSTs
        self.W_sst_vip      =   W_sst_vip           # the synaptic connection from VIPs to SSTs 

        #variables
        self.r_sst  =   bm.Variable(bm.zeros(self.num))
        self.rng    =   bm.random.RandomState(seed)
        
        #other classes
        self.PC     =   None
        self.PV     =   None
        self.VIP    =   None
            
        self.integral = bp.odeint(lambda r_sst, t,  I: 1./self.tau*(-r_sst + I),
                                  method='exp_auto')
    
    '''
    def derivative(self.r_sst, t, I_total):
        dr_sst = 1./self.tau*(-r_sst + self.x_i + I_total)
        return dr_sst
    '''
    
    def update(self, _t, _dt):
        
        #calculate input from different cell type
        pc_input    =   bm.dot(self.W_sst_pc, self.PC.r_pc)
        pv_input    =   bm.dot(self.W_sst_pv, self.PV.r_pv)
        sst_input   =   bm.dot(self.W_sst_sst, self.r_sst)
        vip_input   =   bm.dot(self.W_sst_vip, self.VIP.r_vip)        
        
        I_total     =   self.x_i + pc_input + pv_input + sst_input + vip_input \
                        + self.noise_strength*self.rng.randn(self.num)
        
        r_sst       =   self.integral(self.r_sst, _t, I_total, _dt)
        
        r_sst = bm.where(r_sst < 0, 0, r_sst)

        self.r_sst.value = r_sst
        
class VIPNeuron(bp.dyn.NeuGroup):
    
    def __init__(self, size, tau, noise_strength, x_i, W_vip_pc, W_vip_pv, W_vip_sst, W_vip_vip, seed, **kwargs):

        super(VIPNeuron, self).__init__(size, name='VIP', **kwargs)
        
        #parameters
        self.size           =   size                # number of neurons
        self.tau            =   tau                 # the GABAa time constant for VIPs
        self.noise_strength =   noise_strength      # noise strength add to each neuron
        self.x_i            =   x_i                 # external input to VIPs
        self.W_vip_pc       =   W_vip_pc            # the synaptic connection from PCs to VIPs
        self.W_vip_pv       =   W_vip_pv            # the synaptic connection from PVs to VIPs
        self.W_vip_sst      =   W_vip_sst           # the synaptic connection from SSTs to VIPs
        self.W_vip_vip      =   W_vip_vip           # the synaptic connection from VIPs to VIPs 
        
        #variables
        self.r_vip  =   bm.Variable(bm.zeros(self.num))
        self.rng    =   bm.random.RandomState(seed)
        
        #other classes
        self.PC     =   None
        self.PV     =   None
        self.SST    =   None
        
        self.integral = bp.odeint(lambda r_vip, t,  I: 1./self.tau*(-r_vip + I),
                                  method='exp_auto')


        '''
        def derivative(self.r_vip, t, I_total):
            dr_vip = 1./self.tau*(-r_vip + self.x_i + I_total)
            return dr_vip
        '''
    
    def update(self, _t, _dt):
        
        #calculate input from different cell type
        pc_input    =   bm.dot(self.W_vip_pc, self.PC.r_pc)
        pv_input    =   bm.dot(self.W_vip_pv, self.PV.r_pv)
        sst_input   =   bm.dot(self.W_vip_sst, self.SST.r_sst)
        vip_input   =   bm.dot(self.W_vip_vip, self.r_vip)   
        
        I_total     =    self.x_i + pc_input + pv_input + sst_input + vip_input \
                        + self.noise_strength*self.rng.randn(self.num)
        
        r_vip = self.integral(self.r_vip, _t, I_total, _dt)
        
        r_vip = bm.where(r_vip<0, 0, r_vip)

        self.r_vip.value = r_vip
