from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class OLGclass():

    def __init__(self):
        """ set initial  """

        # a. set SimpleNamespace
        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        # b. set setup the class 
        self.setup()

        # c. set allecate of varnames
        self.varname_allocate()
    
    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. number of simulation periods 
        par.simT = 20 

        # b. household
        # i. CRRA coefficient
        par.sigma = 1.0   
        # ii. discount factor
        par.beta = 1/1.50

        # c. firms
        # i. capital weight
        par.alpha = 0.30
        # ii. depreciation rate
        par.delta = 0.50

        # d. government
        # i. labor income tax
        par.tau_w = 0.0
        # ii. capital income tax
        par.tau_r = 0.0  

        # e. starting points
        # i. capital stock
        par.K_lag_start = 1.0   
        # ii. government debt
        par.B_lag_start = 0.0

        # f. population 
        par.n = 1.0

        # g. technology
        par.At = 1.0

        # f. PAYG
        par.d = 0.0

    def varname_allocate(self):
        """ allocate arrays for simulation """
        
        par = self.par
        sim = self.sim

        # a. dictionary of variables and initial values
        varibale = ['C1','C2','K','Y','K_lag','w','rk','rb','r','rt','G','T','B','balanced_budget','B_lag']
        variables = {varname: np.nan*np.ones(par.simT) for varname in varibale}

        # b. set attributes for each variable in sim object
        for varname, init_value in variables.items():
            sim.__dict__[varname] = init_value

    def sim_model(self):
        """ simulate the OLG model """

        # a. set par and sim 
        par = self.par
        sim = self.sim
        
        # b. initial values
        sim.K_lag[0] = par.K_lag_start
        sim.B_lag[0] = par.B_lag_start

        # c. iterate 
        for t in range(par.simT):
            
            # i. simulate before known s
            sim_b(par,sim,t)

            if t == par.simT-1: 
                continue          

            # i. find bracket to search
            s_min, s_max = find_s_bracket(par,sim,t)

            # ii. find optimal s
            obj = lambda s: c_euler_error(s,par,sim,t=t)
            s_result = optimize.root_scalar(obj,bracket=(s_min,s_max),method='bisect')
            s = s_result.root

            # iii. simulate after known s
            sim_a(par,sim,t,s)

def find_s_bracket(par,sim,t,maxiter=1000):
    """ Find bracket for s to search in """

    # a. starting points for s
    # i. saving almost nothing
    s_min_start = 0.0 + 1e-8
    # ii. saving almost everything
    s_max_start = 1.0 - 1e-8

    # b. saving a lot is always possible 
    value_max = c_euler_error(s_max_start, par, sim, t)
    sign_max = np.sign(value_max)

    # c. find s
    it = 0
    while it < maxiter:
        
        # i. calculate midpoint and its value
        s = (s_min_start + s_max_start) / 2
        value = c_euler_error(s, par, sim, t)

        # ii. check if conditions are met
        if not np.isnan(value) and np.sign(value) * sign_max < 0:
            # o. found bracket
            s_min = s
            s_max = s_max_start
            return s_min, s_max

        # iii. update bracket
        if np.isnan(value) or np.sign(value) * sign_max > 0:
            s_min_start = s
        else:
            s_max_start = s
        
        it += 1

    raise Exception('cannot find bracket for s')

def c_euler_error(s,par,sim,t):
    """ Find the euler error """

    # a. simulate forward
    sim_a(par,sim,t,s)
    sim_b(par,sim,t+1) 

    # c. Euler equation
    LHS = sim.C1[t]**(-par.sigma)
    RHS = (1+sim.rt[t+1])*par.beta * sim.C2[t+1]**(-par.sigma)

    return LHS-RHS

def sim_b(par,sim,t):
    """ 
    Find the values of parameters before the savings are known. 
    We can find all, but the consumption of young (C1), investments (I) and capital (K). 
    """
    
    # a. if we are at t greater than the starting point, then K_lag and B_lag is just the former K and B
    if t > 0:
        sim.K_lag[t] = sim.K[t-1]
        sim.B_lag[t] = sim.B[t-1]

    # b. Cobb-Douglass production function 
    sim.Y[t] = (sim.K_lag[t])**par.alpha * (par.n*par.At)**(1-par.alpha)

    # c. factor prices
    # i. rate
    sim.rk[t] = par.alpha*sim.K_lag[t]**(par.alpha-1) * (par.n*par.At)**(1-par.alpha)
    # ii. wages 
    sim.w[t] = (1-par.alpha)*sim.K_lag[t]**(par.alpha) * (par.n*par.At)**(-par.alpha)
    
    # d. no-arbitrage and after-tax return
    # i. after-depreciation return
    sim.r[t] = sim.rk[t]-par.delta     
    # ii. same return on bonds
    sim.rb[t] = sim.r[t]               
    # iii. after-tax return
    sim.rt[t] = (1-par.tau_r)*sim.r[t]

    # e. consumption of old 
    sim.C2[t] = (1+sim.rt[t])*(sim.K_lag[t]+sim.B_lag[t]) + par.d

    # f. government tax revenues 
    sim.T[t] = par.tau_r*sim.r[t]*(sim.K_lag[t]+sim.B_lag[t]) + par.tau_w*sim.w[t]*par.n*par.At + par.d

    # g. set a balanced budget for the government 
    if sim.balanced_budget[t]:
        sim.G[t] = sim.T[t] - sim.r[t]*sim.B_lag[t]

    # h. find the government debt 
    sim.B[t] = (1+sim.r[t])*sim.B_lag[t] - sim.T[t] + sim.G[t]


def sim_a(par,sim,t,s):
    """ 
    Find the values of young consumption (C1), investments (I) and capital (K) after the savings are found. 
    """

    # a. consumption of young
    sim.C1[t] = (1-par.tau_w)*sim.w[t]*(1.0-s)*par.n*par.At - par.d

    # b. end-of-period stocks
    # i. investments 
    I = sim.Y[t] - sim.C1[t] - sim.C2[t] - sim.G[t]
    # ii. capital 
    sim.K[t] = (1-par.delta)*sim.K_lag[t] + I


def plot_K(K_lag, K_ss = None, K_lag_int = None, K_lag_old = None, K_lag_tau = None):
    """ Plot the capital """

    # a. setup for figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # b. plot 
    ax.plot(K_lag, label=r'$K_{t-1}$', color = 'red')
    # i. analytical steady state 
    if K_ss is not None:
        ax.axhline(K_ss, ls='--', color='black', label='analytical steady state')
    # ii. last simulation 
    if K_lag_old is not None:
        ax.plot(K_lag_old, label=r'$K_{t-1}$ (last simulation)', color = 'blue')
    # iii. capital when implementing labor income tax
    if K_lag_tau is not None:
        ax.plot(K_lag_tau, label=r'$K_{t-1}$ (labor income tax)', color = 'darkblue')
    # iv. initial capital 
    if K_lag_int is not None:
        ax.plot(K_lag_int, label=r'$K_{t-1}$ (initial simulation)', color = 'black')
    
    # c. add legend 
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize=10, ncol = 2)

    # d. make it look pretty and show the beauty 
    fig.tight_layout()
    plt.show()


def plot_C(C1, C1_2, C2, C2_2, title = None):
    """ Plot the consumption """
    # a. setup for figure
    fig = plt.figure(figsize = (2*6,8/1.5))
    
    # b. plot C_1 
    ax = fig.add_subplot(1,2,1)
    ax.plot(C1, label=r'$C_{1t}$ (initial)')
    ax.plot(C1_2, label=r'$C_{1t}$')
    
    # c. add legend (located under plot and two columns)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize=10, ncol = 2)

    # f. set the title
    if title is not None:
        ax.set_title(title, loc = 'left')

    # d. plot C_2
    ax = fig.add_subplot(1,2,2)
    ax.plot(C2, label=r'$C_{2t+1}$ (initial)')
    ax.plot(C2_2, label=r'$C_{2t+1}$')

    # e. add legends (located under plot and two columns)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize=10, ncol = 2)

    # g. make it look pretty and show the beauty 
    fig.tight_layout()
    plt.show()









