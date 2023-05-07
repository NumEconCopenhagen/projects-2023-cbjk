from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd

class OLGclass():

    def __init__(self):
        """ set initial  """

        # a. set SimpleNamespace
        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        # b. set setup the class 
        self.setup()

        # c. set allecate of varnames
        self.varname()
    
    def setup(self):
        """ baseline parameters """

        # a. set par
        par = self.par

        # b. number of simulation periods 
        par.simT = 20 

        # c. household
        # i. CRRA coefficient
        par.sigma = 1.0  
        # ii. discount factor
        par.beta = 1/1.50

        # d. firms
        # i. capital weight
        par.alpha = 0.30
        # ii. depreciation rate
        par.delta = 0.50

        # e. government
        # i. labor income tax
        par.tau_w = 0.0
        # ii. capital income tax
        par.tau_r = 0.0  

        # f. starting points
        # i. capital stock
        par.K_lag_start = 0.01   
        # ii. government debt
        par.B_lag_start = 0.0

        # g. population 
        par.n = 1.0

        # h. technology
        par.At = 1.0

        # i. PAYG
        par.d = 0.0

    def varname(self):
        """ allocate arrays for simulation """
        
        # a. set par and sim 
        par = self.par
        sim = self.sim

        # b. dictionary of variables and initial values
        varibale = ['C1','C2','C2_1','K','Y','K_lag','w','rk','rb','r','rt','G','T','B','balanced_budget','B_lag']
        variables = {varname: np.nan*np.ones(par.simT) for varname in varibale}

        # c. set attributes for each variable in sim object
        for varname, init_value in variables.items():
            sim.__dict__[varname] = init_value

    def sim_model(self):
        """ simulate the OLG model """

        # a. set par and sim 
        par = self.par
        sim = self.sim
        
        # b. initial values for capital and goverment debt 
        sim.K_lag[0] = par.K_lag_start
        sim.B_lag[0] = par.B_lag_start

        # c. iterate 
        for t in range(par.simT):
            
            # i. simulate before known s
            sim_b(par,sim,t)

            # ii. stop in the last period 
            if t == par.simT-1: 
                continue          

            # iii. find bracket to search for optimal s
            s_min, s_max = find_s(par, sim, t)

            # iv. define objective funcion to root find 
            obj = lambda s: euler(s, par, sim, t)
            
            # v. find the optimal s
            s_result = optimize.root_scalar(obj, bracket=(s_min,s_max), method='bisect')
            
            # vi. store the result as s
            s = s_result.root

            # vii. simulate after known s
            sim_a(par, sim, t, s)

def find_s(par,sim,t,maxiter=1000):
    """ Find a bracket for s that can be used to minimize the Euler error """

    # a. starting points for s
    # i. saving almost nothing
    s_min = 0.0 + 1e-8
    # ii. saving almost everything
    s_max = 1.0 - 1e-8

    # b. find Euler error when saving everything 
    value_max = euler(s_max, par, sim, t)

    # c. set new names for s_min and s_max 
    s_min_1 = s_min
    s_max_1 = s_max

    # d. find s
    it = 0
    while it < maxiter:
        
        # i. calculate midpoint
        s = (s_min_1 + s_max_1) / 2

        # ii. Euler error for s 
        value = euler(s, par, sim, t)

        # iii. check if conditions are met
        if not np.isnan(value) and np.sign(value) * np.sign(value_max) < 0:
            # o. found brackets
            s_min = s
            s_max = s_max_1
            return s_min, s_max

        # iii. update brackets 
        elif np.isnan(value):
            s_min_1 = s
        else:
            s_max_1 = s
        
        it += 1

    raise Exception('cannot find bracket for s')

def euler(s, par, sim, t):
    """ Find the euler error """

    # a. simulate forward
    sim_a(par, sim, t, s)
    sim_b(par, sim, t+1) 

    # c. Euler equation
    # i. right hand side 
    C1_l = sim.C1[t]**(-par.sigma)
    # ii. left hand side 
    C2_r = (1+sim.rt[t+1])*par.beta * sim.C2[t+1]**(-par.sigma)

    # d. return the diff = Euler error 
    return C1_l-C2_r

def sim_b(par, sim, t):
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
    sim.C2[t] = (1+sim.rt[t])*(sim.K_lag[t]+sim.B_lag[t])

    # f. firm profits 
    sim.pi = sim.Y[t] - sim.w[t]*par.n*par.At - sim.rk[t]*sim.K_lag

    # g. government tax revenues 
    sim.T[t] = par.tau_r*sim.r[t]*(sim.K_lag[t]+sim.B_lag[t]) + par.tau_w*sim.w[t]*par.n*par.At

    # h. set a balanced budget for the government 
    if sim.balanced_budget[t]:
        sim.G[t] = sim.T[t] - sim.r[t]*sim.B_lag[t]

    # i. government debt 
    sim.B[t] = (1+sim.r[t])*sim.B_lag[t] - sim.T[t] + sim.G[t]


def sim_a(par,sim,t,s):
    """ 
    Find the values of young consumption (C1), investments (I) and capital (K) after the savings are found. 
    """

    # a. consumption of young
    sim.C1[t] = (1-par.tau_w)*sim.w[t]*(1.0-s)*par.n*par.At - par.d
    
    # b. consumption of old to t+1 
    sim.C2_1[t+1] = (1+sim.rt[t+1])*(sim.K_lag[t+1]+sim.B_lag[t+1]) + par.d

    # c. end-of-period stocks
    # i. investments 
    I = sim.Y[t] - sim.C1[t] - sim.C2[t] - sim.G[t]
    # ii. capital 
    sim.K[t] = (1-par.delta)*sim.K_lag[t] + I

    # d. find utility
    if par.sigma == 1.0:
        sim.u = np.log(sim.C1[t]) + par.beta*np.log(sim.C2_1[t])
    if par.sigma > 1.0 or par.sigma < 1.0:
        sim.u = (sim.C1[t]**(1-par.sigma))/(1-par.sigma) + par.beta*(sim.C2_1[t]**(1-par.sigma))/(1-par.sigma)


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
        ax.plot(K_lag_int, label=r'$K_{t-1}$ (initial)', color = 'black')
    
    # c. adjust the x-tickers to show every 2 
    ax.set_xlim(0,20)
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::2])

    # d. add legend 
    ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.3), fontsize = 10, ncol = 2)

    # e. make it look pretty and show the beauty 
    fig.tight_layout()
    plt.show()


def plot_C(C1, C1_2, C2 = None, C2_2 = None, C2_1 = None,  C2_1_2 = None, title = None):
    """ Plot the consumption """
    # a. setup for figure
    fig = plt.figure(figsize = (2*6,8/1.5))
    
    # b. plot C_1 
    ax = fig.add_subplot(1,2,1)
    ax.plot(C1, label=r'$C_{1t}$ (initial)', color = 'black')
    ax.plot(C1_2, label=r'$C_{1t}$')
    
    # c. add legend (located under plot and two columns)
    ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.3), fontsize = 10, ncol = 2)

    # d. adjust the x-tickers
    ax.set_xlim(0,20)
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::1])

    # e. set the title
    if title is not None:
        ax.set_title(title, loc = 'left')

    # f. plot C_2
    ax = fig.add_subplot(1,2,2)
    # i. C_2t
    if C2 is not None: 
        ax.plot(C2, label=r'$C_{2t}$ (initial)', color = 'black')
    if C2_2 is not None: 
        ax.plot(C2_2, label=r'$C_{2t}$')
    # ii. C_2t+1
    if C2_1 is not None: 
        ax.plot(C2_1, label=r'$C_{2t+1}$ (initial)', color = 'black')
    if C2_1_2 is not None: 
        ax.plot(C2_1_2, label=r'$C_{2t+1}$')

    # g. add legends (located under plot and two columns)
    ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.3), fontsize = 10, ncol = 2)

    # h. adjust the x-tickers
    ax.set_xlim(0,20)
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::1])

    # i. make it look pretty and show the beauty 
    fig.tight_layout()
    plt.show()


def plot_pi(pi_n_At = None, pi_n = None, pi_int = None):
    """ Plot the firms profit """

    # a. setup for figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # b. plot 
    # i. pupolation and technological growth 
    if pi_n_At is not None: 
        ax.plot(pi_n_At, label=r'$\Pi_t$ (pop and tech growth)', color = 'red')
    # ii. population 
    if pi_n is not None:
        ax.plot(pi_n, label=r'$\Pi_t$ (pop growth)', color = 'blue')
    # iii. technology
    if pi_int is not None:
        ax.plot(pi_int, label=r'$\Pi_t$ (initial)', color = 'black')

    # c. adjust the x-tickers
    ax.set_xlim(0,20)
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::2])

    # d. add legend 
    ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.3), fontsize = 10, ncol = 2)

    # e. make it look pretty and show the beauty 
    fig.tight_layout()
    plt.show()

def utility_table(lst):
    utilities = []
    for model in lst:
        utility = model.sim.u
        utilities.append(utility)
    
    df = pd.DataFrame(utilities, columns=['Utility'])
    df['Model'] = range(len(lst))
    df = df[['Model', 'Utility']]
    df = df.set_index('Model', drop=True)

    return df

def plot_T_B(T_int, T_tau_w, T_tau_r, B_int, B_tau_w, B_tau_r):

    # a. setup for figure
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (2*6,8/1.5))

    # b. plot tax revenue 
    # i. plot 
    ax1.plot(T_int, label='Initial')
    ax1.plot(T_tau_w, label='$\\tau_w$')
    ax1.plot(T_tau_r, label='$\\tau_r$')
    # ii. adjust x-tickers 
    ax1.set_xlim(0, 20)
    xticks = ax1.get_xticks()
    ax1.set_xticks(xticks[::1])
    # iii. set title 
    ax1.set_title('Tax revenue', loc='left')

    # c. plot debt
    # i. plot 
    ax2.plot(B_int, label='Initial')
    ax2.plot(B_tau_w, label='$\\tau_w$')
    ax2.plot(B_tau_r, label='$\\tau_r$')
    # ii. adjust x-tickers 
    ax2.set_xlim(0, 20)
    xticks = ax2.get_xticks()
    ax2.set_xticks(xticks[::1])
    # iii. set title 
    ax2.set_title('Debt', loc='left')

    # d. add shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'lower center', ncol = 3, bbox_to_anchor = (0.5, -0.15))

    # e. make it look pretty and show the beauty
    fig.tight_layout()
    plt.show()

    







