from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class OLGModelClass():

    def __init__(self):
        """ create the model """
        # a. set SimpleNamespace
        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        # b. set setup the class 
        self.setup()

        # c. set allecate of varnames
        self.allocate()
    
    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. household
        par.sigma = 1.0    # CRRA coefficient
        par.beta = 1/1.40  # discount factor

        # b. firms
        par.production_function = 'ces'
        par.alpha = 0.30  # capital weight
        par.theta = 0.0   # substitution parameter
        par.delta = 0.50  # depreciation rate

        # c. government
        par.tau_w = 0.0  # labor income tax
        par.tau_r = 0.0  # capital income tax

        # d. misc
        par.K_lag_ini = 1.0   # initial capital stock
        par.B_lag_ini = 0.0   # initial government debt
        par.simT = 50         # length of simulation

        # e. population and technology  
        par.n = 1.0
        par.At = 1.0

        # f. PAYG
        par.d = 0.0


    def allocate(self):
        """ allocate arrays for simulation """
        
        par = self.par
        sim = self.sim

        # a. list of variables
        household = ['C1','C2']
        firm = ['K','Y','K_lag']
        prices = ['w','rk','rb','r','rt']
        government = ['G','T','B','balanced_budget','B_lag']

        # b. allocate
        allvarnames = household + firm + prices + government
        for varname in allvarnames:
            sim.__dict__[varname] = np.nan*np.ones(par.simT)

    def simulate(self, do_print=True, pop = False, At = False):
        """ simulate model """

        t0 = time.time()

        par = self.par
        sim = self.sim
        
        # a. initial values
        sim.K_lag[0] = par.K_lag_ini
        sim.B_lag[0] = par.B_lag_ini

        # b. iterate
        for t in range(par.simT):
            
            # i. simulate before s
            simulate_before_s(par,sim,t)

            if t == par.simT-1: continue          

            # i. find bracket to search
            s_min,s_max = find_s_bracket(par,sim,t)

            # ii. find optimal s
            obj = lambda s: calc_euler_error(s,par,sim,t=t)
            result = optimize.root_scalar(obj,bracket=(s_min,s_max),method='bisect')
            s = result.root

            # iii. simulate after s
            simulate_after_s(par,sim,t,s)

        if do_print: print(f'simulation done in {time.time()-t0:.2f} secs')

def find_s_bracket(par,sim,t,maxiter=500,do_print=False):
    """ find bracket for s to search in """

    # a. maximum bracket
    s_min = 0.0 + 1e-8 # save almost nothing
    s_max = 1.0 - 1e-8 # save almost everything

    # b. saving a lot is always possible 
    value = calc_euler_error(s_max,par,sim,t)
    sign_max = np.sign(value)
    if do_print: print(f'euler-error for s = {s_max:12.8f} = {value:12.8f}')

    # c. find bracket      
    lower = s_min
    upper = s_max

    it = 0
    while it < maxiter:
                
        # i. midpoint and value
        s = (lower+upper)/2 # midpoint
        value = calc_euler_error(s,par,sim,t)

        if do_print: print(f'euler-error for s = {s:12.8f} = {value:12.8f}')

        # ii. check conditions
        valid = not np.isnan(value)
        correct_sign = np.sign(value)*sign_max < 0
        
        # iii. next step
        if valid and correct_sign: # found!
            s_min = s
            s_max = upper
            if do_print: 
                print(f'bracket to search in with opposite signed errors:')
                print(f'[{s_min:12.8f}-{s_max:12.8f}]')
            return s_min,s_max
        elif not valid: # too low s -> increase lower bound
            lower = s
        else: # too high s -> increase upper bound
            upper = s

        # iv. increment
        it += 1

    raise Exception('cannot find bracket for s')

def calc_euler_error(s,par,sim,t):
    """ target function for finding s with bisection """

    # a. simulate forward
    simulate_after_s(par,sim,t,s)
    simulate_before_s(par,sim,t+1) # next period

    # c. Euler equation
    LHS = sim.C1[t]**(-par.sigma)
    RHS = (1+sim.rt[t+1])*par.beta * sim.C2[t+1]**(-par.sigma)

    return LHS-RHS

def simulate_before_s(par,sim,t):
    """ simulate forward """

    if t > 0:
        sim.K_lag[t] = sim.K[t-1]
        sim.B_lag[t] = sim.B[t-1]

    # a. production and factor prices
    if par.production_function == 'ces':
        # i. production
        sim.Y[t] = (par.alpha*sim.K_lag[t]**(-par.theta) + (1-par.alpha)*(par.n*par.A)**(-par.theta) )**(-1.0/par.theta)

        # ii. factor prices
        sim.rk[t] = par.alpha*sim.K_lag[t]**(-par.theta-1) * sim.Y[t]**(1.0+par.theta)
        sim.w[t] = (1-par.alpha)*(par.n*par.A)**(-par.theta-1) * sim.Y[t]**(1.0+par.theta)

    elif par.production_function == 'cobb-douglas':
        # i. production
        sim.Y[t] = sim.K_lag[t]**par.alpha * par.n*par.At**(1-par.alpha)

        # ii. factor prices
        sim.rk[t] = par.alpha*sim.K_lag[t]**(par.alpha-1) * (par.n*par.At)**(1-par.alpha)
        sim.w[t] = (1-par.alpha)*sim.K_lag[t]**(par.alpha) * (par.n*par.At)**(-par.alpha)

    else:
        raise NotImplementedError('unknown type of production function')
    

    # b. no-arbitrage and after-tax return
    sim.r[t] = sim.rk[t]-par.delta      # after-depreciation return
    sim.rb[t] = sim.r[t]                # same return on bonds
    sim.rt[t] = (1-par.tau_r)*sim.r[t]  # after-tax return

    # c. consumption
    sim.C2[t] = (1+sim.rt[t])*(sim.K_lag[t]+sim.B_lag[t]) + par.d

    # d. government
    sim.T[t] = par.tau_r*sim.r[t]*(sim.K_lag[t]+sim.B_lag[t]) + par.tau_w*sim.w[t]*par.n*par.At + par.d

    if sim.balanced_budget[t]:
        sim.G[t] = sim.T[t] - sim.r[t]*sim.B_lag[t]

    sim.B[t] = (1+sim.r[t])*sim.B_lag[t] - sim.T[t] + sim.G[t]

def simulate_after_s(par,sim,t,s):
    """ simulate forward """

    # a. consumption of young
    sim.C1[t] = (1-par.tau_w)*sim.w[t]*(1.0-s)*par.n*par.At - par.d

    # b. end-of-period stocks
    I = sim.Y[t] - sim.C1[t] - sim.C2[t] - sim.G[t]
    sim.K[t] = (1-par.delta)*sim.K_lag[t] + I
     
    
def plot_K(K_lag, K_ss = None, K_lag_int = None, K_lag_old = None):
    # a. setup for figure
    fig = plt.figure(figsize=(6,6/1.5))
    ax = fig.add_subplot(1,1,1)

    # b. plot K_lag and k_ss
    ax.plot(K_lag, label=r'$K_{t-1}$', color = 'red')
    if K_ss is not None:
        ax.axhline(K_ss, ls='--', color='black', label='analytical steady state')
    if K_lag_old is not None:
        ax.plot(K_lag_old, label=r'$K_{t-1}$ (last simulation)', color = 'blue')
    if K_lag_int is not None:
        ax.plot(K_lag_int, label=r'$K_{t-1}$ (initial simulation)', color = 'darkblue')
   
    # c. add legend 
    ax.legend(frameon=True)

    # d. make it look pretty and show the beauty 
    fig.tight_layout()
    plt.show()

def plot_C(C1, C1_payg, C2, C2_payg, title = None):
    # a. setup for figure
    fig = plt.figure(figsize=(2*6,6/1.5))
    
    # b. plot C_1 
    ax = fig.add_subplot(1,2,1)
    ax.plot(C1, label=r'$C_{1t}$ (initial)')
    ax.plot(C1_payg, label=r'$C_{1t}$')
    
    # c. add legend
    ax.legend(frameon=True)

    # f. set the title
    if title is not None:
        ax.set_title(title, loc = 'left')

    # d. plot C_2
    ax = fig.add_subplot(1,2,2)
    ax.plot(C2, label=r'$C_{2t+1}$ (initial)')
    ax.plot(C2_payg, label=r'$C_{2t+1}$')

    # e. add legends 
    ax.legend(frameon=True)

    # g. make it look pretty and show the beauty 
    fig.tight_layout()
    plt.show()









