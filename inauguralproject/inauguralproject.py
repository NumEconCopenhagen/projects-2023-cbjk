from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

        # g. constraints 
        par.constraints = ({'type': 'ineq', 'fun': lambda x: 24 - (x[0]+x[1])},{'type': 'ineq', 'fun': lambda x: 24 - (x[2]+x[3])})
       
       


    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production (denne skal ændres)
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
            
        elif par.sigma == 0:
            H = np.min(HM, HF)
           
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))
        

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility



    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]
        opt.u = u[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def plot_illustration(self, plot_dataframe):
        """ Code to make the illutrations """
        
        #a. contruct the figure 
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        #b. adds the variables 
        ax.plot(plot_dataframe['w ratio'],plot_dataframe['H ratio'])
        
        #c. set title and labels 
        ax.set_title('Log relationship between ratio of hours worked and ratio of wage')
        ax.set_xlabel('$log(\omega_F/\omega_M)$')
        ax.set_ylabel('$log(H_F/H_M)$')


    def solve(self, do_print=False):
        """ solve model continously """
  
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # a. objective function (to minimize) 
        def objective(x):
            return -self.calc_utility(x[0], x[1], x[2], x[3])
        
        obj = lambda x: objective(x)
        guess = [3,5.5,5,4]
        bounds = [(0,24)]*4

        # b. optimizer
        result = optimize.minimize(obj,
                            guess,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=par.constraints)
        
        #c. store results
        opt.LM = result.x[0]
        opt.HM = result.x[1]
        opt.LF = result.x[2]
        opt.HF = result.x[3]
        opt.u =self.calc_utility(opt.LM, opt.HM, opt.LF, opt.HF)

        #d. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
        
      

    def solve_wF_vec(self,discrete=False):
        """ 
        solve model for vector of female wages 
        """

        par = self.par
        sol = self.sol

        #a. loop over relative wage and solve model
        for i in par.wF_vec:
            par.wF = i
            if discrete: # for discrete choice model
                results = self.solve_discrete()
            else: #for continous choice model
                results = self.solve()
            # i. find index of argument  
            j = np.where(par.wF_vec ==i)[0][0]

            # ii. store results
            sol.LM_vec[j] = results.LM
            sol.HM_vec[j] = results.HM
            sol.LF_vec[j] = results.LF
            sol.HF_vec[j] = results.HF

        return sol
    

    def run_regression(self):
        """ 
        run regression 
        """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

    
    def estimate(self,alpha=None):
        """ estimate alpha and sigma for variable and fixed alpha """
        par = self.par
        sol = self.sol
        # a. alpha
        if alpha == None:
            # i. objective function (to minimize) 
            def objective(y):
                par.alpha = y[1] #variable
                par.sigma = y[0] #variable
                self.solve_wF_vec()
                self.run_regression()
                return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2

            obj = lambda y: objective(y)
            guess = [0.5]*2
            bounds = [(-0.00001,1)]*2
            # ii. optimizer
            result = optimize.minimize(obj,
                                guess,
                                method='Nelder-Mead',
                                bounds=bounds)
            # iii. print
            print(f'alpha = {result.x[1]}')
            print(f'sigma = {result.x[0]}')
        else:
            # i. objective function (to minimize)
            def objective(y):
                par.alpha = alpha #chosen alpha
                par.sigma = y[0] #variables
                self.solve_wF_vec()
                self.run_regression()
                return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2

            obj = lambda y: objective(y)
            guess = [0.5]
            bounds = [(-0.00001,1)]
            # ii. optimizer
            result = optimize.minimize(obj,
                                guess,
                                method='Nelder-Mead',
                                bounds=bounds)
          
            return result
        

