from types import SimpleNamespace
import numpy as np
from scipy import optimize
import pandas as pd 
import matplotlib.pyplot as plt

# Class 
class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces, and set par and sol 
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

        # g. betas of nan 
        sol.beta0 = np.nan
        sol.beta1 = np.nan

        # h. constraints 
        par.constraints = ({'type': 'ineq', 'fun': lambda x: 24 - (x[0]+x[1])},{'type': 'ineq', 'fun': lambda x: 24 - (x[2]+x[3])})

    def calc_utility(self,LM,HM,LF,HF):
        """ Calculate utility """

        # a. set par 
        par = self.par

        # b. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # c. home production
        # i. Cobb-Douglas 
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        # ii. minimum 
        elif par.sigma == 0:
            H = np.min(HM, HF)
        # iii. ces 
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))
        
        # d. total consumption utility
        Q = C**par.omega * H**(1-par.omega)

        # e. utility 
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # f. disutlity of work
        epsilon_ = 1+1/par.epsilon
        # i. total hours for men 
        TM = LM+HM
        # ii. total hours for women 
        TF = LF+HF
        # iii. disutility
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        # g. return the difference between utility and disutility
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        # a. set opt 
        opt = SimpleNamespace()
        
        # b. all possible choices
        x = np.linspace(0, 24, 49)

        # c. all combinations 
        LM, HM, LF, HF = np.meshgrid(x, x, x, x) 
    
        # d. vectorize LM,HM,LF and HF
        LM = LM.ravel()
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # e. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # f. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24)
        u[I] = -np.inf
    
        # g. find maximizing argument
        j = np.argmax(u)
        
        # h. store LM, HM, LF, HF and utility in opt 
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]
        opt.u = u[j]

        # i. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self, do_print=False):
        """ Solve model continously """

        # a. set par and opt
        par = self.par
        opt = SimpleNamespace()

        # b. objective function (to minimize) 
        def objective(x):
            return -self.calc_utility(x[0], x[1], x[2], x[3])
        # i. objective function 
        obj = lambda x: objective(x)
        # ii. guess on LM, HM, LF and HF
        guess = [3,5.5,5,4]
        # iii. sets bounds for hours on a day 
        bounds = [(0,24)]*4

        # c. optimizer using the 'SLSQP' 
        result = optimize.minimize(obj,
                                guess,
                                method = 'SLSQP',
                                bounds = bounds,
                                constraints = par.constraints)
        
        # d. store results of LM, HM, LF and HF
        opt.LM = result.x[0]
        opt.HM = result.x[1]
        opt.LF = result.x[2]
        opt.HF = result.x[3]

        # e. find the utility 
        opt.u = self.calc_utility(opt.LM, opt.HM, opt.LF, opt.HF)

        # f. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve_wF_vec(self,discrete=False):
        """ 
        Solve model for vector of female wages 
        """

        # a. set par and sol 
        par = self.par
        sol = self.sol

        # b. loop over relative wage and solve model
        for i in par.wF_vec:
            par.wF = i
            # i. discrete choice model
            if discrete:
                results = self.solve_discrete()
            # ii. continous choice model
            else:
                results = self.solve()
            # iii. find index of argument  
            j = np.where(par.wF_vec == i)[0][0]

            # iv. store results
            sol.LM_vec[j] = results.LM
            sol.HM_vec[j] = results.HM
            sol.LF_vec[j] = results.LF
            sol.HF_vec[j] = results.HF

        # c. compute H and w log ratio
        sol.logratioH = np.log(sol.HF_vec/sol.HM_vec)
        sol.logratiow = np.log(par.wF_vec/par.wM)

        return sol
    
    def run_regression(self):
        """ 
        Run regression 
        """

        # a. set par and sol 
        par = self.par
        sol = self.sol

        # b. set x, y and A for regression 
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T

        # c. find beta0 and beta1 from the regression 
        sol.beta0, sol.beta1 = np.linalg.lstsq(A, y, rcond = None)[0]

    
    def estimate(self,alpha=None, do_print = False):
        """ estimate alpha and sigma for variable and fixed alpha """

        # a. set par and sol 
        par = self.par
        sol = self.sol

        # b. if alpha is not given  
        if alpha == None:
            # i. objective function (to minimize) 
            def objective(y):
                # o. variable alpha 
                par.alpha = y[1]
                # oo. varibale sigma 
                par.sigma = y[0] 
                # ooo. solve solve_wF_vec
                self.solve_wF_vec()
                # oooo. run regression
                self.run_regression()
                # ooooo. return the diffrence between target beta and estimated 
                return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2
            # ii. define objective function 
            obj = lambda y: objective(y)
            # iii. guess for alpha and sigma  
            guess = [0.5]*2
            # iv. bounds
            bounds = [(-0.00001,1)]*2
            # v. optimizer
            result = optimize.minimize(obj,
                                guess,
                                method='Nelder-Mead',
                                bounds=bounds)
            
            # vi. print result 
            if do_print:
                print(f'alpha = {result.x[1].round(4)}')
                print(f'sigma = {result.x[0].round(4)}')
        
        # c. if alpha is given  
        else:
            # i. objective function (to minimize)
            def objective(y):
                # o. chosen alpha
                par.alpha = alpha 
                # oo. variables
                par.sigma = y[0] 
                # ooo. solve solve_wF_vec
                self.solve_wF_vec()
                # oooo. run regression 
                self.run_regression()
                # ooooo. return the diffrence between target beta and estimated 
                return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2
            # ii. define objective function 
            obj = lambda y: objective(y)
            # iii. guess for sigma 
            guess = [0.5]
            # iv. bounds 
            bounds = [(-0.00001,1)]
            # v. optimizer
            result = optimize.minimize(obj,
                                guess,
                                method = 'Nelder-Mead',
                                bounds = bounds)
            # vi. print result
            if do_print: 
                print(f'sigma = {result.x[0].round(4)}')

            return result

# Other functions 
def plot_3d(alpha, sigma, df):
    """ Function that makes a 3D-plot """

    # a. make meshgrid of alpha and sigma
    alpha_mesh, sigma_mesh = np.meshgrid(alpha, sigma)

    # b. setup the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # c. plot sigma, alpha and df 
    ax.plot_surface(sigma_mesh, alpha_mesh, df, cmap='viridis', edgecolor='none')
    
    # d. set ax labels 
    ax.set_xlabel('sigma')
    ax.set_ylabel('alpha')
    ax.set_zlabel('HF/HM Ratio')

    # e. set tickers such it match alpha and sigma
    ax.set_yticks(alpha)
    ax.set_xticks(sigma)

    # f. show the figure 
    plt.show()

def plot_illustration(plot_dataframe):
    """ Code to make the illutrations """
    
    # a. contruct the figure 
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # b. adds the variables 
    ax.plot(plot_dataframe['w ratio'], plot_dataframe['H ratio'])
    
    # c. set title and labels 
    ax.set_title('Log relationship between ratio of hours worked and ratio of wage')
    ax.set_xlabel('$log(\omega_F/\omega_M)$')
    ax.set_ylabel('$log(H_F/H_M)$')

    # d. show the figure 
    plt.show()