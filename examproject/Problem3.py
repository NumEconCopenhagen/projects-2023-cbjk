from random import uniform
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from types import SimpleNamespace

class GoMs:
    
    def __init__(self,**kwargs): # called when created

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()
        
        # a. baseline settings
        par.tau = 10**(-8)
        par.K_bar = 10
        par.K = 1000

    def global_opt(self):
        par = self.par

        def griewank1(x1,x2):
            A = x1**2/4000 + x2**2/4000
            B = np.cos(x1/np.sqrt(1))*np.cos(x2/np.sqrt(2))
            return A-B+1

        def griewank(x):
            return griewank1(x[0],x[1])
    
        x_k = np.empty(2)
        x_k0 = np.random.uniform(low=-60, high=60, size=2)
        x_star = np.empty(2)
        x_kstar = np.empty(2)
        fopt = np.inf
        xopt = np.nan
        xs = np.empty((5000,2))
        #fs = np.empty(5000)

        np.random.seed(2023)
        for k in range(par.K):
            print(k)
            while griewank(x_star) >= par.tau:
                x_k = np.random.uniform(low=-600, high=600, size=2)
                if k >= par.K_bar:
                    chi_k = 0.50 * 2/(1+np.exp(k - par.K_bar)/100)
                    x_k0 = chi_k*x_k+(1-chi_k)*x_star
                    
                # a. optimize
                result = optimize.minimize(griewank,x_k0,method='BFGS', tol=par.tau)
                x_kstar = result.x
                f = result.fun
                xs[k,:] = result.x
                if k == 0 or griewank(x_kstar) < griewank(x_star):
                    x_star = x_kstar
            
    
            # b. print first 10 or if better than seen yet
            if k < 10 or f < fopt: # plot 10 first or if improving
                if f < fopt:
                    fopt = f
                    xopt = xs[i,:]
                    
                print(f'{i:4d}: x = ({x_k0[0]:7.2f},{x_k0[1]:7.2f})',end='')
                print(f' -> converged at ({xs[k][0]:7.2f},{xs[k][1]:7.2f}) with f = {f:12.8f}')
                
                # best solution
                print(f'\nbest solution:\n x = ({xopt[0]:7.2f},{xopt[1]:7.2f}) -> f = {fopt:12.8f}')
        return x_star
                            
        
            
            
            
