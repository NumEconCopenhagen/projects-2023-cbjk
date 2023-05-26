from random import uniform
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from types import SimpleNamespace
import time

class GoMs:
    
    def __init__(self, **kwargs):
        """ initialize class """

        # a. set par
        par = self.par = SimpleNamespace()

        # b. baseline parameters
        par.tau = 1e-8
        par.K_bar = 10
        par.K = 1000

    def global_opt(self):
        """ global optimization """

        # a. Start the timer
        start_time = time.time()

        # b. set par 
        par = self.par

        # c. define griewank function
        def griewank1(x1, x2):
            A = x1**2 / 4000 + x2**2 / 4000
            B = np.cos(x1 / np.sqrt(1)) * np.cos(x2 / np.sqrt(2))
            return A - B + 1

        # d. define griewank function for vector input
        def griewank(x):
            return griewank1(x[0], x[1])

        # e. set seed
        np.random.seed(2000)

        # f. initialize
        x_star = np.empty(2)
        fopt = np.inf
        xs = np.empty((par.K, 2))
        x_k0 = np.random.uniform(low=-600, high=600, size=2)  # Initialize x_k0 before the loop

        # g. loop over K
        for k in range(par.K):
            
            # i. step 3.A 
            x_k = np.random.uniform(low=-600, high=600, size=2)

            # ii. step 3.B
            if k < par.K_bar:
                x_k0 = x_k
            else:
            # iii. step 3.C
                chi_k = 0.50 * 2 / (1 + np.exp((k - par.K_bar) / 100))
            # iv. step 3.D
                x_k0 = chi_k * x_k + (1 - chi_k) * x_star

            # v. step 3.E
            # o. optimize
            result = optimize.minimize(griewank, x_k0, method='BFGS', tol=par.tau)
            # oo. update x_kstar and f 
            x_kstar = result.x
            f = result.fun
            # ooo. update xs at k 
            xs[k, :] = x_kstar

            # iv. step 3.F
            if k == 0 or (griewank(x_kstar) < griewank(x_star)):
                x_star = x_kstar

            # v. opdates f
            if f < fopt:
                fopt = f
            
            # vi. print iterations 
            if k > 10 and k < 100:
                if k % 20 == 0:
                    print(f'k = {k:4d}: x_k0 = ({x_k0[0]:7.2f},{x_k0[1]:7.2f})', end='')
                    print(f' -> converged to ({xs[k][0]:7.2f},{xs[k][1]:7.2f}) with f = {f:12.8f}')
            elif k > 100:
                if k % 100 == 0:
                    print(f'k = {k:4d}: x_k0 = ({x_k0[0]:7.2f},{x_k0[1]:7.2f})', end='')
                    print(f' -> converged to ({xs[k][0]:7.2f},{xs[k][1]:7.2f}) with f = {f:12.8f}')
            else:
                print(f'k = {k:4d}: x_k0 = ({x_k0[0]:7.2f},{x_k0[1]:7.2f})', end='')
                print(f' -> converged to ({xs[k][0]:7.2f},{xs[k][1]:7.2f}) with f = {f:12.8f}')

            # vi. step 3.G  
            if fopt < par.tau:
                break
        
        # h. calculate the elapsed time
        elapsed_time = time.time() - start_time
        
        # i. print last iteration 
        print(f'k = {k:4d}: x_k0 = ({x_k0[0]:7.2f},{x_k0[1]:7.2f})', end='')
        print(f' -> converged to ({x_star[0]:7.2f},{x_star[1]:7.2f}) with f = {f:12.8f}')

        # j. print best solution
        print(f'\nk = {k} found the best solution.')
        print(f'\nBest solution:\n x_star = ({xs[k, 0]:7.2f},{xs[k, 1]:7.2f}) -> f = {fopt:12.8f}')

        # k. print best solution with 10 decimals
        print(f'\nBest solution with 10 decimals:\n x_star = ({x_star[0]:7.10f},{x_star[1]:7.10f}) -> f = {griewank(x_star):12.8f}')

        # l. print the elapsed time
        print(f'\nElapsed time: {elapsed_time:7.2f} seconds')
