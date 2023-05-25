from random import uniform
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from types import SimpleNamespace

class GoMs:
    
    def __init__(self, **kwargs):

        # a. set par
        par = self.par = SimpleNamespace()

        # b. baseline parameters
        par.tau = 1e-8
        par.K_bar = 10
        par.K = 1000

    def global_opt(self):

        # a. set par 
        par = self.par

        # b. define griewank function
        def griewank1(x1, x2):
            A = x1**2 / 4000 + x2**2 / 4000
            B = np.cos(x1 / np.sqrt(1)) * np.cos(x2 / np.sqrt(2))
            return A - B + 1

        # c. define griewank function for vector input
        def griewank(x):
            return griewank1(x[0], x[1])

        # d. set seed
        np.random.seed(2000)

        # e. initialize
        x_star = np.empty(2)
        x_kstar = np.empty(2)
        fopt = np.inf
        xopt = np.nan
        xs = np.empty((par.K, 2))
        x_k0 = np.random.uniform(low=-600, high=600, size=2)  # Initialize x_k0 before the loop

        # f. loop over K
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
            xs[k, :] = result.x

            # iv. step 3.F
            if k == 0 or (griewank(x_kstar) < griewank(x_star)):
                x_star = x_kstar
            
            # v. step 3.F
            if k == 0 or f < fopt:
                if f < fopt:
                    fopt = f
                    xopt = xs[k, :]
            
            # vi. print
            if k > 10 and k < 100:
                if k % 20 == 0:
                    print(f'{k:4d}: x = ({x_k0[0]:7.2f},{x_k0[1]:7.2f})', end='')
                    print(f' -> converged at ({xs[k][0]:7.2f},{xs[k][1]:7.2f}) with f = {f:12.8f}')
            elif k > 100:
                if k % 50 == 0:
                    print(f'{k:4d}: x = ({x_k0[0]:7.2f},{x_k0[1]:7.2f})', end='')
                    print(f' -> converged at ({xs[k][0]:7.2f},{xs[k][1]:7.2f}) with f = {f:12.8f}')
            else:
                print(f'{k:4d}: x = ({x_k0[0]:7.2f},{x_k0[1]:7.2f})', end='')
                print(f' -> converged at ({xs[k][0]:7.2f},{xs[k][1]:7.2f}) with f = {f:12.8f}')

            # vi. step 3.G  
            if fopt < par.tau:
                break
        
        print(f'{k:4d}: x = ({x_k0[0]:7.2f},{x_k0[1]:7.2f})', end='')
        print(f' -> converged at ({xs[-1][0]:7.2f},{xs[-1][1]:7.2f}) with f = {f:12.8f}')


        # g. print best solution (step 4)
        print(f'\nbest solution:\n x = ({xopt[0]:7.2f},{xopt[1]:7.2f}) -> f = {fopt:12.8f}')
        return x_star


# class GoMs:
    
#     def __init__(self, **kwargs):

#         # a. set par
#         par = self.par = SimpleNamespace()

#         # b. baseline parameters
#         par.tau = 1e-8
#         par.K_bar = 10
#         par.K = 1000

#     def global_opt(self):

#         # a. set par 
#         par = self.par

#         # b. define griewank function
#         def griewank1(x1, x2):
#             A = x1**2 / 4000 + x2**2 / 4000
#             B = np.cos(x1 / np.sqrt(1)) * np.cos(x2 / np.sqrt(2))
#             return A - B + 1

#         # c. define griewank function for vector input
#         def griewank(x):
#             return griewank1(x[0], x[1])

#         # d. set seed
#         np.random.seed(2000)

#         # e. initialize
#         x_star = np.empty(2)
#         x_kstar = np.empty(2)
#         fopt = np.inf
#         xopt = np.nan
#         xs = np.empty((par.K, 2))
#         x_k0 = np.random.uniform(low=-600, high=600, size=2)  # Initialize x_k0 before the loop

#         # f. loop over K
#         for k in range(par.K):
#             # Step 3.A: Draw random x^k
#             x_k = np.random.uniform(low=-600, high=600, size=2)

#             # Step 3.C and 3.D: Check warm-up iterations and calculate chi^k
#             if k >= par.K_bar:
#                 chi_k = 0.50 * 2 / (1 + np.exp((k - par.K_bar) / 100))
#                 x_k0 = chi_k * x_k + (1 - chi_k) * x_star
#             else:
#                 x_k0 = x_k

#             # Step 3.E: Run optimizer
#             result = optimize.minimize(griewank, x_k0, method='BFGS', tol=self.par.tau)
#             x_kstar = result.x
#             f = result.fun

#             # Step 3.F: Update x_star if necessary
#             if k == 0 or f < fopt:
#                 x_star = x_kstar
#                 fopt = f

#             # Step 3.B: Print iteration details
#             if k > 10 and k < 100:
#                 if k % 20 == 0:
#                     print(f'{k:4d}: x = ({x_k0[0]:7.2f},{x_k0[1]:7.2f})', end='')
#                     print(f' -> converged at ({xs[k][0]:7.2f},{xs[k][1]:7.2f}) with f = {f:12.8f}')
#             elif k > 100:
#                 if k % 50 == 0:
#                     print(f'{k:4d}: x = ({x_k0[0]:7.2f},{x_k0[1]:7.2f})', end='')
#                     print(f' -> converged at ({xs[k][0]:7.2f},{xs[k][1]:7.2f}) with f = {f:12.8f}')
#             else:
#                 print(f'{k:4d}: x = ({x_k0[0]:7.2f},{x_k0[1]:7.2f})', end='')
#                 print(f' -> converged at ({xs[k][0]:7.2f},{xs[k][1]:7.2f}) with f = {f:12.8f}')

#             # Step 3.G: Check termination condition
#             if fopt < par.tau:
#                 break

#             # Step 3.H: Update xs at k
#             xs[k, :] = result.x

#         # Step 4: Print best solution
#         print(f'\nbest solution:\n x = ({xopt[0]:7.2f},{xopt[1]:7.2f}) -> f = {fopt:12.8f}')
#         return x_star     
            
