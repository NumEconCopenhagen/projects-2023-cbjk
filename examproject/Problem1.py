import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy import optimize

class OptTax():

    def __init__(self):
        """ set initial  """

        # a. set SimpleNamespace
        self.par = SimpleNamespace()

        # b. set setup the class 
        self.setup()


    def setup(self):
        """ baseline parameters """

        # a. set par
        par = self.par

        # b. set parameters
        par.alpha = 0.5
        par.kappa = 1.0
        par.nu = 1.0/(2*(16**2))
        par.omega = 1.0 
        par.tau = 0.30 
        par.G = 1.0

        # c. define omega_tilde
        par.omega_tilde = (1-par.tau)*par.omega

        # d. set parameters for question 5
        par.sigma = 1.001
        par.rho = 1.001
        par.epsilon = 1.0

    def L_opt(self):
        """ find L* """
        
        # a. set par
        par = self.par

        # b. return L*
        L =(-par.kappa + np.sqrt((par.kappa**2) + 4*par.alpha/par.nu*(par.omega_tilde**2)))/(2*par.omega_tilde)
        return L

    def V_equation(self, tau, L=None, G=None, out=1):
        """ solve for V given L and G """

        # a. set par and sim 
        par = self.par


        # b. set default values of parameters 
        if L == None:
            L = self.L_opt()

        if G == None:
            G = tau * par.omega * L

        # c. contraint
        C = par.kappa + (1 - tau)*par.omega*L

        # d. if statement for the 2 different utility functions
        if out==1:
            # i. set V LHS and RHS
            V_util = np.log((C**(par.alpha)) * (G**(1-par.alpha))) 
            V_disutil = par.nu*(L**2)/2
        
        if out==2:   
            # i. set V LHS and RHS
            V_util = ((((par.alpha * (C**((par.sigma - 1) / par.sigma))) + (1 - par.alpha) * (G**((par.sigma - 1) / par.sigma)))**(par.sigma / (par.sigma - 1)))**(1 - par.rho) - 1) / (1 - par.rho)
            V_disutil = par.nu * (L ** (1 + par.epsilon)) / (1 + par.epsilon)

        # e. return V
        return V_util - V_disutil
    

    def L_solve(self, print_output = True):
        """ solve for optimal L """

        # a. create starting values for G
        g_vec = np.linspace(1, 2, 2)

        # b. guess for G 
        G_guess = 1

        # c. find L* 
        L = self.L_opt().round(4)

        # d. print L* 
        if print_output:
            print(f'L* = {L}')

        # e. loop through values of G
        for g in g_vec:

            # i. define objective function (to maximize)
            def objective(x):
                return -self.V_equation(x, g)

            # ii. objective function
            obj = lambda x: objective(x)

            # iii. optimize L
            result = optimize.minimize(obj, G_guess, method='Nelder-Mead', bounds=[(0, 24)])

            # iv. print output
            if print_output:
                print(f'For G  = {g}, L  = {result.x[0].round(4)}')
            

    def tau_solve(self,L = None, G = None, out_V = 1):
        """ solve for optimal tau """

        # a. guess for tau
        tau_guess = 0.5


        # b. define objective function (to maximize)
        def objective(x):
            # i. if statement if default 
            if G == None and L == None:
                return -self.V_equation(tau = x, out=out_V)
            
            return -self.V_equation(tau = x, L = L, G = G, out = out_V)
        

        # c. objective function
        obj = lambda x: objective(x)

        # d. optimize tau
        result = optimize.minimize(obj, tau_guess, method='Nelder-Mead', bounds=[(0, 1)])
        optimal_tau = result.x[0] #.round(4)

        # e. return optimal tau 
        return optimal_tau


    # def V_equation_tau(self, t):
    #     """ solve for V """

    #     # a. set par and sim 
    #     par = self.par

    #     # b. set tau equal to input t
    #     par.tau = t

    #     # c. find L*
    #     L = self.L_opt()

    #     # d. find C (contraint)
    #     C = par.kappa + (1 - par.tau) * par.omega * L

    #     # e. find G
    #     G = par.tau * par.omega * L

    #     # f. find V utility and disutility
    #     V_util = np.log((C**(par.alpha)) * (G**(1-par.alpha))) 
    #     V_disutil = par.nu*(L**2)/2

    #     # g. return V
    #     return V_util - V_disutil





# # Question 5 and 6

#     def V_equation_gen(self, L, t, G):

#         # a. set par 
#         par = self.par

#         # b. set tau to input t?? 
#         par.tau = t

#         # # c. find L*
#         # L = self.L_opt()

#         # d. find C (contraint)
#         C = par.kappa + (1 - par.tau) * par.omega * L

#         # e. find V utility and disutility
#         V_util = ((((par.alpha * (C**((par.sigma - 1) / par.sigma))) + (1 - par.alpha) * (G**((par.sigma - 1) / par.sigma)))**(par.sigma / (par.sigma - 1)))**(1 - par.rho) - 1) / (1 - par.rho)
#         V_disutil = par.nu * (L ** (1 + par.epsilon)) / (1 + par.epsilon)


#         # f. return V
#         return V_util - V_disutil


    # def G_solve(self):

    #     par = self.par

    #     G_guess = 8

    #     # Define objective function (to maximize)
    #     def objective(x):
    #         return -self.V_equation_gen(par.tau, x)

    #     # Objective function
    #     obj = lambda x: objective(x)

    #     # Optimize tau
    #     result = optimize.minimize(obj, G_guess, method='L-BFGS-B')

    #     return result.x[0].round(4)
    
    # def tau_solve_G(self):

    #     par = self.par

    #     tau_guess = 0.5

    #     G = self.G_solve()

    #     # Define objective function (to maximize)
    #     def objective(x):
    #         return -self.V_equation_gen(x, G)

    #     # Objective function
    #     obj = lambda x: objective(x)

    #     # Optimize tau
    #     result = optimize.minimize(obj, tau_guess, method='Nelder-Mead', bounds=[(0, 1)])

    #     return result.x[0].round(4)

            






