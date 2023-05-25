import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy import optimize

class question_1():

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
        return (-par.kappa + np.sqrt((par.kappa**2) + 4*par.alpha/par.nu*(par.omega_tilde**2)))/(2*par.omega_tilde)


    def V_equation(self, L, G):
        """ solve for V given L and G """

        # a. set par and sim 
        par = self.par

        # b. contraint 
        C = par.kappa + (1 - par.tau)*par.omega*L

        # c. set V LHS and RHS
        V_util = np.log((C**(par.alpha)) * (G**(1-par.alpha))) 
        V_disutil = par.nu*(L**2)/2

        # d. return V
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
                print(f'G  = {g}')
                print(f'L  = {result.x[0].round(4)}')


    def V_equation_tau(self, t):
        """ solve for V """

        # a. set par and sim 
        par = self.par

        # b. set tau equal to input t
        par.tau = t

        # c. find L*
        L = self.L_opt()

        # d. find C (contraint)
        C = par.kappa + (1 - par.tau) * par.omega * L

        # e. find G
        G = par.tau * par.omega * L

        # f. find V utility and disutility
        V_util = np.log((C**(par.alpha)) * (G**(1-par.alpha))) 
        V_disutil = par.nu*(L**2)/2

        # g. return V
        return V_util - V_disutil


    def tau_solve(self):
        """ solve for optimal tau """

        # a. guess for tau
        tau_guess = 0.5

        # b. define objective function (to maximize)
        def objective(x):
            return -self.V_equation_tau(x)

        # c. objective function
        obj = lambda x: objective(x)

        # d. optimize tau
        result = optimize.minimize(obj, tau_guess, method='Nelder-Mead', bounds=[(0, 1)])

        # e. return optimal tau 
        return result.x[0].round(4)


# Question 5 and 6

    def V_equation_gen(self, t, G):

        # a. set par 
        par = self.par

        # b. set tau to input t?? 
        par.tau = t

        # c. find L*
        L = self.L_opt()

        # d. find C (contraint)
        C = par.kappa + (1 - par.tau) * par.omega * L

        # e. find V utility and disutility
        V_util = (((par.alpha * np.power(C, (par.sigma - 1) / par.sigma)) + (1 + par.alpha) * np.power(G, par.sigma / (par.sigma - 1)))) ** (1 - par.rho) - 1) / (1 - par.rho)
        V_disutil = par.nu * (L ** (1 + par.epsilon)) / (1 + par.epsilon)

        # f. return V
        return V_util - V_disutil


    def G_solve(self):

        par = self.par

        G_guess = 8

        # Define objective function (to maximize)
        def objective(x):
            return -self.V_equation_gen(par.tau, x)

        # Objective function
        obj = lambda x: objective(x)

        # Optimize tau
        result = optimize.minimize(obj, G_guess, method='L-BFGS-B')

        return result.x[0].round(4)
    
    def tau_solve_G(self):

        par = self.par

        tau_guess = 0.5

        G = self.G_solve()

        # Define objective function (to maximize)
        def objective(x):
            return -self.V_equation_gen(x, G)

        # Objective function
        obj = lambda x: objective(x)

        # Optimize tau
        result = optimize.minimize(obj, tau_guess, method='Nelder-Mead', bounds=[(0, 1)])

        return result.x[0].round(4)

            






