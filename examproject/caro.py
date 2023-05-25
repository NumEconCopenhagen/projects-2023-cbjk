from random import seed
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from types import SimpleNamespace

class labor:
    
    def __init__(self,**kwargs): # called when created

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()
        
        # a. baseline parameters
        par.eta = 0.5
        par.w = 1.0
        par.rho = 0.9
        par.iota = 0.01
        par.sigma_e = 0.1
        par.R = (1+0.01)**(1/12)
        par.t = range(120)


        
        # b. initial values
        self.l_0 = 0
        self.kappa_0 = 1
        self.K = 100
        
            
        # c. update parameters and settings
        for key, value in kwargs.items():
            setattr(self,key,value) # like self.key = value
        
         # note: "kwargs" is a dictionary with keyword arguments
    
    def H(self, out=1, Delta=0.05):
        
        par = self.par
        H0 = 0


        for i in range(self.K):
            kappa = np.empty(0)
            L = np.empty(0)
            log_kappa_t0 = np.log(self.kappa_0)
            np.random.seed(i)
            epsilon = np.random.normal(-0.5*par.sigma_e**2, par.sigma_e, len(par.t))
            for e in epsilon:
                kap0 = np.exp(log_kappa_t0)
                kappa = np.append(kappa,kap0)
                log_kappa_t = par.rho*log_kappa_t0 + e
                log_kappa_t0 = log_kappa_t.copy()
            
            l_0 = self.l_0 
            if out == 1:

                for kap in kappa:
                    L = np.append(L,l_0)
                    l_0 = self.l_star(kap)
            
            if out == 2:
                for kap in kappa:
                    L = np.append(L,l_0)
                    l_star = self.l_star(kap)
                    if abs(l_0-l_star) > Delta:
                        l_0 = l_star
            if out == 3:
                for t, kap in enumerate(kappa):
                    L = np.append(L,l_0)
                    l_star = self.l_star(kap)
                    if abs(epsilon[t]) > par.sigma_e:
                        l_0 = l_star
            h = 0
            for t in range(120):
                if t==0:
                    L[t-1] = self.l_0

                if L[t] == L[t-1]:
                    h += par.R**(-t)*(kappa[t]*(L[t]**(1-par.eta))-par.w*L[t])
                else:
                    h += par.R**(-t)*(kappa[t]*(L[t]**(1-par.eta))-par.w*L[t]-par.iota)
            H0 += h
                   
        H = (H0/self.K)
        return H


    def l_star(self, kappa):
        par = self.par
        l_star = (((1-par.eta)*kappa)/par.w)**(1/par.eta)
        return l_star