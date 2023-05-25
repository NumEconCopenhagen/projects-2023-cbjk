import numpy as np
from types import SimpleNamespace

class labor:
    
    def __init__(self,**kwargs): # called when created
        """
        Initialize the labor class.
        """
        par = self.par = SimpleNamespace()
        
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
    
    def H(self, out=1, Delta=0.05, print_out = 0):
        """
        Calculate the value of H based on given settings.
        out=1 is the original policy
        out=2 is the modified policy
        out=3 is our own policy
        """
        # a. set starting values and par
        par = self.par
        H0 = 0

        # b. iterate over the random shock series
        for i in range(self.K):
            kappa = np.empty(0)
            L = np.empty(0)
            log_kappa_t0 = np.log(self.kappa_0)
            np.random.seed(i)
            epsilon = np.random.normal(-0.5*par.sigma_e**2, par.sigma_e, len(par.t))
            # i. calculate kappa for each epsilon
            for e in epsilon:
                kap0 = np.exp(log_kappa_t0)
                kappa = np.append(kappa,kap0)
                log_kappa_t = par.rho*log_kappa_t0 + e
                log_kappa_t0 = log_kappa_t.copy()
            
            l_0 = self.l_0  # make sure that the starting value is the same for each iteration
            
            # ii. calculate l for a given policy

            if out == 1: #original policy

                for kap in kappa:
                    L = np.append(L,l_0)
                    l_0 = self.l_star(kap)
            
            if out == 2: #modified policy
                for kap in kappa:
                    L = np.append(L,l_0)
                    l_star = self.l_star(kap)
                    # o. change l_star if difference larger than Delta
                    if abs(l_0-l_star) > Delta: 
                        l_0 = l_star
            if out == 3: #our policy
                for t, kap in enumerate(kappa):
                    L = np.append(L,l_0)
                    l_star = self.l_star(kap)
                    # oo. change l_star if epsilon larger than the variance
                    if abs(epsilon[t]) < par.sigma_e:
                        l_0 = l_star

            # iii. calculate h given l and kappa            
            h = 0 # make sure that the starting value is the same for each iteration
            for t in range(120):
                if t==0:
                    L[t-1] = self.l_0

                if L[t] == L[t-1]:
                    h += par.R**(-t)*(kappa[t]*(L[t]**(1-par.eta))-par.w*L[t])
                else:
                    h += par.R**(-t)*(kappa[t]*(L[t]**(1-par.eta))-par.w*L[t]-par.iota)
            H0 += h

        # iiii. find H by taken mean of each iteration           
        H = (H0/self.K) #
        if print_out == 1:
            return print(f'For policy {out}: H = {H.round(4)}')
        else:
            return H


    def l_star(self, kappa):
        """
        Calculate the value of l_star based on given kappa.
        """
        # a. set par
        par = self.par

        # b. calculate l 
        l_star = (((1-par.eta)*kappa)/par.w)**(1/par.eta)
        return l_star