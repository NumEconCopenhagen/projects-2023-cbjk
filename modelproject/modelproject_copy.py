from types import SimpleNamespace
from scipy import optimize
import numpy as np

class OLG_CIE_Class():

    def __init__(self,do_print=True):
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.ss = SimpleNamespace()   # What is this?
        self.path = SimpleNamespace() # What is this?

        if do_print: print('calling .setup()')
        self.setup()

    def setup(self):
        """ baseline parameters """

        par = self.par

        par.rho = 0.19  
        par.alpha = 0.25 
        par.t = 2
        par.N_t = 5.93 # people on earth in 2000
        par.growth_rate = 0.011
        par.E_t = 8
        
        
    # a. Model Structure 

    def u_t(self, c_yt, c_ot1):
        """ OLG utility  """
        par = self.par
        return np.log(c_yt) + 1/(1+par.rho)*np.log(c_ot1)
    
    def c_yt(k_t1, w_t, pi_yt):
        """ Young consumption  """
        return w_t + pi_yt - k_t1
    
    def c_ot1(w_t1, r_t1, k_t1, pi_ot1):
        """ Old consumption  """
        return w_t1 + (1+r_t1)* k_t1 + pi_ot1
    
    def euler_error(self, r_t1, c_yt, c_ot1):
        """ The euler error """
        par = self.par
        return c_yt/c_ot1 - (1+r_t1)/(1+par.rho)
    
    def N_total(self):
        par = self.par
        N_total_list = []
        for i in range(par.t):
            N_totals = par.N_t * (1 + par.growth_rate) ** i
            N_total_list.append(N_totals)
        return N_total_list

    
    def C_t(n, c_yt, c_ot1):
        """ Total consumption """
        return n*c_yt + n.shift(1)*c_ot1
    
    def K_t(n, k_t):
        """ Total capital stock or the summed asset holdings of all old individuals """
        return n.shift(1)*k_t
    
    def k_t(n, K_t):
        return K_t/n.shift(1)
    
    def N_t(n):
        """ Total population """
        return n.shift(1) + n
    
    def E_t(C_t, K_t1, K_t, Y_t):
        return (C_t + K_t1) - (Y_t + K_t)
    
    def bc(N_yt, v_t, E_t):
        """ Budget contraint """
        pi_guess = [0.3, 0.4]
        return (pi_guess[0]*N_yt + pi_guess[1]*N_yt.shift(1) - (v_t*E_t))




    # b. Model Calibration

    def N_yt(self):
        """ The number of young people alive at each date in billions """
        par = self.par
        n_y = []
        for i in range(par.t):
            N_yt = 5.27 - 1.42*0.587**i
            n_y.append(N_yt)

        return n_y
    
    def Y_t(self, A_t, K_t, N_t):
        """ Gross world output in trillions """
        par = self.par
        return A_t*K_t**par.alpha*N_t**(1-par.alpha)
    
    def r_t(self, A_t, K_t, N_t):
        par = self.par
        return par.alpha*A_t*K_t**(par.alpha-1)*N_t**(1-par.alpha)

    def w_t(self, A_t, K_t, N_t):
        par = self.par
        return (1-par.alpha)*A_t*K_t**par.alpha * N_t**(1-par.alpha-1)

    def E_0t(self, Y_t):
        """ Potential emissions in billion tons of carbon equivalent pr. period """
        par = self.par
        E_0t_list = []
        for i in range(par.t):
            E_0t = (0.181 + 0.189*0.622**i)*Y_t
            E_0t_list.append(E_0t)
        E_0t = E_0t_list.copy()
        return E_0t
    
    def A_t(self, T_t, E_t, E_0t):
        """ Total factor productivity """
        par = self.par
        A_t_list = []
        for i in range(par.t):
            A_t = (235 - 142*0.739**i)*(1-0.0133*(T_t/3)**2)*(1-0.0686*(1-E_t/E_0t)**2.89)
            A_t_list.append(A_t)
        A_t = A_t_list.copy()
        return A_t
    
    def K_t1(K_t, Y_t, C_t):
        """ Net capital investment """
        return ((1-0.1)**35)*K_t + Y_t - C_t

    def T_t(Q_t1, F_t):
        """ The chnage in mean global temperature relative to the preindistrual form in °C """
        return (5.92*np.log(Q_t1.shift(1)/590) + F_t)/1.41
    
    def F_t(t):
        """ 'Rediative forcing' caused by trace concentrations of methane, nitrous oxide , and water vapor in watts/m^2 """
        F_t_list = []
        for i in range(t):
            F_t = 1.42 - 0.764*0.523**i
            F_t_list.append(F_t)
        F_t = F_t_list.copy()
        return F_t
    
    def Q_t1(self, E_t):
        """ The atmospheric stock of carbon dioxide and chlorofluorocarbons """
        par = self.par
        Q_t1 = [784]  # initialize Q_t1 with a value of 784
        Q_t = Q_t1[0]
        for i in range(1, par.t):
            Q_t1_i = 0.64 * E_t + ((1 - 0.00833) ** 35) * (Q_t - 590) + 590
            Q_t1.append(Q_t1_i)
            Q_t = Q_t1_i
        return Q_t1
    
    def v_t(E_t, E_0t):
        """  """
        return (0.0686*2.89(E_t/E_0t))/E_0t


    # c. Numerical Simulations

    def no_transfers_baseline(v_t, E_t, N_t):
        """ 
        Emissions taxes are chosen according to equation (9) to achieve 
        an efficient allocation of resources, while tax revenues are 
        distributed in equal lump sums to all living persons
        """
        pi_y, pi_o = [v_t*E_t/N_t, v_t*E_t/N_t]
        bc(pi_y, pi_o) 


        return 






    




    



 

