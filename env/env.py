import numpy as np
from copy import deepcopy
# from utils import f_Theta

def f_Theta(a1,a2,a3,c):
    Theta = a3 - a1*1/(np.power(c, a2))
    return Theta

import pdb
class BS_TD(object):
    def __init__(self,logger,K=5,seed=777):
        self.K = K
        self.logger=logger
        self.seed=seed
        np.random.seed(seed)
        self.a1 = np.random.rand(K).astype(np.float64)*10000
        self.a2 = np.random.uniform(low=2.0, high=3.0, size=(K)).astype(np.float64)
        self.a3 = np.random.uniform(low=0.9, high=1.0, size=(K)).astype(np.float64)
        self.g = np.random.rand(K).astype(np.float64)*0.0001
        #### Predefined
        # self.a1 = np.array([5.1940, 5.6421, 6.3067, 7.3235, 7.3418], dtype=np.float64)*10000
        # self.a2 = np.array([2.2633, 2.6584, 2.6519, 2.4226, 2.5756], dtype=np.float64)
        # self.a3 = np.array([0.9159, 0.9425, 0.9894, 0.9777, 0.9935], dtype=np.float64)
        # self.g = np.array([0.6592, 0.0454, 0.8506, 0.9347, 0.6819], dtype=np.float64)*0.0001
        # Communication & Sensing Configuration
        self.P_c = 0.5
        self.P_s = 1.0
        # Time & Energy Constraint
        self.T_m = 3  #total time buget (s)
        self.T_0 = 6e-5   #T_0 = 0.00006  #a slot of each sensing cycles time (s), note that T_0=T_s
        self.E_m = 2.5  #total power buget (J)
        self.B = 10e+6 #total bandwidth (Hz)
        self.delta = 1e-10 # background Noise Power
        # Initialize Sensing Cycle
        self.c = np.zeros((self.K), dtype=np.float64) + 200
        # Initialize Communication Time
        self.t = np.zeros((self.K), dtype=np.float64)
        ##
        y_max_1 = np.floor(self.T_m/self.T_0).astype(np.int64)
        y_max_2 = np.floor(self.E_m/(self.T_0*self.P_s)).astype(np.int64)
        self.action_space = list(range(200*self.K+1, min(y_max_1,y_max_2)+1, 100))
        self.reward = 0.0
        
    def TD_allocate(self,update=False):
        c = np.zeros((self.K), dtype=np.float64) + 200
        c_1= np.zeros((self.K), dtype=np.float64) #c_1 denote c + 1
        M = (self.T_m - self.E_m/self.P_c)/((1-self.P_s/self.P_c)*self.T_0) #M denote the mid cycles numbers
        y_max_1 = np.floor(self.T_m/self.T_0).astype(np.int64)
        y_max_2 = np.floor(self.E_m/(self.T_0*self.P_s)).astype(np.int64)
        #
        out = np.zeros((np.floor(self.T_m/self.T_0).astype(np.int64)))
        x = np.zeros((np.floor(self.T_m/self.T_0).astype(np.int64),self.K)) + 200 #x is 
        #sensing cycles allocation, y is total cycles
        for y in list(range(200*self.K+1, min(y_max_1,y_max_2)+1)): #y的取值是因为每个目标都至少分配了200轮
            #y_1 = y - 200*K  #y_1 is index of 'out'
            c_1 = c + 1  #c_1
            Theta_div = f_Theta(self.a1,self.a2,self.a3,c_1) - f_Theta(self.a1,self.a2,self.a3,c)   #denote the "gradient" of Theta
            max_ind = np.argmax(Theta_div, axis=-1)  #find k^
            max_val = Theta_div[max_ind]
            c[max_ind] = c[max_ind] + 1 
            if (1-self.P_s/self.P_c)*self.T_0 > 0: #判断正负，如果正执行后者E_m/P_c-P_s/P_c*T_0*y
                if y <= M:  
                    out[y] = np.sum(np.log(f_Theta(self.a1,self.a2,self.a3,c))) +self.K*np.log(self.E_m/self.P_c-self.P_s/self.P_c*self.T_0*y)  # out denote the output value if objective function
                else: #执行前者
                    out[y] = np.sum(np.log(f_Theta(self.a1,self.a2,self.a3,c))) +self.K*np.log(self.T_m-self.T_0*y) 
            else:
                if y <=M: #判断正负，如果负执行前者者T_m-T_0*y
                    out[y] = np.sum(np.log(f_Theta(self.a1,self.a2,self.a3,c))) +self.K*np.log(self.T_m-self.T_0*y) 
                else: #执行后者
                    out[y] = np.sum(np.log(f_Theta(self.a1,self.a2,self.a3,c))) +self.K*np.log(self.E_m/self.P_c-self.P_s/self.P_c*self.T_0*y)  # out denote the output value if objective function
            x[y,max_ind] = c[max_ind] 
        #
        opt_y_value = np.argmax(out)  #find object y which makes object max
        max_out = out[opt_y_value]
        print(opt_y_value)
        opt_c_value = np.max(x[0:opt_y_value,:],axis=0)  #the optimal varialble value (c_1,...c_5)
        t_k = 1/self.K * min(self.T_m-self.T_0*opt_y_value, self.E_m/self.P_c-self.P_s/self.P_c*self.T_0*opt_y_value)
        #
        if update:
            self.c = deepcopy(opt_c_value)
            self.t = np.array([t_k]*self.K)
            opt_rate = self.effective_rate()
            print("opt_rate: ",opt_rate)
        else:
            return opt_y_value
        
    def given_y_get_c(self,y_max):
        c = np.zeros((self.K), dtype=np.float64) + 200
        c_1= np.zeros((self.K), dtype=np.float64) #c_1 denote c + 1
        M = (self.T_m - self.E_m/self.P_c)/((1-self.P_s/self.P_c)*self.T_0) #M denote the mid cycles numbers
        y_max_1 = np.floor(self.T_m/self.T_0).astype(np.int64)
        y_max_2 = np.floor(self.E_m/(self.T_0*self.P_s)).astype(np.int64)
        #
        out = 0.0
        x = np.zeros((1,self.K)) + 200 #x is 
        #sensing cycles allocation, y is total cycles
        for y in list(range(200*self.K+1, y_max+1)): #y的取值是因为每个目标都至少分配了200轮
            c_1 = c + 1  #c_1
            Theta_div = f_Theta(self.a1,self.a2,self.a3,c_1) - f_Theta(self.a1,self.a2,self.a3,c)   #denote the "gradient" of Theta
            max_ind = np.argmax(Theta_div, axis=-1)  #find k^
            max_val = Theta_div[max_ind]
            c[max_ind] = c[max_ind] + 1 
            x[:,max_ind] = c[max_ind] 
        #
        defined_y = y_max
        opt_c_value = x  #the optimal varialble value (c_1,...c_5) under given y
        t_k = 1/self.K * min(self.T_m-self.T_0*defined_y, self.E_m/self.P_c-self.P_s/self.P_c*self.T_0*defined_y)
        #
        self.c = deepcopy(opt_c_value)
        self.t = np.array([t_k]*self.K)
        opt_rate = self.effective_rate()
        # print("opt_rate: ",opt_rate)
        self.reward = opt_rate/100.0
        return self.c
        
        
    def effective_rate(self,):
        opt_rate = np.sum(np.log(f_Theta(self.a1,self.a2,self.a3,self.c)*(self.t/self.T_m * self.B *np.log2(1+self.g*self.P_c/self.delta))))
        return opt_rate
        
    def communication_rate(self,):
        _rate = self.t/self.T_m * self.B *np.log2(1+self.g*self.P_c/self.delta)
        return _rate
    
    def sensing_accuracy(self,):
        _acc = f_Theta(self.a1,self.a2,self.a3,self.c)
        return _acc
    
    def radar_estimation_rate(self,):
        duty_factor = 0.1 # Pulse Duration Interval / Pulse Repetition Interval
        T = 0.000001 # 1us
        sigma = 0.01 # Processing Noise
        R_est = duty_factor/(2*T) * np.log2(1 + 4*(np.pi**2)*(sigma**2)*(self.B**2)*T*self.g*self.P_s/self.delta)
        return R_est
    
    def update_channel(self,):
        self.g = np.random.rand(self.K).astype(np.float64)*0.0001
        
    def get_state_batch(self, action_id, batch=1):
        # pdb.set_trace()
        print("action_id",action_id)
        if isinstance(action_id, np.ndarray):
            action_id = action_id[0]
        action = self.action_space[action_id]
        A1 = np.expand_dims(self.a1, axis=0)
        A2 = np.expand_dims(self.a2, axis=0)
        A3 = np.expand_dims(self.a3, axis=0)
        C = self.given_y_get_c(y_max=action)
        # pdb.set_trace()
        # print(A1.shape)
        return [A1,A2,A3,C]
        
        


class UE(object):
    def __init__(self, id, N_r=1, N_t=16, is_s=0, is_c=0, theta=np.random.uniform()*np.pi):
        self.id = id
        self.N_r = N_r
        self.N_t = N_t
        # 
        self.d = np.random.rand()*2+1.0 # m
        self.theta = theta# angle 0~pi
        self.g = (3*10e8/(4*np.pi*self.d*10*10e9))**2*1000# PathLoss
        self.at = np.expand_dims(np.exp(1j * np.arange(start=0, stop=self.N_t , step=1, dtype=np.complex128) * np.pi *np.sin(self.theta)), axis=-1)
        self.ar = np.expand_dims(np.exp(1j * np.arange(start=0, stop=self.N_r , step=1, dtype=np.complex128) * np.pi *np.sin(self.theta)), axis=-1)
        #
        self.h = self.ar@self.at.conjugate().transpose()    #
        # self.h = np.random.rand(self.N_r,self.N_t).astype(np.float64)    #
        self.h_r = self.at@self.at.conjugate().transpose()
        u_,si,v_h = np.linalg.svd(self.h_r) # receiver
        u_h = u_.conjugate().transpose()
        v = np.expand_dims(v_h[0,:], axis=0).conjugate().transpose()
        # pdb.set_trace()
        self.h_norm = np.linalg.norm(self.h)
        self.u = u_h
        self.v = 1/np.sqrt(self.N_t)*deepcopy(self.at)
        #
        self.P_n = 1e-8 # noise power
        self.P_c = 0.0 # Downlink Communication power
        self.P_s = 0.0 # Downlink Sensing power
        self.is_s = is_s 
        self.is_c = is_c 
        # pdb.set_trace()
        # State:
        # NOMA Sinr
        self.Sinr = 0.0 # db
        # NOMA Rate
        self.R_c = 0.0 # bps
        print("range:", self.d)
        print("g:", self.g)

    def update(self,):
        # 
        self.d = np.random.rand()*5 # m
        self.theta = np.random.rand()*np.pi# angle 0~pi
        self.g = (3*10e8/(4*np.pi*self.d*10*10e9))**2# PathLoss
        self.at = np.expand_dims(np.exp(1j * np.arange(start=0, stop=self.N_t , step=1, dtype=np.complex128) * np.pi *np.sin(self.theta)), axis=-1)
        self.ar = np.expand_dims(np.exp(1j * np.arange(start=0, stop=self.N_r , step=1, dtype=np.complex128) * np.pi *np.sin(self.theta)), axis=-1)
        #
        # pdb.set_trace()
        self.h = self.ar@self.at.conjugate().transpose()    #
        # self.h = np.random.rand(self.N_r,self.N_t).astype(np.float64)    #
        self.h_r = self.at@self.at.conjugate().transpose()
        u_,si,v_h = np.linalg.svd(self.h_r) # receiver
        u_h = u_.conjugate().transpose()
        v = np.expand_dims(v_h[0,:], axis=0).conjugate().transpose()
        self.h_norm = np.linalg.norm(self.h)
        self.u = u_h
        # self.v = v
        self.v = 1/np.sqrt(self.N_t)*deepcopy(self.at)
        
        

class BS(object):
    def __init__(self, N_t=16, N_c=5, N_s=2):
        self.c_UE = [UE(id="c"+str(i),N_t=N_t,is_c=1,theta=0.5*np.pi+0.1*i*np.pi) for i in range(N_c)]  
        # Sort to H norm descending     
        self.c_UE.sort(key=lambda x : x.g*x.h_norm, reverse = True)
        self.s_UE = [UE(id="s"+str(i),N_t=N_t,is_s=1,theta=0.2*i*np.pi) for i in range(N_s)] 
        self.B_c = 10e+6 # Hz Communication Bandwidth
        self.B_s = 10e+6 # Hz Sensing Bandwidth
        self.P_tot = 500 # W
        self.P_n = 1e-5 # noise power
        self.N_t = N_t
        self.N_c = N_c
        self.N_s = N_s
        
        # pdb.set_trace()

    def _cal_noma(self, update = True):
        Sinr = []
        R_c = []
        for _id in range(len(self.c_UE)):
            UE = self.c_UE[_id]
            UE_in = self.c_UE[0:_id]
            P_si = UE.g*UE.h_norm**2*UE.P_c
            P_in = 0.0
            if len(UE_in) > 0:
                P_in = UE.g*UE.h_norm**2*sum([UE_.P_c for UE_ in UE_in])
            Sinr.append((P_si/(P_in+UE.P_n)))
            R_c.append(self.B_c * np.log2(1+P_si/(P_in+UE.P_n)))
            if update:
                UE.Sinr = P_si/(P_in+UE.P_n)
                UE.R_c = self.B_c * np.log2(1+UE.Sinr)   
        return Sinr,R_c  

    def _cal_radar(self, update=True):   
        R_est = []
        # Radar Processing Params 
        duty_factor = 0.1 # Pulse Duration Interval / Pulse Repetition Interval
        T = 0.000001 # 1us
        sigma = 0.01 # Processing Noise
        SUM_COMA = np.identity(self.N_t,dtype=np.complex128)*self.P_n
        for _id in range(len(self.s_UE)):
            UE = self.s_UE[_id]
            SUM_COMA += UE.P_s**2*UE.g**2*UE.h_r@UE.v@(UE.v.transpose().conjugate())@(UE.h_r.transpose().conjugate())
            # pdb.set_trace()
        for _id in range(len(self.s_UE)):
            UE = self.s_UE[_id]
            UE_ = self.s_UE[0:_id]+self.s_UE[_id+1:]
            P_in = 0 # Total inference signal power (without noise)
            COMA = SUM_COMA - UE.P_s**2*UE.g**2*UE.h_r@UE.v@(UE.v.transpose().conjugate())@(UE.h_r.transpose().conjugate())
            UE.u = np.linalg.inv(COMA)@UE.h_r@UE.v@np.linalg.inv(UE.v.transpose().conjugate()@UE.h_r.transpose().conjugate()@np.linalg.inv(COMA)@UE.h_r@UE.v)
            UE.u = UE.u.transpose().conjugate()
            # pdb.set_trace()
            for UE_in in UE_:
                P_in += (UE_in.g)**2*np.linalg.norm(UE.u@UE_in.h_r@UE.v)**2*UE_in.P_s**2
            P_useful = (UE.g)**2*np.linalg.norm(UE.u@UE.h_r@UE.v)**2*UE.P_s**2
            R_ = duty_factor/(2*T) * np.log2(1 + 4*(np.pi**2)*(sigma**2)*(self.B_s**2)*T*P_useful/(self.P_n*np.linalg.norm(UE.u)**2+P_in))
            R_est.append(R_)
            print("RSINR:",(P_useful/(self.P_n*np.linalg.norm(UE.u)**2+P_in)) )
            print("RSINR_MDVR:",np.abs((UE.g)**2*UE.P_s**2*(UE.v.transpose().conjugate()@UE.h_r.transpose().conjugate()@np.linalg.inv(COMA)@UE.h_r@UE.v)) )
            print("Pin: ",P_in)
            print("P_useful: ",P_useful)
            print("I+R: ",(self.P_n*np.linalg.norm(UE.u)**2+P_in))
            print("UCOMAUH: ",np.abs(UE.u@COMA@UE.u.conjugate().transpose()))
            a,b,c=np.linalg.svd(COMA)
            # pdb.set_trace()
        return R_est
        
    def update_channels(self,):
        # Update Communication UE Channel
        for UE in self.c_UE:
            UE.update()
        # Re-sort NOMA Index
        self.c_UE.sort(key=lambda x : x.g*x.h_norm, reverse = True)
        # Update Sensing UE Channel
        for UE in self.s_UE:
            UE.update()

    def get_performance(self,):
        Sinr, R_c = self._cal_noma()
        R_est = self._cal_radar()
        print("Sinr (dB)", 10*np.log(Sinr))
        print("R_c", R_c)
        print("R_est", R_est)
        return R_c, R_est

    def _power_allocation(self, Pc_s, Ps_s):
        for _id in range(len(self.s_UE)):
            UE = self.s_UE[_id]
            UE.P_s = Ps_s[_id]
        for _id in range(len(self.c_UE)):
            UE = self.c_UE[_id]
            UE.P_c = Pc_s[_id]
        
    
        

if __name__ == "__main__":
    np.random.seed(777)
    # env = BS(N_t=16, N_c=5, N_s=2)
    # env._power_allocation(Pc_s = [10,20,50,90,150], Ps_s = [1,1])
    # env.get_performance()
    
    
    ###-------------------------------
    env = BS_TD(logger=None,K=5,seed=777)
    # env.given_y_get_c(y_max=1900)
    # env.TD_allocate()
    print("a1:",env.a1)
    print("a2:",env.a2)
    print("a3:",env.a3)
    print("g:",env.g)
    print("c:",env.c)
    print("t:",env.t)
    print("c_rate:",env.communication_rate())
    print("s_acc:",env.sensing_accuracy())
    print("RER:",env.radar_estimation_rate())
    print(env.get_state_batch(action=1900))
    pdb.set_trace()