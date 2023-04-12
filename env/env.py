import numpy as np
from copy import deepcopy
from utils import f_Theta

import pdb
class BS1(object):
    def __init__(self,K=5):
        self.K = K
        self.a1 = np.random.rand(K).astype(np.float128)*10000
        self.a2 = np.random.uniform(low=2.0, high=3.0, size=(K)).astype(np.float128)
        self.a3 = np.random.uniform(low=0.9, high=1.0, size=(K)).astype(np.float128)
        self.g = np.random.rand(K).astype(np.float128)*0.0001
        #### Predefined
        # self.a1 = np.array([5.1940, 5.6421, 6.3067, 7.3235, 7.3418], dtype=np.float128)*10000
        # self.a2 = np.array([2.2633, 2.6584, 2.6519, 2.4226, 2.5756], dtype=np.float128)
        # self.a3 = np.array([0.9159, 0.9425, 0.9894, 0.9777, 0.9935], dtype=np.float128)
        # self.g = np.array([0.6592, 0.0454, 0.8506, 0.9347, 0.6819], dtype=np.float128)*0.0001
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
        self.c = np.zeros((self.K), dtype=np.float128) + 200
        # Initialize Communication Time
        self.t = np.zeros((self.K), dtype=np.float128)
        
    def TD_allocate(self,):
        c = np.zeros((self.K), dtype=np.float128) + 200
        c_1= np.zeros((self.K), dtype=np.float128) #c_1 denote c + 1
        M = (self.T_m - self.E_m/self.P_c)/((1-self.P_s/self.P_c)*self.T_0) #M denote the mid cycles numbers
        y_max_1 = np.floor(self.T_m/self.T_0).astype(np.int64)
        y_max_2 = np.floor(self.E_m/(self.T_0*self.P_s)).astype(np.int64)
        #
        out = np.zeros((np.floor(self.T_m/self.T_0).astype(np.int64)))
        x = np.zeros((np.floor(self.T_m/self.T_0).astype(np.int64),self.K)) + 200 #x is 
        #sensing cycles allocation, y is total cycles
        for y in list(range(200*self.K+1, min(y_max_1,y_max_2)+1)): #y的取值是因为每个目标都至少分配了200轮
            #y_1 = y - 200*K  #y_1 is index of 'out'
            c_1 = c+ 1  #c_1
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
        self.c = deepcopy(opt_c_value)
        self.t = np.array([t_k]*self.K)
        opt_rate = self.effective_rate()

        print("opt_rate: ",opt_rate)
        
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
        sigma = 0.01
        R_est = duty_factor/(2*T) * np.log2(1 + 4*(np.pi**2)*(sigma**2)*(self.B**2)*T*self.g*self.P_s/self.delta)
        return R_est
    
    def update_channel(self,):
        self.g = np.random.rand(self.K).astype(np.float128)*0.0001
        
class UE(object):
    def __init__(self, id, N_r=1, N_t=4, is_s=0, is_c=0):
        self.id = id
        self.N_r = N_r
        self.N_t = N_t
        self.h = np.random.rand(self.N_r,self.N_t).astype(np.float64)*0.0001 #
        u_,si,v_h = np.linalg.svd(self.h) # receiver
        u_h = u_.conjugate().transpose()
        v = np.expand_dims(v_h[0,:], axis=0).conjugate().transpose()
        self.h_norm = np.linalg.norm(self.h)
        self.u = u_h
        self.v = v
        self.P_n = np.random.rand()*0.0001 # noise power
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

    def update(self,):
        self.h = np.random.rand(self.N_r,self.N_t).astype(np.float64)*0.0001 # 
        u_,si,v_h = np.linalg.svd(self.h) # receiver
        u_h = u_.conjugate().transpose()
        v = np.expand_dims(v_h[0,:], axis=0).conjugate().transpose()
        self.u = u_h
        self.v = v
        self.h_norm = np.linalg.norm(self.h)

        

class BS(object):
    def __init__(self, N_t=4, N_c=5, N_s=2):
        self.c_UE = [UE(id="c"+str(i),N_t=N_t,is_c=1) for i in range(N_c)]  
        # Sort to H norm descending     
        self.c_UE.sort(key=lambda x : x.h_norm, reverse = True)
        self.s_UE = [UE(id="s"+str(i),N_t=N_t,is_s=1) for i in range(N_s)] 
        self.B_c = 10e+6 # Hz Communication Bandwidth
        self.B_s = 10e+6 # Hz Sensing Bandwidth
        # pdb.set_trace()

    def _cal_noma(self, update = True):
        Sinr = []
        R_c = []
        for _id in range(len(self.c_UE)):
            UE = self.c_UE[_id]
            UE_in = self.c_UE[0:_id]
            P_si = np.linalg.norm(UE.H)*UE.P_c
            P_in = 0.0
            if len(UE_in) > 0:
                P_in = np.linalg.norm(UE.H)*sum([UE_.P_c for UE_ in UE_in])
            Sinr.append(P_si/(P_in+UE.P_n))
            R_c.append(UE.B_c * np.log2(1+UE.Sinr))
            if update:
                UE.Sinr = P_si/(P_in+UE.P_n)
                UE.R_c = UE.B_c * np.log2(1+UE.Sinr)   
        return Sinr,R_c  

    def _cal_radar(self, update=True):   
        ...
        
    def update(self,):
        # Update Communication UE Channel
        for UE in self.c_UE:
            UE.update()
        # Resort NOMA Index
        self.c_UE.sort(key=lambda x : x.h_norm, reverse = True)
        # Update Sensing UE Channel
        for UE in self.s_UE:
            UE.update()
        

if __name__ == "__main__":
    np.random.seed(777)
    env = BS()
    # env.TD_allocate()
    # print("a1:",env.a1)
    # print("a2:",env.a2)
    # print("a3:",env.a3)
    # print("g:",env.g)
    # print("c:",env.c)
    # print("t:",env.t)
    # print("c_rate:",env.communication_rate())
    # print("s_acc:",env.sensing_accuracy())
    # print("RER:",env.radar_estimation_rate())
    