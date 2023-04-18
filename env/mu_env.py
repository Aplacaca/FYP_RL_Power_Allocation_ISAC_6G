import numpy as np
from copy import deepcopy
# from utils import f_Theta

def f_Theta(a1,a2,a3,c):
    Theta = a3 - a1*1/(np.power(c, a2))
    return Theta

import pdb

class UE(object):
    def __init__(self, id, N_r=1, N_t=16, is_s=0, is_c=0, theta=np.random.uniform()*np.pi):
        self.id = id
        self.N_r = N_r
        self.N_t = N_t
        # Channel Generation
        self.d = np.random.rand()*2+1.0 # range{m}
        self.theta = theta# angle 0~pi
        self.g = (3*10e8/(4*np.pi*self.d*10*10e9))**2# PathLoss of Communication
        self.g_r = (3*10e8/(4*np.pi*(self.d**2)*10*10e9))**2# PathLoss of Radar
        self.at = 1/np.sqrt(self.N_t)*np.expand_dims(np.exp(1j * np.arange(start=0, stop=self.N_t , step=1, dtype=np.complex128) * np.pi *np.sin(self.theta)), axis=-1)# Steering Vector
        self.ar = 1/np.sqrt(self.N_r)*np.expand_dims(np.exp(1j * np.arange(start=0, stop=self.N_r , step=1, dtype=np.complex128) * np.pi *np.sin(self.theta)), axis=-1)# Steering Vector
        self.h = np.sqrt(self.g)*np.sqrt(self.N_t*self.N_r)*self.ar@self.at.conjugate().transpose() # Communication Channel
        self.h_r = np.sqrt(self.g_r)*self.N_t*self.at@self.at.conjugate().transpose() # Radar Channel
        self.A = self.at@self.at.conjugate().transpose()
        self.h_norm = np.linalg.norm(self.h)
        self.v = deepcopy(self.at)
        #
        self.P_n = 1e-5 # noise power
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
        # Channel Generation
        self.d = np.random.rand()*2+1.0 # range{m}
        self.theta = np.random.uniform()*np.pi# angle 0~pi
        self.g = (3*10e8/(4*np.pi*self.d*10*10e9))**2# PathLoss of Communication
        self.g_r = (3*10e8/(4*np.pi*(self.d**2)*10*10e9))**2# PathLoss of Radar
        self.at = 1/np.sqrt(self.N_t)*np.expand_dims(np.exp(1j * np.arange(start=0, stop=self.N_t , step=1, dtype=np.complex128) * np.pi *np.sin(self.theta)), axis=-1)# Steering Vector
        self.ar = 1/np.sqrt(self.N_r)*np.expand_dims(np.exp(1j * np.arange(start=0, stop=self.N_r , step=1, dtype=np.complex128) * np.pi *np.sin(self.theta)), axis=-1)# Steering Vector
        self.h = np.sqrt(self.g)*np.sqrt(self.N_t*self.N_r)*self.ar@self.at.conjugate().transpose() # Communication Channel
        self.h_r = np.sqrt(self.g_r)*self.N_t*self.at@self.at.conjugate().transpose() # Radar Channel
        self.h_norm = np.linalg.norm(self.h)
        self.v = deepcopy(self.at)
        #
        
        

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

    def _cal_zf(self, update = True):
        Sinr = []
        R_c = []
        Full_H = np.concatenate([UE.h for UE in self.c_UE], axis=0)
        Full_V = Full_H.conjugate().transpose()@np.linalg.inv(Full_H@Full_H.conjugate().transpose())
        for _id in range(len(self.c_UE)):
            UE = self.c_UE[_id]
            UE.v = Full_V[:,_id]
        for _id in range(len(self.c_UE)):
            UE = self.c_UE[_id]
            UE_in = self.c_UE[0:_id] + self.c_UE[_id+1:]
            P_si = np.linalg.norm(UE.h@UE.v)**2*UE.P_c
            P_in = 0.0
            if len(UE_in) > 0:
                P_in += np.sum([np.linalg.norm(UE.h@UE_.v)**2*UE_.P_c for UE_ in UE_in])
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
        # Get the Covariance Matrix of Total Transmitted Signal 
        SUM_COMA = np.identity(self.N_t,dtype=np.complex128)*self.P_n
        for _id in range(len(self.s_UE)):
            UE = self.s_UE[_id]
            SUM_COMA += UE.P_s*UE.h_r@UE.v@(UE.v.transpose().conjugate())@(UE.h_r.transpose().conjugate())
        for _id in range(len(self.s_UE)):
            UE = self.s_UE[_id]
            UE_ = self.s_UE[0:_id]+self.s_UE[_id+1:]+self.c_UE[:]
            P_in = 0 # Total inference signal power (without noise)
            COMA = SUM_COMA - UE.P_s*UE.h_r@UE.v@(UE.v.transpose().conjugate())@(UE.h_r.transpose().conjugate())
            UE.u = np.linalg.inv(COMA)@UE.h_r@UE.v@np.linalg.inv(UE.v.transpose().conjugate()@UE.A.transpose().conjugate()@np.linalg.inv(COMA)@UE.h_r@UE.v)
            UE.u = UE.u.transpose().conjugate()
            # pdb.set_trace()
            for UE_in in UE_:
                P_in += np.linalg.norm(UE.u@UE_in.h_r@UE.v)**2*UE_in.P_s
            P_useful = np.linalg.norm(UE.u@UE.h_r@UE.v)**2*UE.P_s
            R_ = duty_factor/(2*T) * np.log2(1 + 4*(np.pi**2)*(sigma**2)*(self.B_s**2)*T*P_useful/(self.P_n*np.linalg.norm(UE.u)**2+P_in))
            R_est.append(R_)
            print("RSINR:",(P_useful/(self.P_n*np.linalg.norm(UE.u)**2+P_in)) )
            # print("RSINR_MDVR:",np.abs(UE.P_s*(UE.v.transpose().conjugate()@UE.h_r.transpose().conjugate()@np.linalg.inv(COMA)@UE.h_r@UE.v)) )
            print("Pin: ",P_in)
            print("P_useful: ",P_useful)
            print("I+R: ",(self.P_n*np.linalg.norm(UE.u)**2+P_in))
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
        Sinr, R_c = self._cal_zf()
        # Sinr, R_c = self._cal_noma()
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
    env = BS(N_t=16, N_c=5, N_s=3)
    env._power_allocation(Pc_s = [10,20,50,90,150], Ps_s = [1,1,1])
    env.get_performance()
    