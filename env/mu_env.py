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
        self.d = np.random.rand()*10.0 # range{m}
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
        self.w = deepcopy(self.at)
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
        # print("range:", self.d)
        # print("g:", self.g)

    def update(self,rand_theta=True):
         # Channel Generation
        # self.d = np.random.rand()*2+1.0 # range{m}
        self.d = np.random.rand()*10.0 # range{m}
        if rand_theta:
            self.theta = np.random.rand()*np.pi# angle 0~pi
        self.g = (3*10e8/(4*np.pi*self.d*10*10e9))**2# PathLoss of Communication
        self.g_r = (3*10e8/(4*np.pi*(self.d**2)*10*10e9))**2# PathLoss of Radar
        self.at = 1/np.sqrt(self.N_t)*np.expand_dims(np.exp(1j * np.arange(start=0, stop=self.N_t , step=1, dtype=np.complex128) * np.pi *np.sin(self.theta)), axis=-1)# Steering Vector
        self.ar = 1/np.sqrt(self.N_r)*np.expand_dims(np.exp(1j * np.arange(start=0, stop=self.N_r , step=1, dtype=np.complex128) * np.pi *np.sin(self.theta)), axis=-1)# Steering Vector
        self.h = np.sqrt(self.g)*np.sqrt(self.N_t*self.N_r)*self.ar@self.at.conjugate().transpose() # Communication Channel
        self.h_r = np.sqrt(self.g_r)*self.N_t*self.at@self.at.conjugate().transpose() # Radar Channel
        self.A = self.at@self.at.conjugate().transpose()
        self.h_norm = np.linalg.norm(self.h)
        self.v = deepcopy(self.at)
        self.w = deepcopy(self.at)
        #
        
        

class BS(object):
    def __init__(self, N_t=16, N_c=5, N_s=2):
        self.c_UE = [UE(id="c"+str(i),N_t=N_t,is_c=1,theta=0.05*i*np.pi) for i in range(N_c)]  
        # Sort to H norm descending     
        self.s_UE = [UE(id="s"+str(i),N_t=N_t,is_s=1,theta=0.05*i*np.pi+0.3) for i in range(N_s)] 
        self.B_c = 10e+6 # Hz Communication Bandwidth
        self.B_s = 10e+6 # Hz Sensing Bandwidth
        self.P_tot = 10*(N_c+N_s) # W
        self.P_n = 1e-7 # noise power
        self.N_t = N_t
        self.N_c = N_c
        self.N_s = N_s
        ##############################
        self.time = 0
        self.max_time = 200
        self.reward = 0
        ##############################
        #           MEMORY           #
        ##############################
        self.Rc_s = np.zeros((self.max_time,self.N_c))
        self.Rs_s = np.zeros((self.max_time,self.N_s))
    
    def _cal_zf(self, update = True):
        Sinr = []
        R_c = []
        Full_H = np.concatenate([UE.h for UE in self.c_UE], axis=0)
        _H = Full_H.conjugate().transpose()@np.linalg.inv(Full_H@Full_H.conjugate().transpose())
        Full_W = _H*np.sqrt(self.N_c)/np.linalg.norm(_H)
        
        for _id in range(len(self.c_UE)):
            UE = self.c_UE[_id]
            UE.w = Full_W[:,_id]
        for _id in range(len(self.c_UE)):
            UE = self.c_UE[_id]
            UE_in = self.c_UE[0:_id] + self.c_UE[_id+1:]
            P_si = np.linalg.norm(UE.h@UE.w)**2*UE.P_c
            P_in = 0.0
            if len(UE_in) > 0:
                P_in += np.sum([np.linalg.norm(UE.h@UE_.w)**2*UE_.P_c for UE_ in UE_in])
            Sinr.append((P_si/(P_in+UE.P_n)))
            R_c.append(self.B_c * np.log2(1+P_si/(P_in+UE.P_n)))
            if update:
                UE.Sinr = P_si/(P_in+UE.P_n)
                UE.R_c = self.B_c * np.log2(1+UE.Sinr)   
            # pdb.set_trace()
            
        return Sinr,R_c 

    def _cal_radar_v0(self, update=True):   
        R_est = []
        # Radar Processing Params 
        duty_factor = 0.1 # Pulse Duration Interval / Pulse Repetition Interval
        T = 0.000001 # 1us
        sigma = 0.01 # Processing Noise
        # Get the Covariance Matrix of Total Transmitted Signal 
        SUM_COMA = np.zeros((self.N_t,self.N_t), dtype=np.complex128)
        full_ue = self.s_UE+self.c_UE
        for _id in range(len(full_ue)):
            UE = full_ue[_id]
            SUM_COMA += UE.P_s*UE.h_r@UE.v@(UE.v.transpose().conjugate())@(UE.h_r.transpose().conjugate())
        for _id in range(len(self.s_UE)):
            UE = self.s_UE[_id]
            UE_ = self.s_UE[0:_id]+self.s_UE[_id+1:]+self.c_UE[:]
            P_in = 0 # Total inference signal power (without noise)
            # COMA Calculated by DIFF
            COMA_DIFF = SUM_COMA - UE.P_s*UE.h_r@UE.v@(UE.v.transpose().conjugate())@(UE.h_r.transpose().conjugate()) + np.identity(self.N_t,dtype=np.complex128)*self.P_n
            CHECK_COMA = np.identity(self.N_t,dtype=np.complex128)*self.P_n
            for check_id in range(len(UE_)):
                UE_check = UE_[check_id]
                CHECK_COMA += UE_check.P_s*UE_check.h_r@UE_check.v@(UE_check.v.transpose().conjugate())@(UE_check.h_r.transpose().conjugate()) 
            # COMA Calculated by ADD
            COMA = deepcopy(CHECK_COMA)            
            try:
                UE.u = np.linalg.inv(COMA)@UE.A@UE.v@np.linalg.inv(UE.v.transpose().conjugate()@UE.A.transpose().conjugate()@np.linalg.inv(COMA)@UE.A@UE.v)
            except:
                print("inv error")
                fp = open("invlog.txt","w+")
                print("inv error\r",file=fp)
                print(f"Sensing user {UE.id} inv error\r",file=fp)
                print(f"Interference users {[log_UE.id for log_UE in UE_]} inv error\r",file=fp)
                print("COMA:\r",file=fp)
                print(COMA,"\r",file=fp)
                print("COMA_DIFF:\r",file=fp)
                print(COMA_DIFF-COMA,"\r",file=fp)
                
                for logp in range(len(full_ue)):
                    log_UE = full_ue[logp]
                    log_coma = log_UE.P_s*log_UE.h_r@log_UE.v@(log_UE.v.transpose().conjugate())@(log_UE.h_r.transpose().conjugate())
                    print(f"check UE {log_UE.id} COMA:\r",file=fp)
                    print(log_coma,"###\r",file=fp)
                    print(log_UE.id,file=fp)
                fp.close()    
                pdb.set_trace()
            UE.u = UE.u.transpose().conjugate()
            UE.u = 1/np.linalg.norm(UE.u)*UE.u
            # pdb.set_trace()
            for UE_in in UE_:
                P_in += np.linalg.norm(UE.u@UE_in.h_r@UE.v)**2*UE_in.P_s
            P_useful = np.linalg.norm(UE.u@UE.h_r@UE.v)**2*UE.P_s
            R_ = duty_factor/(2*T) * np.log2(1 + 4*(np.pi**2)*(sigma**2)*(self.B_s**2)*T*P_useful/(self.P_n*np.linalg.norm(UE.u)**2+P_in))
            R_est.append(R_)
            # print("RSINR (dB):",10*np.log10(P_useful/(self.P_n*np.linalg.norm(UE.u)**2+P_in)) )
            # # print("RSINR_MDVR:",np.abs(UE.P_s*(UE.v.transpose().conjugate()@UE.h_r.transpose().conjugate()@np.linalg.inv(COMA)@UE.h_r@UE.v)) )
            # print("Pin: ",P_in)
            # print("P_useful: ",P_useful)
            # print("N: ",(self.P_n*np.linalg.norm(UE.u)**2))
            # pdb.set_trace()
        return R_est
    
    def _cal_radar(self, update=True):   
        R_est = []
        Sinr_r = []
        # Radar Processing Params 
        duty_factor = 0.1 # Pulse Duration Interval / Pulse Repetition Interval
        T = 0.000001 # 1us
        sigma = 0.01 # Processing Noise
        full_ue = self.s_UE+self.c_UE
        # Get the Covariance Matrix of Total Transmitted Signal 
        for _id in range(len(self.s_UE)):
            UE = self.s_UE[_id]
            UE_ = self.s_UE[0:_id]+self.s_UE[_id+1:]+self.c_UE[:]
            P_in = 0 # Total inference signal power (without noise)
            # COMA Calculated by DIFF
            CHECK_COMA = np.identity(self.N_t,dtype=np.complex128)*self.P_n
            for check_id in range(len(UE_)):
                UE_check = UE_[check_id]
                CHECK_COMA += UE.P_s*UE_check.h_r@UE.v@(UE.v.transpose().conjugate())@(UE_check.h_r.transpose().conjugate()) 
            # COMA Calculated by ADD
            COMA = deepcopy(CHECK_COMA)   
            #         
            UE.u = np.linalg.inv(COMA)@UE.A@UE.v@np.linalg.inv(UE.v.transpose().conjugate()@UE.A.transpose().conjugate()@np.linalg.inv(COMA)@UE.A@UE.v)
            UE.u = UE.u.transpose().conjugate()
            UE.u = 1/np.linalg.norm(UE.u)*UE.u
            # pdb.set_trace()
            for UE_in in UE_:
                P_in += np.linalg.norm(UE.u@UE_in.h_r@UE.v)**2*UE.P_s
            P_useful = np.linalg.norm(UE.u@UE.h_r@UE.v)**2*UE.P_s
            R_ = duty_factor/(2*T) * np.log2(1 + 4*(np.pi**2)*(sigma**2)*(self.B_s**2)*T*P_useful/(self.P_n*np.linalg.norm(UE.u)**2+P_in))
            R_est.append(R_)
            Sinr_r.append(P_useful/(self.P_n*np.linalg.norm(UE.u)**2+P_in))
            # print("RSINR (dB):",10*np.log10(P_useful/(self.P_n*np.linalg.norm(UE.u)**2+P_in)) )
            # print("RSINR_MDVR:",np.abs(UE.P_s*(UE.v.transpose().conjugate()@UE.h_r.transpose().conjugate()@np.linalg.inv(COMA)@UE.h_r@UE.v)) )
            # print("Pin: ",P_in)
            # print("P_useful: ",P_useful)
            # print("N: ",(self.P_n*np.linalg.norm(UE.u)**2))
            # pdb.set_trace()
        return Sinr_r, R_est
    
    
    def update_channels(self,):
        # Update Communication UE Channel
        for UE in self.c_UE:
            UE.update()
        # Re-sort NOMA Index
        # Update Sensing UE Channel
        for UE in self.s_UE:
            UE.update()

    def get_performance(self, print_log=False):
        Sinr_c, R_c = self._cal_zf()
        Sinr_r, R_est = self._cal_radar()
        if print_log:
            print("C Sinr (dB)", 10*np.log(Sinr_c))
            print("S Sinr (dB)", 10*np.log(Sinr_r))
            print("R_c", R_c)
            print("R_est", R_est)
        return R_c, R_est

    def _power_allocation(self, Pc_s, Ps_s):
        for _id in range(len(self.c_UE)):
            UE = self.c_UE[_id]
            UE.P_c = Pc_s[_id]
        for _id in range(len(self.s_UE)):
            UE = self.s_UE[_id]
            UE.P_s = Ps_s[_id]
    
    def _equal_allocation(self,):
        Pc_s = np.array([self.P_tot/(self.N_c+self.N_s)]*self.N_c)
        Ps_s = np.array([self.P_tot/(self.N_c+self.N_s)]*self.N_s)
        for _id in range(len(self.c_UE)):
            UE = self.c_UE[_id]
            UE.P_c = Pc_s[_id]
        for _id in range(len(self.s_UE)):
            UE = self.s_UE[_id]
            UE.P_s = Ps_s[_id]
        R_c, R_est = self.get_performance()
        return R_c, R_est
        
    
    def _reward(self,R_c,R_est):
        return (np.sum(R_c)/10000000.0+np.sum(R_est)/100000.0*0.5)
    
    def _penalty(self,pa_ratio):
        sum_ratio = np.sum(pa_ratio) 
        # return (sum_ratio - 1)>1e-4 # upper bounded penalty
        return (sum_ratio - 1)>1e-4 and (1-sum_ratio)>1e-1# upper bounded penalty
        
    def _penalty_lower(self,pa_ratio):
        sum_ratio = np.sum(pa_ratio) 
        return (1-sum_ratio)>1e-1 # LOWER bounded penalty
    
    def step(self, pa_ratio=None, freeze=False):
        done = 0
        _penalty = 0
        if pa_ratio is None:
            power_allocation = np.array([self.P_tot/(self.N_c+self.N_s)]*(self.N_c+self.N_s))
        else:
            # EQUAL COMM POWER
            pa_ratio_ = np.zeros(self.N_c+self.N_s)
            pa_ratio_[0:self.N_c] = pa_ratio[0]/self.N_c
            pa_ratio_[self.N_c:] = pa_ratio[1:]
            power_allocation = pa_ratio_*self.P_tot
            _penalty = self._penalty(pa_ratio_)
            _penalty_lower = self._penalty_lower(pa_ratio_)
            if np.sum(pa_ratio_)-1.0 > 1e-4:
                pa_ratio_ = pa_ratio_/np.sum(pa_ratio_)
            assert(np.sum(pa_ratio_)-1.0 <= 1e-4)
        
        # get basline PA
        bl_R_c, bl_R_est = self._equal_allocation()
        
        # apply PA
        self._power_allocation(Pc_s=power_allocation[0:self.N_c], Ps_s=power_allocation[self.N_c:])
        # calculate performance
        R_c, R_est = self.get_performance()
        # update history
        self.Rc_s[self.time,:] = R_c
        self.Rs_s[self.time,:] = R_est
        # update env reward
        # self.reward = pa_ratio
        # self.reward = (_penalty==0)*10+(_penalty==1)*-10 +\
        #     (_penalty_lower==0)*10+(_penalty_lower==1)*-10#*np.abs(1-np.sum(pa_ratio_))
        self.reward = (_penalty==0)*10*self._reward(R_c,R_est)/50.0 + (_penalty==1)*-10
        bl_reward = self._reward(bl_R_c,bl_R_est)/50.0*10
        assert(np.isinf(self.reward).sum()==0)
        # get next state
        self.time += 1
        if self.time % self.max_time == 0:
            self.time = 0
            done = 1
            reward = self.reward
            next_state,_ = self.reset()
            return next_state,reward,done,bl_R_c, bl_R_est, bl_reward
        if not freeze:
            self.update_channels()
        next_state = self._get_state()
        # next_state = np.concatenate([Full_Hc,Full_Hr])
        return next_state, self.reward, done, bl_R_c, bl_R_est, bl_reward
    
    def _get_state(self,):
        Full_Hc = np.concatenate([UE.h for UE in self.c_UE], axis=0).flatten()
        Full_Hr = np.concatenate([UE.h for UE in self.s_UE], axis=0).flatten()
        Full_d = np.array([UE.d for UE in self.c_UE+self.s_UE]).flatten()/10.0
        Full_theta = np.array([UE.theta for UE in self.c_UE+self.s_UE]).flatten()/np.pi
        next_state = np.concatenate([Full_Hc,Full_Hr],axis=0)
        # next_state = np.concatenate([Full_d,Full_theta])
        return next_state
    
    def reset(self):
        done = 0
        self.time = 0
        # self.reward = 0
        ##############################
        #           MEMORY           #
        ##############################
        self.Rc_s = np.zeros((self.max_time,self.N_c))
        self.Rs_s = np.zeros((self.max_time,self.N_s))
        next_state = self._get_state()
        return next_state,done
        

if __name__ == "__main__":
    np.random.seed(777)
    env = BS(N_t=16, N_c=5, N_s=3)
    env._power_allocation(Pc_s = [10,10,10,10,10], Ps_s = [10,10,10])
    env.get_performance(True)
    
    # ns,d = env.step()
    for i in range(0,4000):
        rio = np.random.rand(8)
        rio = np.exp(rio)/np.sum(np.exp(rio))
        ns,r,d = env.step()
    #     if d == 1:
    #         print(d)
    pdb.set_trace()
    