import torch
import numpy as np
import pdb
class State():
    def __init__(self, device, A1=None, A2=None, A3=None, C=None, states=None):
        self.device = device
        if states is not None:
            self.A1 = torch.cat([state.A1 for state in states])
            self.A2 = torch.cat([state.A2 for state in states])
            self.A3 = torch.cat([state.A3 for state in states])
            self.C = torch.cat([state.C for state in states])
            self.As = torch.cat([self.A1,self.A2,self.A3], dim=-1)
            
            
        else:
            self.flatten = torch.nn.Flatten()
            self.A1 = (torch.tensor(A1, device=device, dtype=torch.float32))
            self.A2 = (torch.tensor(A2, device=device, dtype=torch.float32))
            self.A3 = (torch.tensor(A3, device=device, dtype=torch.float32))
            self.As = torch.cat([self.A1,self.A2,self.A3], dim=-1)
            self.C = (torch.tensor(C, device=device, dtype=torch.float32))
            
class TDNet(torch.nn.Module):
    '''
    DQN Net
    
    '''
    def __init__(self, HV_num=2, Layer_num=4):
        super().__init__()
        self.A_F = torch.nn.Linear(15, 8)
        self.C_F = torch.nn.Linear(5, 8)         
        self.joint_layer = torch.nn.Linear(16, 200) # old: 18; new: 14
        self.relu = torch.nn.ReLU()
        
    def forward(self, state):
        A_F = self.A_F(state.As)
        C_f = self.C_F(state.C)
        joint_value = self.joint_layer(torch.cat((A_F, C_f), dim=-1))
        return joint_value
