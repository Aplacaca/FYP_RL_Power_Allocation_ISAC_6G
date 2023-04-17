from select import select
import torch
import numpy as np
from .extract_modules import State
from .memory import ReplayMemoryPool
import torch.optim.lr_scheduler as lr_scheduler
from .EMA import EMA
class D3QN():
    def __init__(self, device, action_space, net_class,
                max_epsilon=1.0, min_epsilon=0.1, epsilon_decay=1/2000,
                lr=1e-3, memory_size=10000, batch_size=100, gamma=0.9, model=None, enable_ema=False):
        # Basic Part
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.action_space = action_space
        self.batch_size = batch_size
        
        # Memory Part
        self.memory_pool = ReplayMemoryPool(size=memory_size)
        
        # Network Part
        if model is None:
            self.net = net_class().to(device)
        else:
            self.net = net_class().to(device)
            self.net.load_state_dict(torch.load(model))
        self.target_net = net_class().to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()
        
        # Optim Part
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer,step_size=1,gamma=0.8)
        self.loss_func = torch.nn.SmoothL1Loss()
        self.gamma = gamma
        # Mode
        self.is_test = False # If False, action would be selected randomly in some prob.
        
        # EMA
        self.enable_ema = enable_ema
        if enable_ema:
            self.ema = EMA(self.net, decay=0.999)
            self.ema.register()
    def train(self):
        if self.enable_ema:
            self.ema.restore()
        self.net.train()
    def eval(self):
        if self.enable_ema:
            self.ema.apply_shadow()
        self.net.eval()
    def set_lr(self, lr):
        for p in self.optimizer.param_groups:
            p['lr'] = lr
    def select_action(self, state):
        if self.epsilon > np.random.random() and not self.is_test:
            selected_action = np.random.choice(self.action_space)
        else:
            q_val = self.net(state)
            selected_action = q_val.argmax(dim=1)
            selected_action = selected_action.detach().cpu().numpy()
            selected_action = selected_action[0]
        return selected_action
    def optim(self):
        if len(self.memory_pool.memories) < self.batch_size:
            return
        batch = self.memory_pool.get_batch(self.batch_size)
        state = State(self.device, states=[batch[i][0] for i in range(self.batch_size)])
        reward = torch.tensor(np.array([batch[i][1] for i in range(self.batch_size)]), device=self.device).reshape(-1)
        action = torch.tensor(np.array([batch[i][2] for i in range(self.batch_size)]), device=self.device).reshape(-1)
        next_state = State(self.device, states=[batch[i][3] for i in range(self.batch_size)])
        done = torch.tensor(np.array([1-batch[i][4] for i in range(self.batch_size)]), device=self.device)
        current_Q = self.net(state).gather(1, action.unsqueeze(-1))
        with torch.no_grad():
            next_Q = self.target_net(next_state).gather(1, self.net(next_state).argmax(dim=1,keepdim=True)).detach()
        target = reward.unsqueeze(-1)  + self.gamma*next_Q*done.unsqueeze(-1)
        self.optimizer.zero_grad()
        loss = self.loss_func(current_Q, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())
    
    def update_epsilon(self):
        self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
