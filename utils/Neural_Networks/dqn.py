import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np
import os

from utils.Neural_Networks.CudaTracker import CUDAMemoryTracker


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)   # input layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # hidden layer
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # output layer

    def forward(self, x):
        # activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --------

class MLP2(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, dropout_prob = 0.5):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)   # input layer
        self.dropout1 = nn.Dropout(p = 0.2)        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # hidden layer
        self.dropout2 = nn.Dropout(p=dropout_prob)        
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # output layer

    def forward(self, x):
        # activation function
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)        
        return self.fc3(x)

# --------

class MLP3(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, dropout_prob = 0.5):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # input layer  
        self.dropout1 = nn.Dropout(p = 0.2)
        self.fc2 = nn.Linear(hidden_dim, 16)         # hidden layer
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(16, action_dim)         # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))     
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.softmax(self.fc3(x), dim=-1) 
        return x    


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # capacity of buffer
        self.buffer = []  # replay buffer
        self.position = 0
        print('Replay memory created.')

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

  
class DQN:
    
    dropout = True
    softupdate = False    
    
    def __init__(self, state_dim, action_dim, cfg):

        self.Algo = cfg.algo_name  
        self.memory_tracker = CUDAMemoryTracker(self, output_dir = cfg.Output_PATH, device_id = 0)            
        self.action_dim = action_dim
        self.device = cfg.device
        self.tau = cfg.tau                
        self.gamma = cfg.gamma         
        self.soft_update = cfg.softupdate        
        self.frame_idx = 0  
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-1. * frame_idx / cfg.epsilon_decay)
        
        self.batch_size = cfg.batch_size
        
        if self.dropout:
            self.policy_net = MLP2(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
            self.target_net = MLP2(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        else:            
            self.policy_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
            self.target_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):  
            target_param.data.copy_(param.data)
            
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        #self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)   
             
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.loss = 0        

        self.__repr__()      
    
    def choose_action(self, state) -> int:
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()  # choose the action with maximum q-value
        else:
            action = random.randrange(self.action_dim)
        return action

    def update(self) -> int:
        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        
        # transfer to tensor
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)

        self.memory_tracker.track_memory_usage()        

        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)        

        if self.Algo == 'DQN':                            
            next_q_values = self.target_net(next_state_batch).max(1)[0].detach()

        if self.Algo == 'DDQN':          
             argmax_a = self.policy_net(next_state_batch).argmax(dim=1).unsqueeze(-1)         
             next_q_values = self.target_net(next_state_batch).gather(1,argmax_a).squeeze(1)                        

        if self.Algo not in ['DQN', 'DDQN']:
            raise NotImplementedError                         

        target_Q = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        
        self.loss = nn.MSELoss()(q_values, target_Q.unsqueeze(1))
        #self.HS = nn.SmoothL1Loss()(q_values, target_Q.unsqueeze(1))             
        
        self.optimize_policy_net()

    def optimize_policy_net(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        for param in self.policy_net.parameters():  # avoid gradient explosion by using clip
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()   

        # has been tried, but gradient clipping withing (-1,1) is easier and less prone to potential errors
        # if self.normClipping:
        #     # Gradient clipping by the L^2 norm
        #     nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0, norm_type=2)           

        if self.Algo == "DDQN" and self.softupdate is True:
             for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
                 target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)                

    def save(self, path):
        torch.save(self.target_net.state_dict(), os.path.join(path,'model.pth'))
        print(f'Network has been saved to: {path}')        

    def load(self, path):
        try:
            self.target_net.load_state_dict(torch.load( os.path.join( path, 'model.pth')) )
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                param.data.copy_(target_param.data)
            print('Network has been loaded')   

        except FileNotFoundError:
            print(f'Error: The file {path}\\model.pth does not exist.')   

    def __repr__(self) -> str:
        content = f"""New {self.Algo} network instance has been created \nDevice: {self.device}"""     
        print(content)
                                   