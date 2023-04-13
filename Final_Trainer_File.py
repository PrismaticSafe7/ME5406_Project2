import numpy as np
import random
import gym
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from crowd_sim.envs.utils.action import ActionRot, ActionXY


from crowd_sim.envs.policy.policy import Policy

Transition = namedtuple('Transition', ['state','action','reward','observation'])

class action_space():
    def __init__(self):
        self.action

class AgentModel(nn.Module): # CADRL
    def __init__(self, inputs, output_dimensions):
        super().__init__()
        self.output_dimensions = [int(dim) for dim in output_dimensions.split(', ')]
        self.layers = []
        self.dimensions = [inputs] + self.output_dimensions
        for i in range(len(self.output_dimensions)):
            self.layers.append(nn.Linear(self.dimensions[i], self.output_dimensions[i]))

    def forward(self, x):
        for i in range(len(self.output_dimensions) - 1):
            x = self.layers[i](x)
            x = nn.ReLU(x)
        x = self.layers[len(self.layers)](x)
        return x

class AgentModel1(nn.Module):  # LSTM-CADRL
    def __init__(self, num_actions, lstm_hidden_dim = 20, self_state_dim = 4):
        super().__init__()
        self.state_dim = self_state_dim
        self.final_output = num_actions + 1

        # LSTM Component
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(7, lstm_hidden_dim, batch_first=True)  # to use: self.lstm(inputs,(h0,c0)) - c0: initial cell state for each element
        
        # NN Component
        mlp_input_dim = self_state_dim + lstm_hidden_dim
        self.mlp_dim = [256,256,self.final_output]  # 1 output for value function, rest for num_action
        dimensions = [mlp_input_dim] + self.mlp_dim
        self.layers = []
        for i in range(len(self.mlp_dim)):
            self.layers.append(nn.Linear(dimensions[i], self.mlp_dim[i]))

    def forward(self, main_robot_state, other_robot_states):
        # LSTM Portion
        h0 = []
        c0 = []
        _, (hn,cn) = self.lstm(other_robot_states, (h0,c0))
        hn = hn.squeeze(0)
        
        # NN Portion
        output = torch.cat([main_robot_state, hn], dim=1)
        for i in range(len(self.mlp_dim) - 1):
            output = self.layers[i](output)
            output = nn.ReLU(output)
        output = self.layers[len(self.mlp_dim)](output)
        return output

class ReplayMemory(Dataset):
    def __init__(self, cap):
        self.capacity = cap
        self.memory = []
        self.idx = 0
    
    def push(self, data):
        if len(self.memory) < self.idx + 1:
            self.memory.append(data)
        else:
            self.memory[self.idx] = data
        
        self.idx = (self.idx + 1) % self.capacity

    def memory_full(self):
        memory_len = len(self.memory)
        return self.capacity == memory_len

    def __getitem__(self,idx):
        return self.memory[idx]
    
    def __len__(self):
        return len(self.memory)
    
    def clear_memory(self):
        self.memory = []


class Trainer():
    def __init__(self, device, model, memory, batch_size, lr, criterion):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = optim.SGD(self.model.parameters(), lr, momentum=0.8)

    def train_batch(self, num_batches):
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)

        running_loss = 0

        for i in range(num_batches):
            inputs, value = next(iter(self.data_loader))
            
            self.optimizer.zero_grad()
            output = self.model(inputs)
            loss = self.criterion(output,value)
            loss.backward()

            self.optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss/num_batches

        return average_loss
    

class RNN_CADRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'CADRL'
        self.trainable = True
        self.multiagent_training = None
        self.kinematics = "holonomic"
        self.epsilon = None
        self.gamma = 0.9
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.query_env = None
        self.action_space = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.num_actions = None
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.self_state_dim = 4
        self.ob_state_dim = 7
    
    def setup_model(self):
        self.model = AgentModel1(self.num_actions)

    def set_device(self, device):
        self.device = device
        self.model.to(device)
    
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def build_action_space(self, v_pref):
        speeds = [v_pref, v_pref/2, 0]
        action_space = []
        if self.kinematics == "holonomic":
            rotations = np.linspace(0, 2*np.pi, 8, endpoint=False)
        else:
            rotations = np.linspace(-np.pi/6, np.pi/6, 5)
        
        for rotation, speed in zip(rotations, speeds):
            if self.kinematics == "holonomic":
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed,rotation))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space
        self.num_actions = len(action_space)
