import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super().__init__()
        self.device = device
        self.seq = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)            
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.seq(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, lr, gamma):
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = gamma

        self.TARGET_REPLACE_FREQ = 10 
        self.learn_step_counter = 0
        self.evaluate_net  = Linear_QNet(16, 256, 3, self.device).to(self.device)
        self.target_net = Linear_QNet(16, 256, 3, self.device).to(self.device)
        self.target_net.load_state_dict(self.evaluate_net.state_dict())
        self.optimizer = optim.Adam(self.evaluate_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        

        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        # Q 
        q_eval = self.evaluate_net(state)

        target = q_eval.clone()
        for idx in range(len(done)):
            q_target = reward[idx]
            if not done[idx]:
                q_target = reward[idx] + self.gamma * torch.max(self.target_net(next_state[idx]).detach())

            target[idx][torch.argmax(action[idx]).item()] = q_target
    
        self.optimizer.zero_grad()
        loss = self.criterion(q_eval, target)
        loss.backward()

        self.optimizer.step()
        return loss.item()



