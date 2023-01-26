import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        #print (input_size, hidden_size, output_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

        #print (type(self.linear1))

    def forward(self, x: torch.Tensor):
        x = x.float()
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x
    
    def save(self, iter,  file_name = 'snapshot.pth'):

        # TODO: Fix saving path
        dirname = os.path.dirname(__file__)
        model_folder_path = os.path.join(dirname, './selfplay_snapshots')
        
        file_name = f"generation{iter}_{file_name}"

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__ (self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()
        self.counter = 0

    def train_step(self, state, action, reward, next_state, done):
        self.counter += 1
        
        state = torch.tensor(state, dtype = torch.float, device="cuda"
        )
        next_state = torch.tensor(next_state, dtype = torch.float, device="cuda"
        )
        action = torch.tensor(action, dtype = torch.long, device="cuda"
        )
        reward = torch.tensor(reward, dtype = torch.float, device="cuda"
        )
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            #print (done)

        
        if not (self.counter+49)%50:
            self.lagging_model = self.model
        
        pred = self.model(state)


        target = self.lagging_model(state) #pred.clone()

        for i in range(len(done)):

            Q_new = reward[i]
            if not done[i]: 
                Q_new = (reward[i] + (self.gamma * torch.max(self.model(next_state[i])))) #/(1+self.gamma) # PROBLEM

            target[i][torch.argmax(action[i]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()