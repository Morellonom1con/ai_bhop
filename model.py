import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, device=None):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # convert to tensors on device
        state = torch.tensor(np.array(state), dtype=torch.float, device=self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float, device=self.device)
        action = torch.tensor(np.array(action), dtype=torch.float, device=self.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float, device=self.device)

        # ensure batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # preds: (batch, action_dim)
        pred = self.model(state)

        # next predictions (detached)
        with torch.no_grad():
            next_pred = self.model(next_state)

        # compute scalar target Q_new per sample
        target = pred.clone()
        batch_size = state.shape[0]
        for idx in range(batch_size):
            r = reward[idx]
            if not done[idx]:
                Q_new = r.item() + self.gamma * torch.max(next_pred[idx]).item()
            else:
                Q_new = r.item()

            # action_mask: use absolute action to weight which outputs to update
            # normalize action to [-1,1] for mx and [0,1] for buttons expected
            # assume action vector layout: [mx, b1, b2, b3, b4, b5]
            act = action[idx]

            # build mask in [0,1] form: for mx use |mx| (how strongly chosen),
            # for buttons use the binary value
            # clamp to [0,1]
            mx_mask = torch.clamp(act[0].abs() / 20.0, 0.0, 1.0)  # if mx ∈ ~[-20,20], scale it
            btn_mask = torch.clamp(act[1:], 0.0, 1.0)             # buttons should be 0/1
            mask = torch.cat([mx_mask.unsqueeze(0), btn_mask], dim=0).to(self.device)

            # update only where mask>0: move pred toward Q_new by mask fraction
            # target = pred + mask*(Q_new - pred)
            target[idx] = pred[idx] + mask * (Q_new - pred[idx])

        # optimize
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
