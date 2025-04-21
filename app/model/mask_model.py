import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskModel(nn.Module):
    def __init__(self):
        super(MaskModel, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.7)
        self.fc3 = nn.Linear(64, 28)
        self.bn3 = nn.BatchNorm1d(28)
        self.dropout3 = nn.Dropout(0.7)
        self.fc4 = nn.Linear(28, 3)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

    @classmethod
    def load_model(cls, model_path, device='cpu'):
        model = cls()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model