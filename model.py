import torch
import torch.nn as nn
import torch.nn.functional as F

from wav_pool import DWT, IWT

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) 
        self.conv2 = nn.Conv2d(40, 5, kernel_size=5)
        
        # ------------------------------------------ #
        self.wp    = DWT()
        # ------------------------------------------ #
        
        self.fc    = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.wp(self.conv1(x)))
        x = F.relu(self.wp(self.conv2(x)))
        x = x.view(in_size, -1) 
        x = self.fc(x)
        return x

model = Net()
    
print(model)
