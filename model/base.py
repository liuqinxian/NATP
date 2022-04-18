import torch
import torch.nn as nn
import numpy as np

class base(nn.Module):
    def __init__(self):
        super(base, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm3d(32)
        
        self.conv3 = nn.Conv3d(32, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm3d(32)
        
        self.conv4 = nn.Conv3d(32, 16, kernel_size=3)
        self.bn4 = nn.BatchNorm3d(16)
        
        self.conv5 = nn.Conv3d(16, 8, kernel_size=3)
        self.bn5 = nn.BatchNorm3d(8)
        
        self.conv6 = nn.Conv3d(8, 1, kernel_size=3)
        self.bn6 = nn.BatchNorm3d(1)
        
        self.line = nn.Linear(64, 1)
        self.line1 = nn.Linear(8000, 400)
        self.line2 = nn.Linear(400, 20)
        self.line3 = nn.Linear(20, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, cubes):
        x_output = []
        n_cube = cubes.shape[1]
        for i in range(n_cube):
            x = cubes[:, i, :, :, :]    # x: [b, 64, 64, 64]
            # print(x.shape)
            # for j in range(cubes.shape[0]):
            #     print(cubes[j, 16, :, :])
            # return 0
            x = x.type(torch.cuda.FloatTensor)
            x = x.unsqueeze(1)          # x: [b, 1, 64, 64, 64]
            
            x = self.conv1(x)           # x: [b, 32, 30, 30, 30]
            # x = self.bn1(x)             # x: [b, 32, 30, 30, 30]  
            x = self.relu(x)
            
            x = self.conv2(x)           # x: [b, 32, 28, 28, 28]
            # x = self.bn2(x)
            x = self.relu(x)
            
            x = self.conv3(x)           # x: [b, 32, 26, 26, 26]
            # x = self.bn3(x)
            x = self.relu(x)
            
            x = self.conv4(x)           # x: [b, 16, 24, 24, 24]
            # x = self.bn4(x)
            x = self.relu(x)
            
            x = self.conv5(x)           # x: [b, 8, 22, 22, 22]
            # x = self.bn5(x)
            x = self.relu(x)
            
            x = self.conv6(x)           # x: [b, 1, 20, 20, 20]
            # x = self.bn6(x)
            x = self.relu(x)

            batch_size = x.shape[0]
            x = x.view(batch_size, -1)      # x: [b, 20*20*20]
            
            x = self.line1(x)           # x: [b, 400]
            x = self.relu(x)
            x = self.dropout(x)
            
            x = self.line2(x)           # x: [b, 20]
            x = self.relu(x)
            x = self.dropout(x)
            
            x = self.line3(x)           # x: [b, 1]
            x = self.relu(x)
            x = self.dropout(x)
            
            x_output.append(x)
        
        x = torch.stack(x_output, dim=1)   # x: [b, 64, 1]        
        x = torch.squeeze(x, dim=-1)
        
        x = self.line(x)            # x: [b, 1]
        x = self.sigmoid(x)
        return x



