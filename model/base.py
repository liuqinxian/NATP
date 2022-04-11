import torch
import torch.nn as nn

class base(nn.Module):
    def __init__(self, args):
        super(base, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3)
        
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.AvgPool3d(kernel_size=3, stride=3)

        self.conv2 = nn.Conv3d(32, 32, kernel_size=3)
        
        self.conv3 = nn.Conv3d(32, 1, kernel_size=3)
        self.bn2 = nn.BatchNorm3d(1)
        self.pool2 = nn.AvgPool3d(kernel_size=3, stride=3)
        
        self.line1 = nn.Linear(729, 32)
        self.line2 = nn.Linear(8, 1)
        self.line3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, cubes):
        x_output = []
        for i in range(8):
            x = cubes[:, i, :, :, :]
            x = x.type(torch.cuda.FloatTensor)
            x = x.unsqueeze(1)
            x = self.relu(self.conv1(x))
            x = self.pool1(x)

            x = self.relu(self.conv3(x))
            x = self.relu(self.conv5(x))
            x = self.pool2(x)

            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
            x = self.relu(self.line1(x))
            x_output.append(x)
        
        x = np.stack(x_output, axis=1)
        x = torch.from_numpy(x)
        x = x.permute(0, 4, 2, 3, 1)
        x = self.line2(x)
        x = x.permute(0, 4, 2, 3, 1)
        x = self.sigmoid(self.line3(x))
        return x



