import torch

class SRCNN(torch.nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x