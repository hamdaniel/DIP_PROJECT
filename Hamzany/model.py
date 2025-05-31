import torch
import torch.nn as nn
import torch.nn.functional as F

class CompressionTimePredictor(nn.Module):
    def __init__(self, hidden_size=64, iter_size=16):
        super(CompressionTimePredictor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc_img = nn.Linear(32 * 4 * 4, hidden_size)
        self.fc_iter = nn.Linear(1, iter_size)
        self.fc_final = nn.Linear(hidden_size + iter_size, 1)

    def forward(self, image, iter_num):
        x = self.conv(image)
        x = x.view(x.size(0), -1)
        x_img = F.relu(self.fc_img(x))
        x_iter = F.relu(self.fc_iter(iter_num))
        x_cat = torch.cat((x_img, x_iter), dim=1)
        out = self.fc_final(x_cat)
        return out
