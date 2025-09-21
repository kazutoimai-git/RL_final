import torch
import torch.nn as nn
import torch.nn.functional as F

class OthelloNNet(nn.Module):
    def __init__(self, game):#パーツづくりの箇所
        super(OthelloNNet, self).__init__()
        self.board_x, self.board_y = 8, 8
        action_size = game.getActionSize()
        
        self.conv1 = nn.Conv2d(1, 512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512 * (self.board_x - 4) * (self.board_y - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        
        self.fc_pi = nn.Linear(512, action_size) 
        self.fc_v = nn.Linear(512, 1)

    def forward(self, s):#処理の流れ、
        s = s.view(-1, 1, self.board_x, self.board_y)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        
        s = s.view(-1, 512 * (self.board_x - 4) * (self.board_y - 4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=0.3, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=0.3, training=self.training)
        
        pi = self.fc_pi(s)
        v = self.fc_v(s)
        
        return F.log_softmax(pi, dim=1), torch.tanh(v)