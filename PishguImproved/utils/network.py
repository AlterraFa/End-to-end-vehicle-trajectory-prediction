#network.py
import torch
from utils.gin_conv2 import GINConv
import torch.nn.functional as F
import torch.nn as nn
from utils.CBAM import CBAM
from utils.optimCBAM import LinCBAM


class SelfAggregate(torch.nn.Module):
    def __init__(self, input_ch):
        super(SelfAggregate, self).__init__()
        
        output_ch = int(input_ch/2)
        self.l1 = torch.nn.Linear(input_ch, output_ch)
        torch.nn.init.xavier_uniform_(self.l1.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        input_ch = output_ch
        output_ch = int(input_ch/2)
        self.l2 = torch.nn.Linear(input_ch, output_ch)
        torch.nn.init.xavier_uniform_(self.l2.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        return x


class NeighborAggregate(torch.nn.Module):
    def __init__(self, input_ch):
        super(NeighborAggregate, self).__init__()
        # 120 -> 30
        
        output_ch = input_ch // 2 #
        
        self.LAD1 = nn.Sequential(
            nn.Linear(input_ch, output_ch),
            nn.Dropout(0.02),
            LinCBAM(output_ch),
        )
        
        input_ch = output_ch
        output_ch = input_ch // 2 # Reducing further to 30
        
        self.LAD2 = nn.Sequential(
            nn.Linear(input_ch, output_ch),
            nn.Dropout(0.02),
            LinCBAM(output_ch),
        )
        
    def forward(self, x):
        x = self.LAD1(x)
        x = self.LAD2(x)
        return x
        

        

class NetGINConv(torch.nn.Module):
    def __init__(self, num_features, output_size):
        super(NetGINConv, self).__init__()
        self.num_cords = 2
        self.input_steps = int(num_features/self.num_cords) # 15 steps

        feature_expansion = 2
        self.fc = torch.nn.Linear(int(num_features*2),int(num_features*2*feature_expansion))
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        nn = SelfAggregate(self.input_steps*self.num_cords*2*feature_expansion)
        nn2 = NeighborAggregate(self.input_steps*self.num_cords*2*feature_expansion) 
        self.conv1 = GINConv(nn, nn2, train_eps=True)

        input_ch = self.num_cords
        output_ch = 64
        self.conv2Da = torch.nn.Conv2d(input_ch, output_ch, (2, 2),stride=2)
        torch.nn.init.xavier_uniform_(self.conv2Da.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        self.cbam1 = CBAM (output_ch)
        
        input_ch = output_ch
        output_ch = output_ch*2
        self.conv2Db = torch.nn.Conv2d(input_ch, output_ch, (2, 1), stride=2)
        torch.nn.init.xavier_uniform_(self.conv2Db.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        self.cbam2 = CBAM (output_ch)

        input_ch = output_ch
        output_ch = output_ch*2
        self.conv2Dc = torch.nn.Conv2d(input_ch, output_ch, (2, 1), stride=2)
        torch.nn.init.xavier_uniform_(self.conv2Dc.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        self.cbam3 = CBAM (output_ch)

        input_ch = output_ch
        output_ch = output_size
        self.conv2Dd = torch.nn.Conv2d(input_ch, output_ch, (1, 1))
        torch.nn.init.xavier_uniform_(self.conv2Dd.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
        
    def forward(self, x, x_real, edge_index):
        x1 = F.leaky_relu(self.fc(x_real))
        x1 = F.leaky_relu(self.conv1(x1, edge_index))
        x1 = x1.reshape(x.shape)
        x = torch.cat((x,x1),1)
        x = F.leaky_relu(self.conv2Da(x))
        x = self.cbam1(x)
        x = F.leaky_relu(self.conv2Db(x))
        x = self.cbam2(x)
        x = F.leaky_relu(self.conv2Dc(x))
        x = self.cbam3(x)
        #Prediction
        x = F.leaky_relu(self.conv2Dd(x))
        return x