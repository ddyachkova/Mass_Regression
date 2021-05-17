import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = out_channels//in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.downsample, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.downsample)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample > 1:
            residual = self.shortcut(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):

    #def __init__(self, in_channels, nblocks, fmaps, fc_nodes, fc_layers):
    def __init__(self, in_channels, nblocks, fmaps):
        super(ResNet, self).__init__()
        self.fmaps = fmaps
        self.nblocks = nblocks
        #self.fc_nodes = fc_nodes
        #self.fc_layers = fc_layers
        
        #self.conv0 = nn.Conv2d(in_channels, fmaps[0], kernel_size=7, stride=2, padding=1)
        self.conv0 = nn.Conv2d(in_channels, fmaps[0], kernel_size=7, stride=1, padding=1)
        self.layer1 = self.block_layers(self.nblocks, [fmaps[0],fmaps[0]])
        self.layer2 = self.block_layers(1, [fmaps[0],fmaps[1]])
        self.layer3 = self.block_layers(self.nblocks, [fmaps[1],fmaps[1]])

        # no FC
        #if self.fc_layers == 0:
        #self.fc = nn.Linear(self.fmaps[1], 1)
        self.fc = nn.Linear(self.fmaps[1]+2, 1)
        # with FC
        #else:
        #    #self.fcin = nn.Linear(self.fmaps[1], self.fc_nodes)
        #    self.fcin = nn.Linear(self.fmaps[1]+2, self.fc_nodes)
        #    self.fc = nn.Linear(self.fc_nodes, self.fc_nodes)
        #    self.fcout = nn.Linear(self.fc_nodes, 1)
        #    self.drop = nn.Dropout(p=0.2)
        #    self.relu = nn.ReLU()
        self.GlobalMaxPool2d = nn.AdaptiveMaxPool2d((1,1))
        
    def block_layers(self, nblocks, fmaps):
        layers = []
        for _ in range(nblocks):
            layers.append(ResBlock(fmaps[0], fmaps[1]))
        return nn.Sequential(*layers)

    def forward(self, X):

#         print(X.shape)
        x = self.conv0(X[0])
#         print(x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
#         print(x.shape)

        x = self.layer1(x)
#         print(x.shape)
        x = self.layer2(x)
#         print(x.shape)
        x = self.layer3(x)
#         print(x.shape)
        
        #x = F.max_pool2d(x, kernel_size=x.size()[2:])
        #x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = self.GlobalMaxPool2d(x)
        print(x.shape)
        x = x.view(x.size()[0], self.fmaps[1])
        # concat with seed pos
        x = torch.cat([torch.tensor(x), X[1], X[2]], 1)
        # FC
        #if self.fc_layers == 0:
        x = self.fc(x)
        #else:
        #    x = self.fcin(x)
        #    for _ in range(self.fc_layers):
        #        x = self.fc(x)
        #        x = self.relu(x)
        #        x = self.drop(x)
        #    x = self.fcout(x)
        return x

