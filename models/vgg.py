'''VGG in PyTorch.
Reference:
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks For Large-scale Image Recognition. arXiv:1409.1556
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, landmark_num):
        super(VGG, self).__init__()
        self.landmark_num = landmark_num
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 7)
        self.attention = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.landmark_layer = nn.Linear(512, self.landmark_num*2)
        # self.softmax_opt = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        class_prob = self.classifier(out)
        attention_weights = self.attention(out)
        weighted_prob = attention_weights * class_prob
        landmark = self.landmark_layer(out).reshape(-1, self.landmark_num, 2)
        return attention_weights, weighted_prob, landmark

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
