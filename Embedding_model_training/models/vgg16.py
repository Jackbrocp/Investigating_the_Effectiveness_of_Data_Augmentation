import torch
import torch.nn as nn

class Vgg16(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.feature_1 = nn.Sequential()
        self.classifier = nn.Sequential()

        # add feature layers
        self.feature_1.add_module('conv1_1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu1_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv1_2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu1_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_1.add_module('conv2_1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu2_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv2_2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu2_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_1.add_module('conv3_1', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu3_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv3_2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu3_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv3_3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu3_3', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool3', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_1.add_module('conv4_1', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu4_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv4_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu4_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv4_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu4_3', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool4', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_1.add_module('conv5_1', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu5_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv5_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu5_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv5_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu5_3', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool5', nn.MaxPool2d(kernel_size=2, stride=2))

        # add classifier
        self.classifier.add_module('fc6', nn.Linear(512, 10))
 

    def forward(self, x):
        x = self.feature_1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x