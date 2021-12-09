import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from copy import copy


class ConvolutionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super(ConvolutionLayer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (int(padding), int(padding))
        self.conv = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, stride=self.stride,
                              padding=self.padding)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x)



class Net_vanilla_cnn_mnist_regularized(nn.Module):
    def __init__(self):
        super(Net_vanilla_cnn_mnist_regularized, self).__init__()

        kernel_sizes = [7, 3, 3]
        pads = (np.array(kernel_sizes) - 1) / 2
        pads = pads.astype(int)
        layers = [30, 60, 120, 60, 128]
        self.post_filter = False
        # network layers
        self.conv1 = ConvolutionLayer(1, layers[0], [kernel_sizes[0], kernel_sizes[0]], stride=1, padding=pads[0])
        self.conv2 = ConvolutionLayer(layers[0], layers[1], [kernel_sizes[1], kernel_sizes[1]], stride=1,
                                      padding=pads[1])
        self.conv3 = ConvolutionLayer(layers[1], layers[2], [kernel_sizes[2], kernel_sizes[2]], stride=1,
                                      padding=pads[2])
        # self.conv4 = ConvolutionLayer(layers[2], layers[3], [kernel_sizes[2], kernel_sizes[2]], stride=1, padding=pads[2])
        # self.conv5 = ConvolutionLayer(layers[3], layers[4], [kernel_sizes[2], kernel_sizes[2]], stride=1, padding=pads[2])

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1 = nn.BatchNorm2d(layers[0])
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn2 = nn.BatchNorm2d(layers[1])
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(layers[2])
        # self.pool4 = nn.MaxPool2d(kernel_size=(2,2))
        # self.bn4 = nn.BatchNorm2d(layers[3])
        # self.pool5 = nn.MaxPool2d(2)
        # self.bn5 = nn.BatchNorm2d(layers[4])
        self.fc1 = nn.Conv2d(layers[2] * 16, 256, 1)
        self.fc1bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.4)
        self.fc2 = nn.Conv2d(256, 10, 1)

        # self.fc1_alt = nn.Conv2d(layers[2]*16, 256, 1)
        # self.fc1bn_alt = nn.BatchNorm2d(256)
        # self.relu_alt = nn.ReLU()
        # self.dropout_alt = nn.Dropout2d(0.7)
        # self.fc2_alt = nn.Conv2d(256, 10, 1)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.bn2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.pool3(x)

        # print(x.shape)
        x = self.bn3(x)
        # # print(x.shape)
        # x = self.conv4(x)
        # # print(x.shape)
        # x = self.pool4(x)
        # # print(x.shape)
        # xm = self.bn4(x)

        # x = self.conv5(x)
        # x = self.pool5(x)
        # xm = self.bn5(x)
        # xm = self.bn3_mag(xm)
        # print(xm.shape)

        x_checkpoint = x.view([x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        xm = self.fc1(x_checkpoint)
        xm = self.relu(self.fc1bn(xm))
        xm = self.dropout(xm)
        xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm, x_checkpoint





class Net_vanilla_cnn_small(nn.Module):
    def __init__(self):
        super(Net_vanilla_cnn_small, self).__init__()

        kernel_sizes = [7, 3, 3]
        pads = (np.array(kernel_sizes) - 1) / 2
        pads = pads.astype(int)
        layers = [30, 60]
        self.post_filter = False
        # network layers
        self.conv1 = ConvolutionLayer(1, layers[0], [kernel_sizes[0], kernel_sizes[0]], stride=1, padding=pads[0])
        self.conv2 = ConvolutionLayer(layers[0], layers[1], [kernel_sizes[1], kernel_sizes[1]], stride=1,
                                      padding=pads[1])
        # self.conv5 = ConvolutionLayer(layers[3], layers[4], [kern1el_sizes[2], kernel_sizes[2]], stride=1, padding=pads[2])

        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(layers[0])
        self.pool2 = nn.MaxPool2d(kernel_size=(8, 8))
        self.bn2 = nn.BatchNorm2d(layers[1])
        # self.pool5 = nn.MaxPool2d(2)
        # self.bn5 = nn.BatchNorm2d(layers[4])
        self.fc1 = nn.Conv2d(layers[1], 100, 1)
        self.fc1bn = nn.BatchNorm2d(100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.7)
        self.fc2 = nn.Conv2d(100, 10, 1)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x_checkpoint = self.bn2(x)
        # print(x.shape)
        # print(x.shape)
        # x_checkpoint = self.bn3(x)
        # # print(x.shape)
        # x = self.conv4(x)
        # # print(x.shape)
        # x = self.pool4(x)
        # # print(x.shape)
        # xm = self.bn4(x)

        # x = self.conv5(x)
        # x = self.pool5(x)
        # xm = self.bn5(x)
        # xm = self.bn3_mag(xm)
        # print(xm.shape)

        xm = x_checkpoint.view(
            [x_checkpoint.shape[0], x_checkpoint.shape[1] * x_checkpoint.shape[2] * x_checkpoint.shape[3], 1, 1])
        xm = self.fc1(xm)
        xm = self.relu(self.fc1bn(xm))
        # xm = self.dropout(xm)
        xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm, x_checkpoint



class Net_vanilla_cnn_smaller(nn.Module):
    def __init__(self):
        super(Net_vanilla_cnn_smaller, self).__init__()

        kernel_sizes = [7, 3, 3]
        pads = (np.array(kernel_sizes) - 1) / 2
        pads = pads.astype(int)
        layers = [30]
        self.post_filter = False
        # network layers
        self.conv1 = ConvolutionLayer(1, layers[0], [kernel_sizes[0], kernel_sizes[0]], stride=1, padding=pads[0])
        # self.conv5 = ConvolutionLayer(layers[3], layers[4], [kern1el_sizes[2], kernel_sizes[2]], stride=1, padding=pads[2])

        self.pool1 = nn.MaxPool2d(kernel_size=(14, 14))
        self.bn1 = nn.BatchNorm2d(layers[0])
        # self.pool5 = nn.MaxPool2d(2)
        # self.bn5 = nn.BatchNorm2d(layers[4])
        self.fc1 = nn.Conv2d(layers[0] * 4, 30, 1)
        self.fc1bn = nn.BatchNorm2d(30)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.7)
        self.fc2 = nn.Conv2d(30, 10, 1)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x_checkpoint = self.bn1(x)
        # print(x.shape)
        # print(x.shape)
        # x_checkpoint = self.bn3(x)
        # # print(x.shape)
        # x = self.conv4(x)
        # # print(x.shape)
        # x = self.pool4(x)
        # # print(x.shape)
        # xm = self.bn4(x)

        # x = self.conv5(x)
        # x = self.pool5(x)
        # xm = self.bn5(x)
        # xm = self.bn3_mag(xm)
        # print(xm.shape)

        xm = x_checkpoint.view(
            [x_checkpoint.shape[0], x_checkpoint.shape[1] * x_checkpoint.shape[2] * x_checkpoint.shape[3], 1, 1])
        xm = self.fc1(xm)
        xm = self.relu(self.fc1bn(xm))
        # xm = self.dropout(xm)
        xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm, 0

