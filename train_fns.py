
import torch
from torch.utils import data
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from Network import *
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy

import cv2
import torch.nn.functional as F
from Nets import *
import numpy as np
import sys, os
from utils import Dataset, load_dataset, Dataset_NearestNeighbor
import torchvision.transforms.functional as TF

import random

def train_network_n2n_regularization_2STEP_noisy_labels(net, net_small, net_smaller, epoch_big,
                                                        epoch_small, trainloader, init_rate,total_epochs, weight_decay, reg_lambda=0.1):
    net = net
    net = net.cuda()
    net = net.train()
    net_smaller = net_smaller.cuda()
    net_small = net_small.cuda()
    optimizer = optim.SGD(net.parameters(), lr=init_rate, momentum=0.9, weight_decay=weight_decay)
    optimizer_small = optim.SGD(net_small.parameters(), lr=0.02, momentum=0.9, weight_decay=0.0005)
    optimizer_smaller = optim.SGD(net_smaller.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], last_epoch=-1)

    criterion = nn.CrossEntropyLoss()
    net.post_filter = False
    init_epoch = 0
    for epoch in range(init_epoch):

        # print("Time for one epoch:",time.time()-s)
        # s = time.time()

        # scheduler.step()
        print('epoch: ' + str(epoch))

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer_small.zero_grad()
            allouts = net_small(inputs)

            loss = criterion(allouts[0], labels.long())
            loss.backward()
            optimizer_small.step()

    for epoch in range(init_epoch):

        print('epoch: ' + str(epoch))

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer_smaller.zero_grad()
            allouts = net_smaller(inputs)

            loss = criterion(allouts[0], labels.long())
            loss.backward()
            optimizer_smaller.step()

    if not os.path.exists('networks_mnist_noisy_asym0.45_latest'):
        os.makedirs('networks_mnist_noisy_asym0.45_latest')

    for epoch in range(total_epochs):

        scheduler.step()

        print('epoch: ' + str(epoch))

        for e_big in range(epoch_big):

            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                optimizer_small.zero_grad()
                outputs, feats = net(inputs)
                outputs_small, feats = net_small(inputs.detach())
                square_loss = F.mse_loss(outputs_small, outputs)

                if epoch == 0:
                    loss = criterion(outputs, labels.long())
                else:
                    losses = [criterion(outputs, labels.long()), reg_lambda * square_loss]
                    loss = sum(losses)

                loss.backward()
                optimizer.step()

        if reg_lambda > 0.0:
            # print("lambda mode")

            for e_small in range(epoch_small):

                for i, data in enumerate(trainloader, 0):
                    # get the inputs
                    inputs, labels = data
                    inputs = inputs.cuda()

                    optimizer.zero_grad()
                    optimizer_small.zero_grad()
                    optimizer_smaller.zero_grad()

                    outputs, feats = net(inputs.detach())
                    outputs_small, feats_small = net_small(inputs)
                    outputs_smaller, feats_smaller = net_smaller(inputs)
                    square_loss = F.mse_loss(outputs_small, outputs)
                    smaller_square_loss = F.mse_loss(outputs_smaller, outputs_small)

                    if epoch == 0:
                        losses = [square_loss]

                    else:
                        losses = [square_loss, smaller_square_loss * (0.1)]

                    loss = sum(losses)
                    loss.backward()

                    optimizer_small.step()

            for e_small in range(epoch_small):

                for i, data in enumerate(trainloader, 0):
                    # get the inputs
                    inputs, labels = data
                    inputs = inputs.cuda()

                    optimizer.zero_grad()
                    optimizer_small.zero_grad()
                    optimizer_smaller.zero_grad()

                    outputs_small, feats_small = net_small(inputs.detach())
                    outputs_smaller, feats_smaller = net_smaller(inputs)
                    square_loss = F.mse_loss(outputs_smaller, outputs_small)
                    losses = [square_loss]

                    loss = sum(losses)
                    loss.backward()

                    optimizer_smaller.step()

        # print('break')

    net = net.eval()
    net_small = net_small.eval()

    return net

if __name__ == "__main__":

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # --------------- Hyperparams for MNIST------------------#
    dataset_name = 'MNIST'
    val_splits = 1
    training_size = 60000
    training_sizes = [60000]
    test_size = 10000
    batch_size = 256
    init_rate = 0.02
    decay_normal = 0.001
    step_size = 10
    gamma = 0.7
    total_epochs = 100
    post_filter_epochs = 30
    filter_thres = 1

    epoch_big = 3
    epoch_small = 1
    reg_lambda = 10.0
    Networks_to_train = [Net_vanilla_cnn_mnist_regularized(), Net_vanilla_cnn_small(),
                         Net_vanilla_cnn_smaller()]

