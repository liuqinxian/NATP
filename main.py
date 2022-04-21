import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random

from config import DefaultConfig, Logger
from data import cubeDataset
from model import CubeBase

args = DefaultConfig()

sys.stdout = Logger(args.save_path + '/record.txt')

# random seed
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)


def train(**kwargs):
    args.parse(kwargs)

    train_dataset = cubeDataset(train=True)
    test_dataset = cubeDataset(train=False)
    train_iter = torch.utils.data.DataLoader(train_dataset, args.batch_size)
    test_iter = torch.utils.data.DataLoader(test_dataset, args.batch_size)

    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.BCELoss()

    model = CubeBase()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    loss_min = 1.0
    for epoch in range(args.epochs):
        model.train()        
        loss_sum, n = 0.0, 0
        for x, y in train_iter:
            print('\n')
            print(y)
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            y_pred = model(x).squeeze(1)
            # print(y_pred.shape)
            # print(y.shape)
            # print("train:", y_pred)
            # print("train:", y)
            print('pred:', y_pred)
            # loss = loss_func(y_pred, y.long())
            loss = loss_func(y_pred, y.to(torch.float32))
            print(loss)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n += 1
        scheduler.step()


if __name__ == '__main__':
    import fire
    fire.Fire()