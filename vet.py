# This code is part of [Semi-supervised Learning for Labeling Veterinary X-Ray Data]
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# Portions of this code are derived from the DARP project: https://github.com/bbuing9/DARP
# Copyright (c) 2020 bbuing9, Licensed under the MIT License.

from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from scipy import optimize

import models.wrn as models
from arguments import parse_args
from dataset import get_cifar10, get_cifar100, get_stl10, get_vetdata, get_vet_original
from training_functions import trains
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from common import validate, validate_vet, estimate_pseudo, opt_solver, make_imb_data, save_checkpoint, SemiLoss, WeightEMA, interleave

import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault
import numpy as np
from collections import Counter

args = parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

print(torch.__version__)
print(torch.cuda.is_available())

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

# best test accuracy; 초기값 0으로 설정
best_acc = 0  

# num class; 7개로 지정
args.num_class = 7

# semi_method
if args.semi_method == 'remix':
    args.lambda_u = 1.5


def main():
    global best_acc

    # Directory to output the result
    args.out = args.dataset + '@N_' + str(args.num_max) + '_r_'
    # ratio_l과 ratio_u의 동일 여부에 따라 directory명 수정
    if args.imb_ratio_l == args.imb_ratio_u:
        args.out += str(args.imb_ratio_l) + '_' + args.semi_method
    else:
        args.out += str(args.imb_ratio_l) + '_' + str(args.imb_ratio_u) + '_' + args.semi_method
    # darp 적용할 시 directory명 수정
    if args.darp:
        args.out += '_darp_alpha' + str(args.alpha) + '_iterT' + str(args.iter_T) + '_warm' + str(args.warm) + '_epoch' + str(args.epochs)
    else:
        args.out += '_epoch' + str(args.epochs)

    # 그 directory 기존에 없으면 새로 만들기
    if not os.path.isdir(args.out):
        mkdir_p(args.out)
    # 디렉토리명 확인
    print(args.out)
    
    # # labeled/unlabeled data 개수 정하기
    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, args.num_class, args.imb_ratio_l)
    U_SAMPLES_PER_CLASS = make_imb_data(args.ratio * args.num_max, args.num_class, args.imb_ratio_u)
    N_SAMPLES_PER_CLASS_T = torch.Tensor(N_SAMPLES_PER_CLASS)
    # num_max = 500, num_class = 7, imb_ratio_l = 100, ratio = 4, imb_ratio_u = 1로 지정할 것
    # N_SAMPLES_PER_CLASS = make_imb_data(500, 7, 100)
    # U_SAMPLES_PER_CLASS = make_imb_data(4 * 500, 7, 1)

    if args.dataset == 'vet':
        train_labeled_set, train_unlabeled_set, test_set = get_vet_original('/home/aailab/yund02/ML/data/bonedata',
                                                                        N_SAMPLES_PER_CLASS,
                                                                        U_SAMPLES_PER_CLASS,
                                                                        args.out)

    print(len(train_labeled_set))
    print(len(train_unlabeled_set))
    print(len(test_set))

    # batch training을 위한 dataloader 생성
    labeled_trainloader = data.DataLoader(train_labeled_set,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=4, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set,
                                            batch_size=args.batch_size,
                                            shuffle=True, num_workers=4, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    print("==> creating WRN-28-2")
    def create_model(ema=False):
        # models.WRN에 선언한 class를 바탕으로 model 구조 지정
        model = models.WRN(2, args.num_class)
        model = model.cuda()

        # ema를 True로 중 경우, model parameter에 대해 detach 시행(?)
        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    # CUDA Deep Neural Network library (cuDNN)이 연산 그래프를 최적화하기 위해 알고리즘을 동적으로 선택할 수 있도록 허용
    cudnn.benchmark = True
    # training할 parameter 개수 확인
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # semi-supervised learning을 위한 loss function 지정
    # supervised / unsupervised learning에 대한 loss를 별도로 계산함
    train_criterion = SemiLoss()

    # loss function, 그냥 model optimizer, ema model optimizer, start_epoch 지정 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema_optimizer= WeightEMA(model, ema_model, lr=args.lr, alpha=args.ema_decay)
    start_epoch = 0

    # Resume; 이전에 학습되던 것 이어받을지, 새로 시작할지 지정
    title = 'Imbalanced' + '-' + args.dataset + '-' + args.semi_method
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Test Loss', 'Test Acc.(top1)', 'Test Acc.(top2)', 'Test GM.'])

    test_accs = []
    test_gms = []

    # Default values for MixMatch and DARP
    emp_distb_u = torch.ones(args.num_class) / args.num_class
    pseudo_orig = torch.ones(len(train_unlabeled_set.data), args.num_class) / args.num_class
    pseudo_refine = torch.ones(len(train_unlabeled_set.data), args.num_class) / args.num_class

    # Main function
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        # Unlabeled data의 distribution을 지정하는 부분

        # est; 이 인자를 true로 줄 경우, unlabeled data의 estimated distribution을 사용
        # Use the estimated distribution of unlabeled data
        if args.est:
            if args.dataset == 'cifar10':
                est_name = './estimation/cifar10@N_1500_r_{}_{}_estim.npy'.format(args.imb_ratio_l, args.imb_ratio_u)
            else:
                est_name = './estimation/stl10@N_450_r_{}_estim.npy'.format(args.imb_ratio_l)
            est_disb = np.load(est_name)
            target_disb = len(train_unlabeled_set.data) * torch.Tensor(est_disb) / np.sum(est_disb)
        
        # est; 이 인자를 false로 줄 경우, labeled data로부터 추론된 distribution을 사용
        # Use the inferred distribution with labeled data
        else:
            target_disb = N_SAMPLES_PER_CLASS_T * len(train_unlabeled_set.data) / sum(N_SAMPLES_PER_CLASS)

        # model 학습
        train_loss, train_loss_x, train_loss_u, emp_distb_u, pseudo_orig, pseudo_refine = trains(args, labeled_trainloader,
                                                                                                unlabeled_trainloader,
                                                                                                model, optimizer,
                                                                                                ema_optimizer,
                                                                                                train_criterion,
                                                                                                epoch, use_cuda,
                                                                                                target_disb, emp_distb_u,
                                                                                                pseudo_orig, pseudo_refine)

        # Evaluation part
        test_loss, test_acc, test_acc2, test_cls, test_gm = validate_vet(test_loader, ema_model, criterion, use_cuda,
                                                          mode='Test Stats', num_class=args.num_class)

        # Append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, test_loss, test_acc, test_acc2, test_gm])

        # Save models
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch + 1, args.out)
        test_accs.append(test_acc)
        test_gms.append(test_gm)

    logger.close()

    # Print the final results
    print('Mean bAcc:', end='')
    print(torch.mean(torch.tensor(test_accs[-20:])).item())

    print('Mean GM:', end='')
    print(torch.mean(torch.tensor(test_gms[-20:])).item())

    print('Name of saved folder:')
    print(args.out)

if __name__ == '__main__':
    main()