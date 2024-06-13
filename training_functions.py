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

from scipy import optimize
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from common import validate, estimate_pseudo, opt_solver, make_imb_data, save_checkpoint, SemiLoss, WeightEMA, interleave, linear_rampup

# semi_method를 무엇으로 줬느냐에 따라 train function 다르게 불러옴
def trains(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda,
          target_disb, emp_distb_u, pseudo_orig, pseudo_refine):

    if args.semi_method == 'mix':
        return train_mix(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch,
                  use_cuda, target_disb, emp_distb_u, pseudo_orig, pseudo_refine)
    elif args.semi_method == 'remix':
        return train_remix(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch,
                    use_cuda, target_disb, emp_distb_u, pseudo_orig, pseudo_refine)
    elif args.semi_method == 'fix':
        return train_fix(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch,
                  use_cuda, target_disb, emp_distb_u, pseudo_orig, pseudo_refine)
    else:
        raise Exception('Wrong type of semi-supervised method (Please select among |mix|remix|fix|)')

def train_mix(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch,
              use_cuda, target_disb, emp_distb_u, pseudo_orig, pseudo_refine):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x, _ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _ = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _, idx_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _, idx_u = unlabeled_train_iter.next()

        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, args.num_class).scatter_(1, targets_x.view(-1,1), 1)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, inputs_u2  = inputs_u.cuda(), inputs_u2.cuda()

        # Generate the pseudo labels by aggregation and sharpening
        with torch.no_grad():
            outputs_u, _ = model(inputs_u)
            outputs_u2, _ = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2

            pt = p ** (1 / args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)

            # Update the saved predictions with current one
            p = targets_u
            pseudo_orig[idx_u, :] = p.data.cpu()
            pseudo_orig_backup = pseudo_orig.clone()

            # Applying DARP
            if args.darp and epoch > args.warm:
                if batch_idx % args.num_iter == 0:
                    # Iterative normalization
                    targets_u, weights_u = estimate_pseudo(target_disb, pseudo_orig, args.num_class, args.alpha)
                    scale_term = targets_u * weights_u.reshape(1, -1)
                    pseudo_orig = (pseudo_orig * scale_term + 1e-6) \
                                      / (pseudo_orig * scale_term + 1e-6).sum(dim=1, keepdim=True)

                    if args.dataset == 'stl10' or args.dataset == 'cifar100':
                        opt_res = opt_solver(pseudo_orig, target_disb, args.iter_T, 0.3)
                    else:
                        opt_res = opt_solver(pseudo_orig, target_disb, args.iter_T, 0.1)

                    # Updated pseudo-labels are saved
                    pseudo_refine = opt_res

                    print(pseudo_refine.sum(dim=0))
                    print(target_disb)

                    # Select
                    targets_u = opt_res[idx_u].detach().cuda()
                    pseudo_orig = pseudo_orig_backup
                else:
                    # Using previously saved pseudo-labels
                    targets_u = pseudo_refine[idx_u].cuda()

        # Mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(args.mix_alpha, args.mix_alpha)
        l = max(l, 1-l)
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])[0]]
        for input in mixed_input[1:]:
            logits.append(model(input)[0])

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(args, logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
                              epoch+batch_idx/args.val_iteration)
        Lu *= w
        loss = Lx + Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg, emp_distb_u, pseudo_orig, pseudo_refine)


def train_remix(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch,
                use_cuda, target_disb, emp_distb_u, pseudo_orig, pseudo_refine):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_r = AverageMeter()
    losses_e = AverageMeter()
    ws = AverageMeter()
    end = time.time()
    
    # epoch 500 & validation을 500번 마다 한번씩
    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    # model train mode로 설정
    model.train()
    for batch_idx in range(args.val_iteration):
        # labeled_train_iter에서 input값, target값 받아옴
        try:
            inputs_x, targets_x, _ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _ = labeled_train_iter.next()

        # unlabeled_train_iter에서 input값들(?), index값 받아옴 (-1로 설정한 index는 불러오지 않는다!)
        try:
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = unlabeled_train_iter.next()

        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)
        # Transform label to one-hot; label을 one hot vector로 수정
        targets_x = torch.zeros(batch_size, args.num_class).scatter_(1, targets_x.view(-1,1), 1)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, inputs_u2, inputs_u3  = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()

        # Rotate images
        temp = []                                                         # 회전된 이미지를 저장할 리스트 생성
        targets_r = torch.randint(0, 4, (inputs_u2.size(0),)).long()      # 0~3까지의 정수 중 무작위로 선택된 값으로 텐서 구성 -> 회전할 때 사용할 각도
        for i in range(inputs_u2.size(0)):
            inputs_rot = torch.rot90(inputs_u2[i], targets_r[i], [1, 2]).reshape(1, 3, 256, 256)
            #inputs_rot = torch.rot90(inputs_u2[i], targets_r[i], [1, 2]).reshape(1, 3, 32, 32)  # 이미지를 회전시킨 후, reshaping
            temp.append(inputs_rot)                                                             # 그 결과를 append
        inputs_r = torch.cat(temp, 0)                                                           # temp에 저장된 모든 이미지를 결합해서 하나의 tensor로 만듦 (image batch)
        targets_r = torch.zeros(batch_size, 4).scatter_(1, targets_r.view(-1, 1), 1) # 회전된 이미지에 대한 label생성 (길이가 4인 영행렬 만들고, 각 행에 해당하는 rotation index에 1 할당)
        inputs_r, targets_r = inputs_r.cuda(), targets_r.cuda(non_blocking=True)       # 생성된 rotated image, rotation label을 GPU로 이동

        # Generate the pseudo labels
        with torch.no_grad():
            # unlabel data를 model에 넣어서 output을 얻고
            outputs_u, _ = model(inputs_u)
            # softmax 함수를 취해서 각 class에 대한 확률 분포 얻기
            p = torch.softmax(outputs_u, dim=1)

            # Tracking the empirical distribution on the unlabeled samples (ReMixMatch)
            real_batch_idx = batch_idx + epoch * args.val_iteration
            # 첫 번째 배치일 경우; unlabeled sample에 대한 확률분포의 평균 계산 후, 이를 emp_distb_u에 저장
            if real_batch_idx == 0:
                emp_distb_u = p.mean(0, keepdim=True)
            # 각 epoch의 128번째 batch까지; 이전 unlabeled sample에 대한 확률분포의 평균과 현재 배치의 평균을 연결해서 emp_distb_u에 update
            elif real_batch_idx // 128 == 0:
                emp_distb_u = torch.cat([emp_distb_u, p.mean(0, keepdim=True)], 0)
            # 각 epoch의 128번째 batch 이후부터; 최근 128개의 batch만을 이용하여 새로운 unlabeled sample에 대한 확률분포 평균이 계산되면 update
            else:
                emp_distb_u = emp_distb_u[-127:]
                emp_distb_u = torch.cat([emp_distb_u, p.mean(0, keepdim=True)], 0)

            # Distribution alignment
            # align 인자가 True일 경우,
            # target_disb(unlabeled dataset에 labeling을 하면, 이 분포를 따라 가야함)와 emp_dist_u(pseudo-labeling을 해서 얻어낸 label의 분포)
            # 가 align 되도록 함
            if args.align:
                pa = p * (target_disb.cuda() + 1e-6) / (emp_distb_u.mean(0).cuda() + 1e-6)
                p = pa / pa.sum(dim=1, keepdim=True) # 조정된 분포를 정규화 함 (각 행을 따라 합계가 1이 되도록 조정)

            # Temperature scailing
            pt = p ** (1 / args.T)     # 하나의 one-hot vector로 분류되도록 Sharpening 하는 과정
            targets_u = (pt / pt.sum(dim=1, keepdim=True)).detach() # 조정된 분포를 정규화하고 backpropagation 대상에서 분리

            # Update the saved predictions with current one
            # 최종 조정된 unlabeled data의 분포를 변수 'p'에 저장
            p = targets_u
            # pseudo label 결과 list 내에서 이번 batch를 구성한 data에 해당하는 index마다 예측한 label로 변경
            pseudo_orig[idx_u, :] = p.data.cpu()
            # pseudo label 결과 backup 해두기
            pseudo_orig_backup = pseudo_orig.clone()

            # Applying DARP; 지정한 warm-up period 이후의 epoch부터 DARP를 적용하여 psuedo label의 quality를 개선함
            if args.darp and epoch > args.warm:
                # 각 epoch내에서, num_iter(기본: 10)마다 darp를 적용
                if batch_idx % args.num_iter == 0:
                    # Iterative normalization
                    # estimate_pseudo함수; target_disb(목표분포), pseudo_orig(pseudo-labeling 결과), class 개수, alpha 입력받으면
                    targets_u, weights_u = estimate_pseudo(target_disb, pseudo_orig, args.num_class, args.alpha) # 추정된 refined pseudo label 결과 + 각 class에 대한 가중치 반환 받음
                    # 추정된 refined pseudo label 결과 + 각 class에 대한 가중치로 -> pseudo label을 update
                    scale_term = targets_u * weights_u.reshape(1, -1)
                    pseudo_orig = (pseudo_orig * scale_term + 1e-6) \
                                      / (pseudo_orig * scale_term + 1e-6).sum(dim=1, keepdim=True)

                    if args.dataset == 'stl10' or args.dataset == 'cifar100':
                        opt_res = opt_solver(pseudo_orig, target_disb, args.iter_T, 0.3)
                    # datset이 cifar10일 경우 이걸 opt_res로 지정
                    # opt_solver함수;
                    else:
                        opt_res = opt_solver(pseudo_orig, target_disb, args.iter_T, 0.1)

                    # Updated pseudo-labels are saved
                    pseudo_refine = opt_res

                    # Select
                    targets_u = opt_res[idx_u].detach().cuda()
                    pseudo_orig = pseudo_orig_backup
                else:
                    # Using previously saved pseudo-labels
                    targets_u = pseudo_refine[idx_u].cuda()

        # Mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2, inputs_u3], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u, targets_u], dim=0)

        l = np.random.beta(args.mix_alpha, args.mix_alpha)
        l = max(l, 1-l)
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])[0]]
        for input in mixed_input[1:]:
            logits.append(model(input)[0])

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(args, logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
                              epoch+batch_idx/args.val_iteration)
        _, logits_r = model(inputs_r)
        Lu *= w
        Lr = -1 * torch.mean(torch.sum(F.log_softmax(logits_r, dim=1) * targets_r, dim=1))
        Lr *= args.w_rot

        # Entropy minimization for unlabeled samples (strong augmented)
        outputs_u2, _ = model(inputs_u2)
        Le = -1 * torch.mean(torch.sum(F.log_softmax(outputs_u2, dim=1) * targets_u, dim=1))
        Le *= args.w_ent * linear_rampup(epoch+batch_idx/args.val_iteration, args.epochs)
        loss = Lx + Lu + Lr + Le

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        losses_r.update(Lr.item(), inputs_x.size(0))
        losses_e.update(Le.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | Loss_r: {loss_r:.4f} | ' \
                      'Loss_e: {loss_e:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    loss_r=losses_r.avg,
                    loss_e=losses_e.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg, emp_distb_u, pseudo_orig, pseudo_refine)


def train_fix(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch,
              use_cuda, target_disb, emp_distb_u, pseudo_orig, pseudo_refine):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x, _ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _ = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = unlabeled_train_iter.next()

        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, args.num_class).scatter_(1, targets_x.view(-1,1), 1)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, inputs_u2, inputs_u3  = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()

        # Generate the pseudo labels
        with torch.no_grad():
            # Generate the pseudo labels by aggregation and sharpening
            outputs_u, _ = model(inputs_u)
            targets_u = torch.softmax(outputs_u, dim=1)

            # Update the saved predictions with current one
            pseudo_orig[idx_u, :] = targets_u.data.cpu()
            pseudo_orig_backup = pseudo_orig.clone()

            # Applying DARP
            if args.darp and epoch > args.warm:
                if batch_idx % args.num_iter == 0:
                    # Iterative normalization
                    targets_u, weights_u = estimate_pseudo(target_disb, pseudo_orig, args.num_class, args.alpha)
                    scale_term = targets_u * weights_u.reshape(1, -1)
                    pseudo_orig = (pseudo_orig * scale_term + 1e-6) \
                                      / (pseudo_orig * scale_term + 1e-6).sum(dim=1, keepdim=True)

                    if args.dataset == 'stl10' or args.dataset == 'cifar100':
                        opt_res = opt_solver(pseudo_orig, target_disb, args.iter_T, 0.3)
                    else:
                        opt_res = opt_solver(pseudo_orig, target_disb, args.iter_T, 0.1)

                    # Updated pseudo-labels are saved
                    pseudo_refine = opt_res

                    # Select
                    targets_u = opt_res[idx_u].detach().cuda()
                    pseudo_orig = pseudo_orig_backup
                else:
                    # Using previously saved pseudo-labels
                    targets_u = pseudo_refine[idx_u].cuda()

        max_p, p_hat = torch.max(targets_u, dim=1)
        p_hat = torch.zeros(batch_size, args.num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)

        select_mask = max_p.ge(args.tau)
        select_mask = torch.cat([select_mask, select_mask], 0).float()

        all_inputs = torch.cat([inputs_x, inputs_u2, inputs_u3], dim=0)
        all_targets = torch.cat([targets_x, p_hat, p_hat], dim=0)

        all_outputs, _ = model(all_inputs)
        logits_x = all_outputs[:batch_size]
        logits_u = all_outputs[batch_size:]

        Lx, Lu = criterion(args, logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], epoch, select_mask)
        loss = Lx + Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg, emp_distb_u, pseudo_orig, pseudo_refine)
