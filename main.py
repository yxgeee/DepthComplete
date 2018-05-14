from __future__ import print_function, absolute_import
import argparse
import os,sys
import shutil
import time
import math
import os.path as osp
import numpy as np

fileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, fileDir)

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.optim import lr_scheduler

import models
from models.SparseConvNet import *
import datasets
from datasets.depth_loader import DepthDataset
from util.utils import AverageMeter, Logger, save_checkpoint, Evaluate
from util.criterion import init_criterion, get_criterions

parser = argparse.ArgumentParser(description='PyTorch Depth Completion Training')
parser.add_argument('--dataset', default='kitti', choices=datasets.get_names(),
                    help='name of dataset')
parser.add_argument('--data-root', default='./data', help='root path to datasets')
parser.add_argument('--save-root', default='./checkpoints', help='root path to datasets')
parser.add_argument('--arch', '-a', metavar='ARCH', default='sparseconv',
                    choices=models.get_names(),
                    help='model architecture: ' +
                        ' | '.join(models.get_names()) +
                        ' (default: sparseconv)')
parser.add_argument('--tag', default='', help='tag in save path')
parser.add_argument('--height', type=int, default=352,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=1216,
                    help="width of an image (default: 128)")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--step-size', default=20, type=int, metavar='N',
                    help='stepsize to decay learning rate (>0 means this is enabled)')
parser.add_argument('--eval-step', default=20, type=int, metavar='N',
                    help='stepsize to evaluate')
parser.add_argument('--criterion', default='masked_mseloss', choices=get_criterions(), help="type of criterion")
parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--optim', default='adam', type=str)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--gpu-ids', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

def init_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optim: {}".format(optim))

def main():
    global args
    args = parser.parse_args()
    best_epoch = 0
    best_pipline = np.inf

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    cudnn.benchmark = True

    if args.tag:
        save_name = args.arch+'_'+args.criterion + '_'+args.tag
    else:
        save_name = args.arch+'_'+args.criterion
    args.save_root = osp.join(args.save_root, args.dataset, save_name)
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_root, 'log_train.txt'))
    else:
        if args.resume:
            args.save_root = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(args.save_root, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.init_model(name=args.arch)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    # optionally resume from a checkpoint
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_epoch = checkpoint['epoch']
            best_pipline = checkpoint['rmse']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    model = torch.nn.DataParallel(model).cuda()
    criterion = init_criterion(args.criterion).cuda()
    if args.step_size > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    print("Initializing dataset {}".format(args.dataset))
    dataset = datasets.init_dataset(args.dataset, root=osp.join(args.data_root,args.dataset))

    # Data loading code
    train_dataset = DepthDataset(osp.join(args.data_root,args.dataset), dataset.trainset, args.height, args.width)
    val_dataset = DepthDataset(osp.join(args.data_root,args.dataset), dataset.valset, 352, 1216, isVal=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        print("Evaluate only")
        validate(val_loader, model, criterion)
        return

    print("==> Start training")
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, optimizer, criterion, epoch)

        if args.step_size > 0: scheduler.step()

        # evaluate on validation set
        if args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.epochs:
            print("==> Test")
            rmse = validate(val_loader, model, criterion)

            is_best = rmse < best_pipline
            if is_best:
                best_epoch = epoch + 1
                best_pipline = rmse
            save_checkpoint({
                'epoch': epoch + 1,
                'rmse': rmse,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best, osp.join(args.save_root, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    print("==> Minimal RMSE {:.6f}, achieved at epoch {}".format(best_pipline, best_epoch))


def train(train_loader, model, optimizer, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    rmses = AverageMeter()
    results = Evaluate()

    model.train()

    end = time.time()
    for i, (input, target, scale) in enumerate(train_loader):
        # measure data loading time
        input, target, scale = input.cuda(), target.cuda(), scale.float().cuda()
        # compute output
        output = model(input)

        loss = criterion(output, target)
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output * scale.view(input.size(0),1,1,1).expand_as(output)
        target = target * scale.view(input.size(0),1,1,1).expand_as(target)
        results.evaluate(output, target)
        rmses.update(results.rmse, input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                  'RMSE {rmse.val:.6f} ({rmse.avg:.6f})'.format(
                   epoch+1, i+1, len(train_loader), batch_time=batch_time,
                   loss=losses, rmse=rmses))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    results = Evaluate()
    rmses = AverageMeter()
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, scale) in enumerate(val_loader):
            input, target, scale = input.cuda(), target.cuda(), scale.float().cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            output = output * scale.view(input.size(0),1,1,1).expand_as(output)
            target = target * scale.view(input.size(0),1,1,1).expand_as(target)
            results.evaluate(output, target)
            rmses.update(results.rmse, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                      'RMSE {rmse.val:.6f} ({rmse.avg:.6f})'.format(
                       i, len(val_loader), batch_time=batch_time,
                       loss=losses, rmse=rmses))

        print(' * RMSE {rmse.avg:.6f}'.format(rmse=rmses))

    return rmses.avg

if __name__ == '__main__':
    main()
