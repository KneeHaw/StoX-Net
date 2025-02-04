import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import src.toyModel as toyModel
import src.utilities as utilities
import dill
import src.argsparser as argsparser
from src.resnet20 import resnet20

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
args = argsparser.get_parser().parse_args()
best_prec1 = 0

if not args.evaluate:
    Log_Vals = open(args.logs_save_dir, 'w')


if not os.path.exists(args.batch_log):
    with open(args.batch_log, 'wb') as f:
        pass


def main():
    start_time = time.time()

    global args, best_prec1

    print(f"Time @ args: {time.time() - start_time}")

    # Get model parameters from argument parser
    stox_params_conv = [args.conv_num_ab, args.conv_num_wb, args.conv_ab_sw,
                        args.conv_wb_sw, args.subarray_size, args.sensitivity, args.conv_time_steps]
    stox_params_linear = [args.linear_num_ab, args.linear_num_wb, args.linear_ab_sw,
                          args.linear_wb_sw, args.subarray_size, args.sensitivity, args.linear_time_steps]
    model_params = [args.MTJops_Exit, args.stox]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Build model according to params

    model = None
    if args.model == 'toymodel':
        if args.dataset == 'MNIST':
            model = toyModel.toy_model_mnist(stox_params_conv, stox_params_linear, model_params).to(device)
        elif args.dataset == 'CIFAR10':
            model = toyModel.toy_model_cifar10(stox_params_conv, stox_params_linear, model_params).to(device)
        else:
            raise NotImplementedError

    elif args.model == 'resnet20':
        if args.dataset == 'MNIST':
            model = resnet20(stox_params_conv, stox_params_linear, model_params, 1).to(device)
        elif args.dataset == 'CIFAR10':
            model = resnet20(stox_params_conv, stox_params_linear, model_params, 3).to(device)
    else:
        raise NotImplementedError

    print(f"Time @ model load: {time.time()-start_time}")

    # optionally resume from a checkpoint
    if args.resume and args.evaluate:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = False

    train_loader, val_loader = utilities.get_loaders(dataset=args.dataset, batch_size=args.batch_size, workers=4)
    print(f"Time @ data load: {time.time()-start_time}")

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)

    # print(model.modules)  # Print all model components/layers
    start_train = time.time()

    # train for one epoch
    print(f"Time @ training start: {time.time()-start_time}")

    if args.evaluate is True:
        validate(val_loader, model, criterion)
        exit()

    # begin epoch training loop
    for epoch in range(args.start_epoch, args.epochs):
        start_epoch = time.time()

        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if is_best:
            save_checkpoint({
                'model': model,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.model_save_dir))
        print("Epoch Time: " + str(time.time() - start_epoch))
        Log_Vals.write(str(prec1) + '\n')
    print("Total Time: " + str(time.time() - start_train))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch+1, i+1, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    Log_Vals.write(str(epoch+1) + ', ' + str(losses.avg) + ', ' + str(top1.avg) + ', ')


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        last_finished_batch = args.skip_to_batch
        if i < last_finished_batch:
            continue

        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())

        # compute output
            output = model(input_var)

            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if args.print_batch_info:
            string = f"Batch {i+1} Prec@1 = {prec1:.2f}%, Avg = {top1.avg:.2f}%"
            print(string)
            with open(args.batch_log, 'a') as log_file:
                log_file.write(string)


    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save the training model"""
    torch.save(state, filename, pickle_module=dill)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
