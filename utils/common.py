import torch
import torch.nn as nn
import shutil
import torch.nn.functional as F
import numpy as np
import os

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


def save_checkpoint(state, is_best, model):
    latest_check = 'checkpoint/' + model + '_latest.pth.tar'
    best_check = 'checkpoint/' + model + '_best.pth.tar' 
    
    torch.save(state, latest_check)
    if is_best:
        shutil.copyfile(latest_check, best_check)

def save_checkpoint_epoch(state, epoch, arch):
    torch.save(state, 'checkpoint/' + arch + '_' + str(epoch)+'.pth.tar')
    
def resume(model, check, arch):
    if check == 'latest':
        checkpoint_ = 'checkpoint/' + arch + '_latest.pth.tar' 
    if check == 'best':
        checkpoint_ = 'checkpoint/' + arch + '_best.pth.tar' 
    
    if os.path.isfile(checkpoint_):
        print("=> loading checkpoint '{}'".format(checkpoint_))
        checkpoint = torch.load(checkpoint_)
#        best_metric = checkpoint['best_metric']
#        loss = checkpoint['loss']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'"
              .format(checkpoint_))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_))
        
      
    

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, logits, targets):
        return self.nll_loss(F.log_softmax(logits), targets)

class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs        = F.sigmoid(logits)
        probs_flat   = probs.view (-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self):  #weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()


    def forward(self, logits, targets):

        probs = F.sigmoid(logits)
        num = targets.size(0)
        m1  = probs.view(num,-1)
        m2  = targets.view(num,-1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1- score.sum()/num
        return score

def dice_loss(preds, trues, weight=None, is_average=True):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = 2. * (intersection + 1) / (preds.sum(1) + trues.sum(1) + 1) # soft dice
    #scores = 2. * (intersection) / (preds.sum(1) + trues.sum(1) + 1e-15)

    if is_average:
        score = scores.sum()/num
        return torch.clamp(score, 0., 1.)
    else:
        return scores

def dice_clamp(preds, trues, is_average=True):
    preds = torch.round(preds)
    return dice_loss(preds, trues, is_average=is_average)

class DiceLoss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target, weight=None):
        return 1-dice_loss(F.sigmoid(input), target, weight=weight, is_average=self.size_average)

class BCEDiceLoss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.dice = DiceLoss(size_average=size_average)

    def forward(self, input, target, weight=None):
        return nn.modules.loss.BCEWithLogitsLoss(size_average=self.size_average, weight=weight)(input, target) + self.dice(input, target, weight=weight)

class UNet11_Loss: 
    """Vladimir’s Approach, same as BCEDiceLoss (different eps settings)
    """
    def __init__(self, dice_weight=1):
        self.nll_loss = nn.BCELoss()
        self.dice_weight = dice_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.dice_weight:
            eps = 1e-15
            dice_target = (targets == 1).float()
            dice_output = outputs
            intersection = (dice_output * dice_target).sum()
            union = dice_output.sum() + dice_target.sum() + eps

            loss -= torch.log(2 * intersection / union)

        return loss
    
def adjust_lr(optimizer, epoch, init_lr=0.1, num_epochs_per_decay=10, lr_decay_factor=0.1):
    lr = init_lr * (lr_decay_factor ** (epoch // num_epochs_per_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Vladimir’s Approach
def cyclic_lr(optimizer, epoch, init_lr=1e-4, num_epochs_per_cycle=5, cycle_epochs_decay=2, lr_decay_factor=0.5):
    epoch_in_cycle = epoch % num_epochs_per_cycle
    lr = init_lr * (lr_decay_factor ** (epoch_in_cycle // cycle_epochs_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    
losses = {
    'BCELoss2d': BCELoss2d,
    'CrossEntropyLoss2d': CrossEntropyLoss2d,
    'SoftDiceLoss': SoftDiceLoss,
    'BCEDiceLoss': BCEDiceLoss,
    'UNet11_Loss': UNet11_Loss
} # more complex loss: https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/asanakoy/losses.py


def metric(output, mask_var):
    return np.zeros(output.size(0))

def submit():
    pass