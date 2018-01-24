import torch
import torch.nn as nn
import shutil
from utils import functional_newest as F
import torch.nn.functional as Fn

import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.morphology import closing, opening, disk
import pandas as pd
from PIL import Image

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
        probs        = Fn.sigmoid(logits)
        probs_flat   = probs.view (-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self):  #weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()


    def forward(self, logits, targets):

        probs = Fn.sigmoid(logits)
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
#    scores = 2. * (intersection) / (preds.sum(1) + trues.sum(1) + 1e-15)

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
        return -dice_loss(Fn.sigmoid(input), target, weight=weight, is_average=self.size_average)
        #return 1-dice_loss(Fn.sigmoid(input), target, weight=weight, is_average=self.size_average)

class BCEDiceLoss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.dice = DiceLoss(size_average=size_average)

    def forward(self, input, target, weight=None):
        return nn.modules.loss.BCEWithLogitsLoss(size_average=self.size_average, weight=weight)(input, target)*0.5 + self.dice(input, target, weight=weight)
        #return nn.modules.loss.BCEWithLogitsLoss(size_average=self.size_average, weight=weight)(input, target) + self.dice(input, target, weight=weight)

'''
BCEDiceLoss

https://www.kaggle.com/takuok/keras-generator-starter-lb-0-326
Epoch 1/10
101/100 [==============================] - 105s 1s/step - loss: -0.6069 - mean_iou: 0.6256 - val_loss: -0.7858 - val_mean_iou: 0.7488
Epoch 2/10
101/100 [==============================] - 95s 942ms/step - loss: -0.8007 - mean_iou: 0.7810 - val_loss: -0.8263 - val_mean_iou: 0.8032

Humm...
Here is 10 epochs training log and LB score 0.302.
'''

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


def plot_tensor(inp2):
    """Imshow for Tensor. BCHW"""
    mean = np.array([0.1707, 0.1552, 0.1891])
    std = np.array([0.2635, 0.2432, 0.2959])
    inp2 = inp2[0,:,:,:]
    inp2 = inp2.numpy().transpose((1, 2, 0)) # hwc
    print(inp2.shape)
    inp2 = std*inp2 + mean
    print(inp2.max())
    print(inp2.min())
    inp2 = np.clip(inp2, 0, 1)
    plt.figure()
    plt.imshow(inp2)
    plt.pause(2)
    
def plot_tensor_mask(inp):    
    inp = inp[0,:,:,:]
    inp = inp.numpy().transpose((1, 2, 0))
    inp = inp[:,:,0] #HW
    print(inp.shape)
    print(inp.max())
    print(inp.min())
    plt.figure()
    plt.hist(inp)
    plt.pause(2)    
    plt.figure()
    plt.imshow(inp, cmap='gray')
    plt.pause(2)
    
def plot_resized_mask(predicts, img_size, img_name, th):
    """predicts: BCHW tensor.
       img_size: (H, W)
    """
    print("plot_resized_mask")
    resized_PIL = F.resize(F.to_pil_image(predicts[0,:,:,:]), tuple(img_size[0,:]))  # [0, 1] converted into [0, 255]
    resized_np = np.array(resized_PIL) 
    print(tuple(img_size[0,:]))
    print(resized_np.shape)
    print(resized_np.max())
    print(resized_np.min())
    plt.figure()
    plt.hist(resized_np)
    plt.pause(2)  
    plt.figure()
    plt.imshow(resized_np, cmap='gray')
    plt.pause(2)
    
    print(np.unique(resized_np))

#    resized_np = clean_img(resized_np, th)
    plt.figure()
    plt.imshow(resized_np, cmap='gray')
    plt.pause(2)

def run_length_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    run_lengths = ' '.join([str(r) for r in run_lengths])
    return run_lengths

#def clean_img(x, th):
#    """http://blog.csdn.net/haoji007/article/details/52063306
#    """
#    x[x<(th*255)] = 0
#    x[x>=(th*255)] = 255
#    return opening(closing(x, disk(1)), disk(3))

def resize_tensor_2_numpy_and_encoding(predicts, img_size, img_name, th):
    """predicts: BCHW tensor.
       img_size: (H, W)
    """
    ImageId = []
    EncodedPixels = []
    for i in range(predicts.size(0)): #[0 , 1]
        resized_PIL = F.resize(F.to_pil_image(predicts[i,:,:,:]), tuple(img_size[i,:]))  # [0, 1] converted into [0, 255]
        resized_np = np.array(resized_PIL) 
#        print(resized_np.max())
#        print(resized_np.min())
#        print(resized_np.shape)
#        resized_np = clean_img(resized_np, th)
#        plt.figure()
#        plt.hist(resized_np)
#        plt.pause(2) 
        resized_bool = resized_np>(th*255)
       
        label = morphology.label(resized_bool)
        num = label.max()+1
        for m in range(1, num):
            rle = run_length_encoding(label==m)
            ImageId.append(img_name[i])
            EncodedPixels.append(rle)    
    return ImageId, EncodedPixels

def write2csv(file, ImageId, EncodedPixels):
    df = pd.DataFrame({ 'ImageId' : ImageId , 'EncodedPixels' : EncodedPixels})
    df.to_csv(file, index=False, columns=['ImageId', 'EncodedPixels'])


    
class mean_image_IoU:
    """Used during training and val on transformed masks
    args: a batch of Tensor
    """
    def __call__(self, preds, trues, th):
#        print("IoU")
#        print(output.size())
#        print(trues.size())
        preds = preds.numpy()
        trues = trues.numpy()
        bs = preds.shape[0]
        
        preds[preds<(th)] = 0
        preds[preds>=(th)] = 1
        trues[trues<(th)] = 0
        trues[trues>=(th)] = 1    
        
        m_iou = 0
        for i in range(bs):
            
            intersection = \
                np.histogram2d(trues[i,:,:,:].flatten(), preds[i,:,:,:].flatten(), bins=(2, 2))[0]
        
            area_true = np.histogram(trues[i,:,:,:], bins = 2)[0]
            area_pred = np.histogram(preds[i,:,:,:], bins = 2)[0]
            area_true = np.expand_dims(area_true, -1)
            area_pred = np.expand_dims(area_pred, 0)    
            
            union = area_true + area_pred - intersection
            
            intersection = intersection[1:,1:]
            union = union[1:,1:]
            union[union == 0] = 1e-9
        
            iou = intersection / union 
            m_iou += iou[0][0]
    
        return m_iou/bs

def metric (output_logits, encodings):
    """For validation set on original masks,
    to see how post processing like resize, seg instances and clean affects the metric on validation set.
    """
    pass

metrics = {
    'mean_image_IoU': mean_image_IoU,
    'metric':metric,
    'jaccard_distance':None
    }

'''
https://www.kaggle.com/kmader/nuclei-overview-to-submission  基于形态学清理mask。
考虑一些奇怪的(重叠，圆环，密集)，是否用instances分割会更好【感觉会，因为不涉及后面的后处理，
尤其是将整图的mask分割为小masks这一步】？
'''
    