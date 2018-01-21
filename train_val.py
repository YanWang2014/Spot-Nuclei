from config import opt
from models import net_factory
from utils import common
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
from utils import nuclei_dataset
import torch.utils.data as data
import torch.nn.functional as Fn


def run():
    model = net_factory.loader(opt.model, opt.num_classes)
                                
    if opt.use_multi_gpu:
        model = nn.DataParallel(model)
        
    if opt.use_gpu:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        model = model.cuda()
        criterion = opt.criterion.cuda()
    else:
        criterion = opt.criterion

    best_metric = 0
    best_loss = 10000
    
    if opt.try_resume:
        common.resume(model, opt.resumed_check, opt.model)


    transformed_dataset_train = nuclei_dataset.NucleiDataset(root_dir=opt.train_data_root,
                                             mode = 'train',
                                             transform=opt.transforms['train'])
    train_loader = data.DataLoader(transformed_dataset_train, 
                                   batch_size=opt.batch_size,
                                   shuffle=True, 
                                   num_workers=opt.num_workers, 
                                   pin_memory=opt.use_gpu)
    
    transformed_dataset_val = nuclei_dataset.NucleiDataset(root_dir=opt.val_data_root,
                                             mode = 'val',
                                             transform=opt.transforms['val'])
    val_loader = data.DataLoader(transformed_dataset_val, 
                                   batch_size=opt.batch_size,
                                   shuffle=False, 
                                   num_workers=opt.num_workers, 
                                   pin_memory=opt.use_gpu)
    print(len(train_loader)*opt.batch_size)
    print(len(val_loader)*opt.batch_size)

    if opt.optim_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay) 
    elif opt.optim_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    
    lr_scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, 
                                         verbose=True, threshold=0.0001, threshold_mode='rel', 
                                         cooldown=0, min_lr=1e-5, eps=1e-08)


    for epoch in range(opt.epochs):

        # train for one epoch
        metric, loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        metric, loss = validate(val_loader, model, criterion, epoch)

        # remember best 
        is_best = loss <= best_loss
        best_loss = min(loss, best_loss)
        if epoch % opt.save_freq == 0:
            common.save_checkpoint_epoch({
                    'epoch': epoch,
                    'arch': opt.model,
                    'state_dict': model.state_dict(),
                    'best_metric': best_metric,
                    'loss': loss
                    },  epoch, opt.model)
        if is_best:
            common.save_checkpoint({
                    'epoch': epoch,
                    'arch': opt.model,
                    'state_dict': model.state_dict(),
                    'best_metric': best_metric,
                    'loss': loss
                    },  is_best, opt.model)
    
        lr_scheduler.step(loss)



def _each_epoch(mode, loader, model, criterion, optimizer=None, epoch=None):
    losses = common.AverageMeter()
    metrics = common.AverageMeter()

    if mode == 'train':
        model.train()
    else:
        model.eval()

    for i, (image, mask, img_name)  in enumerate(loader):

        if opt.use_gpu:
            image = image.cuda(async=True)
            mask = mask.cuda(async=True)            
        input_var = torch.autograd.Variable(image, volatile=(mode != 'train'))
        mask_var = torch.autograd.Variable(mask, volatile=(mode != 'train'))

        output = model(input_var)
        
        loss = criterion(output, mask_var)
        losses.update(loss.data[0], image.size(0))
        metric = opt.metric(Fn.sigmoid(output).data.cpu(), mask_var.data.cpu(), opt.seg_th)
        metrics.update(metric, image.size(0))
 
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        if opt.if_debug and i % opt.print_freq == 0: 
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Metric {metric.val:.3f} ({metric.avg:.3f})'.format(
                epoch, i, len(loader), loss=losses, metric=metrics))


    print(' *Epoch:[{0}] Metric {metric.avg:.3f}  Loss {loss.avg:.4f}'
          .format(epoch, metric=metrics, loss=losses))

    return metrics.avg, losses.avg


def validate(val_loader, model, criterion, epoch):
    return _each_epoch('validate', val_loader, model, criterion, optimizer=None, epoch=epoch)


def train(train_loader, model, criterion, optimizer, epoch):
    return _each_epoch('train', train_loader, model, criterion, optimizer, epoch)


    
if __name__ == '__main__':
    run()