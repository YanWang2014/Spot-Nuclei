from config import opt
from models import net_factory
from utils import common
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
from utils import nuclei_dataset
import torch.utils.data as data


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


    
    transformed_dataset_val = nuclei_dataset.NucleiDataset(root_dir=opt.train_data_root,
                                             mode = 'test',
                                             transform=opt.transforms['test'])
    val_loader = data.DataLoader(transformed_dataset_val, 
                                   batch_size=opt.batch_size,
                                   shuffle=False, 
                                   num_workers=opt.num_workers, 
                                   pin_memory=opt.use_gpu)
    print(len(val_loader)*opt.batch_size)
    
    metric, loss = validate(val_loader, model, criterion, epoch = 0)




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
        metric = common.metric(output, mask_var)
        metrics.update(metric[0], image.size(0))
 
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        if opt.if_debug and i % opt.print_freq == 0: 
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Metric {metric.val:.3f} ({metric.avg:.3f})'.format(
                epoch, i, len(loader), loss=losses, metric=metrics))


    print(' *Epoch:[{0}] metric {metric.avg:.3f}  Loss {loss.avg:.4f}'
          .format(epoch, metric=metrics, loss=losses))

    return metrics.avg, losses.avg


def validate(val_loader, model, criterion, epoch):
    return _each_epoch('validate', val_loader, model, criterion, optimizer=None, epoch=epoch)

    
if __name__ == '__main__':
    run()