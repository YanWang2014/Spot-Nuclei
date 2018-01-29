from config import opt
from models import net_factory
from utils import common
import torch
import torch.nn as nn
from utils import nuclei_dataset
import torch.utils.data as data
import torch.nn.functional as Fn
import numpy as np
    
def run():
    model = net_factory.loader(opt.model, opt.num_classes)
                                
    if opt.use_multi_gpu:
        model = nn.DataParallel(model)
        
    if opt.use_gpu:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        model = model.cuda()

    common.resume(model, opt.resumed_check, opt.model)
    

    transformed_dataset_val = nuclei_dataset.NucleiDataset(root_dir=opt.val_data_root,
                                             mode = 'val_as_test', # val used in test mode
                                             transform=opt.transforms['val'])
    val_loader = data.DataLoader(transformed_dataset_val, 
                                   batch_size=opt.batch_size,
                                   shuffle=False, 
                                   num_workers=opt.num_workers, 
                                   pin_memory=opt.use_gpu)
    print(len(val_loader)*opt.batch_size)
#    val_test('val', val_loader, model)
    
    
    transformed_dataset_test = nuclei_dataset.NucleiDataset(root_dir=opt.test_data_root,
                                             mode = 'test',
                                             transform=opt.transforms['test'])
    test_loader = data.DataLoader(transformed_dataset_test, 
                                   batch_size=opt.batch_size,
                                   shuffle=False, 
                                   num_workers=opt.num_workers, 
                                   pin_memory=opt.use_gpu)
    print(len(test_loader)*opt.batch_size)
    val_test('test', test_loader, model)


    
def _each_epoch(mode, loader, model):
    ImageId = []
    EncodedPixels = []

    model.eval()

    for i, (image, img_name, img_size)  in enumerate(loader):
#        if i%10 == 0:
#            print(i)
#        print('tuple')
#        print(img_size)
            
        if opt.use_gpu:
            image = image.cuda(async=True)
        input_var = torch.autograd.Variable(image, volatile=(mode != 'train'))

        logits = model(input_var)
        predicts = Fn.sigmoid(logits) # between 0 and 1
        
        if opt.if_debug:
            print('check')
            common.plot_tensor(image.cpu())
            common.plot_tensor_mask(predicts.data.cpu())
            common.plot_resized_mask(predicts.data.cpu(), img_size, img_name, opt.seg_th)
            print(img_size)
        
        ImageId_batch, EncodedPixels_batch = common.resize_tensor_2_numpy_and_encoding(predicts.data.cpu(), img_size, img_name, opt.seg_th)
        ImageId += ImageId_batch
        EncodedPixels += EncodedPixels_batch

    common.write2csv('results/'+opt.model+'_'+mode+'.csv', ImageId, EncodedPixels)
    print(len(np.unique(ImageId)))
    

def val_test(mode, test_loader, model):
    return _each_epoch(mode, test_loader, model)

    
if __name__ == '__main__':
    run()