from config import opt
from models import net_factory
from utils import common
import torch
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

    common.resume(model, 'best', opt.model)
    
    transformed_dataset_val = nuclei_dataset.NucleiDataset(root_dir=opt.test_data_root,
                                             mode = 'test',
                                             transform=opt.transforms['test'])
    val_loader = data.DataLoader(transformed_dataset_val, 
                                   batch_size=opt.batch_size,
                                   shuffle=True, 
                                   num_workers=opt.num_workers, 
                                   pin_memory=opt.use_gpu)
    print(len(val_loader)*opt.batch_size)
    print('total:', len(val_loader))
    
    test(val_loader, model)


    
def _each_epoch(mode, loader, model):
    ImageId = []
    EncodedPixels = []

    model.eval()

    for i, (image, img_name, img_size)  in enumerate(loader):
#        if i%10 == 0:
#            print(i)

        if opt.use_gpu:
            image = image.cuda(async=True)
        input_var = torch.autograd.Variable(image, volatile=(mode != 'train'))

        logits = model(input_var)
        predicts = Fn.sigmoid(logits) # between 0 and 1
        
#        if i == 0:
#            common.plot_tensor(image.cpu())
#            common.plot_tensor_mask(predicts.data.cpu())
#            print(img_size)
        
        ImageId_batch, EncodedPixels_batch = common.resize_tensor_2_numpy_and_encoding(predicts.data.cpu(), img_size, img_name, th = 0.5)
        ImageId += ImageId_batch
        EncodedPixels += EncodedPixels_batch

    common.write2csv('results/'+opt.model+'.csv', ImageId, EncodedPixels)
    

def test(val_loader, model):
    return _each_epoch('test', val_loader, model)

    
if __name__ == '__main__':
    run()