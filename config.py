'''
gpu linknet18 SGD 16
cpu linknet34 8
gpu linknet34 3 SGD 4
gpu UNet11  SGD 1

'''
import torch
from utils import common
from utils import preprocessing

class Config(object):
    
    train_data_root = 'data/stage1_train/' 
    val_data_root = 'data/stage1_train/'
    test_data_root = 'data/stage1_test/' 

    try_resume = False
    resumed_check = 'best'

    model = 'LinkNet34'
    num_classes = 1
    
    input_size = 256
    
    # for this task, transforms are distributed in preprocessing and dataset
    transforms = {'train':preprocessing.data_transforms('resize', input_size),              
                 'val':preprocessing.data_transforms('resize', input_size),
                 'test':preprocessing.data_transforms('resize', input_size)   
                 }

    
    batch_size = 32
    epochs = 200
    save_freq = 20

    optim_type = 'SGD'
    lr = 0.02 # 0.001
    momentum = 0.95
    weight_decay = 0.000#1e-4 
    
    criterion = common.losses['BCELoss2d']() #BCEDiceLoss  -0.8263 达到了, BCELoss2d: 0.0057
    metric = common.metrics['mean_image_IoU']()
    
    
    use_gpu = torch.cuda.is_available()
    use_multi_gpu = torch.cuda.device_count() > 1
    num_workers = 4
    
    if_debug = False
    print_freq = 1
    
    seg_th = 0.29


opt = Config()



if __name__== '__main__':
    print(opt.use_gpu)