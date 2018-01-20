import torch
from utils import common
from utils import preprocessing

class Config(object):
    
    train_data_root = 'data/stage1_train/' 
    test_data_root = 'data/stage1_test/' 

    
    model = 'LinkNet18'
    num_classes = 1
    
    input_size = 224
    transforms={'train':preprocessing.data_transforms('default', input_size),
                'val':preprocessing.data_transforms('default', input_size),
                'test':preprocessing.data_transforms('default', input_size)
    }
    
    
    batch_size = 16
    epochs = 50
    try_resume = True
    optim_type = 'SGD'
    lr = 0.001 
    momentum = 0.9
    weight_decay = 0#1e-4 
    
    criterion = common.losses['BCEDiceLoss']()
    
    
    use_gpu = torch.cuda.is_available()
    use_multi_gpu = torch.cuda.device_count() > 1
    num_workers = 4 
    save_freq = 2
    if_debug = False
    print_freq = 2   


opt = Config()



if __name__== '__main__':
    print(opt.use_gpu)