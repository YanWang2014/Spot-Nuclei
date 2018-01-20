import torch
from utils import transforms_master
from utils import common

class Config(object):
    
    train_data_root = 'data/stage1_train/' 
    test_data_root = 'data/stage1_test/' 

    
    model = 'UNet11' # models/__init__.py
    num_classes = 1
    
    image_size = 224
    transforms={'train':transforms_master.Compose([transforms_master.Resize((image_size,image_size)),
                                         transforms_master.ToTensor() 
                                         ]),
                'val':transforms_master.Compose([transforms_master.Resize((image_size,image_size)),
                                         transforms_master.ToTensor() 
                                         ]),
                'test':transforms_master.Compose([transforms_master.Resize((image_size,image_size)),
                                         transforms_master.ToTensor() 
                                         ])
    }
    
    
    batch_size = 16
    epochs = 20
    optim_type = 'Adam'
    lr = 0.001 
    momentum = 0.9
    weight_decay = 0#1e-4 
    
    if model == 'UNet11':
        criterion = common.losses['UNet11_Loss']()
    else:
        criterion = common.losses['BCELoss2d']()   
    
    
    use_gpu = torch.cuda.is_available()
    use_multi_gpu = torch.cuda.device_count() > 1
    num_workers = 4 
    save_freq = 2
    if_debug = True
    print_freq = 2   


opt = Config()



if __name__== '__main__':
    print(opt.use_gpu)