from utils import transforms_master


def data_transforms(phase, input_size = 256, train_scale = 256, test_scale = 256):
    print('input_size %d, train_scale %d, test_scale %d' %(input_size,train_scale,test_scale))
    
    composed_data_transforms = {
    'default': transforms_master.Compose([transforms_master.Resize((input_size,input_size)),
                                         transforms_master.ToTensor() 
                                         ])
    }
    return composed_data_transforms[phase]