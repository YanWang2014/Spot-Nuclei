from utils import transforms_master


def data_transforms(phase, input_size = 256):

    
    composed_data_transforms = {

    'resize': transforms_master.Compose([transforms_master.Resize((input_size,input_size)),
                                          transforms_master.ToTensor()]
                                         ),
    'resize_short': transforms_master.Compose([transforms_master.Resize(input_size),
                                          transforms_master.ToTensor()]
                                         ),
    'default': transforms_master.Compose([transforms_master.ToTensor()]
                                         )
    }
    return composed_data_transforms[phase]