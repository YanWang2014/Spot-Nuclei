'''
imagenet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

nuclei:
     0.1707
     0.1552
     0.1891
     1.0000
    
     0.2635
     0.2432
     0.2959
     0.0000

'''

from PIL import Image
import os
import os.path
import torch.utils.data as data
import torch
import numpy as np
from utils import transforms_master
import cv2
from utils import functional_newest

class NucleiDataset(data.Dataset):

    def __init__(self, root_dir=None, mode=None, split_ratio=0.8, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_names = os.listdir(root_dir)
        
        length = len(self.image_names)
        split_ratio = split_ratio
        np.random.seed(100)
        self.image_names = list(np.random.permutation(self.image_names))
        if mode == 'train':
            self.image_names = self.image_names[: int(split_ratio*length)]
        if mode == 'val':
            self.image_names = self.image_names[int(split_ratio*length):]
        
        self.paths = [root_dir + img_name + '/images/' for img_name in self.image_names]
        self.mode = mode

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img_name = str(self.image_names[idx])
        img_path = self.paths[idx] + img_name + '.png'
        mask_path = self.paths[idx] + 'mask.png'
        
        image = Image.open(img_path)
        if image.mode == 'RGBA':
            image = self.pil_alpha_to_color_v2(image)
        
        if self.mode != 'test':
            mask = Image.open(mask_path)
            #mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)
            if self.mode != 'test':
                mask = functional_newest.to_grayscale(mask)
                mask = self.transform(mask)
        
        if self.mode != 'test':
            return image, mask, img_name
        else:
            return image, img_name
    
    def pil_alpha_to_color_v2(self, image, color=(255, 255, 255)):
        """Alpha composite an RGBA Image with a specified color.
    
        Simpler, faster version than the solutions above.
    
        Source: https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
    
        Keyword Arguments:
        image -- PIL RGBA Image object
        color -- Tuple r, g, b (default 255, 255, 255)
    
        """
        image.load()  # needed for split()
        background = Image.new('RGB', image.size, color)
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return background

if __name__ == "__main__":
    
#    x = torch.randn(5,2,3,4) # nchw
#    y = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
#    print(y)
#    print(y+1)
#    print(y.squeeze())
#    print((y+1).squeeze().sqrt())
#    print((y+1).squeeze()**2)
    print('start here')

    class AverageMeter(object):
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
    
    
    phases = 'train1'
    batch_size  = 64
    INPUT_WORKERS = 4
    
    if phases == 'test1':
        test_root = 'data/stage1_test/'
    elif phases == 'train1':
        test_root = 'data/stage1_train/'
    
    transformed_dataset_test = NucleiDataset(root_dir=test_root,
                                             mode = None,
                                             transform=transforms_master.Compose([
                                                     transforms_master.Resize((256,256)),
#                                                     transforms_master.CenterCrop(224),
                                                     transforms_master.ToTensor() 
                                                     ])
                                               )           
    dataloader = data.DataLoader(transformed_dataset_test, batch_size=batch_size,shuffle=False, num_workers=INPUT_WORKERS)

    
#    #calculate mean and variance
#    mean_meter = AverageMeter()
#    for i, (image, mask, img_name) in enumerate(dataloader):  # nchw
#        if i%10 ==0:
#            print(i)
#        mean_meter.update(image.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True), image.size(0))  
#    
#    mean = mean_meter.avg
#    print(mean.squeeze())
#    std_meter =  AverageMeter()
#    for i, (image, mask, img_name) in enumerate(dataloader):  # nchw
#        if i%10 ==0:
#            print(i)
#        std_meter.update(((image-mean)**2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True), image.size(0))  
#    print(std_meter.avg.squeeze().sqrt())
#    
#    
    
    
    
    print(len(dataloader))
    img_batch, mask_batch, name_batch = next(iter(dataloader))
    print(img_batch.size())
    print(mask_batch.size())
    print(len(name_batch))

    import matplotlib.pyplot as plt
    
    def imshow(inp, title=None):
        inp = inp.numpy().transpose((1, 2, 0))
        plt.figure()
        plt.imshow(inp)
        plt.pause(1)
        
    imshow(img_batch[0,:,:,:])
    imshow(mask_batch[0,:,:,:])
    print(name_batch[0])