'''
imagenet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

nuclei (without using alpha):
     0.1707
     0.1552
     0.1891
     1.0000
    
     0.2635
     0.2432
     0.2959
     0.0000

(Training examples, Test examples): ( 670 , 65 )
(256, 256, 3) : 334
(1024, 1024, 3) : 16
(520, 696, 3) : 92
(360, 360, 3) : 91
(512, 640, 3) : 13
(256, 320, 3) : 112
(1040, 1388, 3) : 1
(260, 347, 3) : 5
(603, 1272, 3) : 6


# todo: 
    check if mask is only 0 and 1 (seems to be true)
    load images of different size directly (very different sizes may be a big problem). And maybe masks 
        should not be resized: resize images, predict, resize back, calculate loss and metric
    support CV
    support more transforms, especially random ones like RandomCrop and rotate
'''
import sys
sys.path.append('../')
from PIL import Image
import os
import os.path
import torch.utils.data as data
import torch
import numpy as np
from utils import transforms_master
from utils import functional_newest as F
import random


class NucleiDataset(data.Dataset):

    def __init__(self, root_dir=None, mode=None, split_ratio=0.9, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_names = os.listdir(root_dir)
        
        length = len(self.image_names)
        split_ratio = split_ratio
        np.random.seed(1000)
        self.image_names = list(np.random.permutation(self.image_names))
        
        if mode == 'train':
            self.image_names = self.image_names[: int(split_ratio*length)]
        if mode == 'val':
            self.image_names = self.image_names[int(split_ratio*length):]
        if mode == 'val_as_test':
            self.image_names = self.image_names[int(split_ratio*length):]     
            mode = 'test'
        
        self.paths = [root_dir + img_name + '/images/' for img_name in self.image_names]
        self.mode = mode
        
        self.normalize = transforms_master.Normalize(mean=[0.1707, 0.1552, 0.1891], std=[0.2635, 0.2432, 0.2959])
        self.color_jitter = transforms_master.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1, hue=0.3)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img_name = str(self.image_names[idx])
        img_path = self.paths[idx] + img_name + '.png'
        mask_path = self.paths[idx] + 'mask.png'
        
        image = Image.open(img_path)
        if image.mode == 'RGBA':
            image = self.drop_alpha(image)
        image_size = image.size
        
        if self.mode != 'test':
            mask = Image.open(mask_path)
         
        if self.mode == 'train':
            pass
            #random transforms of PIL images here
#            if random.random() < 0.5:
#                image = F.hflip(image)
#                mask = F.hflip(mask)
#            if random.random() <0.5:
#                image = F.vflip(image)
#                mask = F.vflip(mask)    
#            image = self.color_jitter(image)

        if self.transform:
            image = self.transform(image)
#            image = self.normalize(image)
            if self.mode != 'test':
                mask = F.to_grayscale(mask)
                mask = self.transform(mask)  
#        print(image.size())
        if self.mode != 'test':
            return image, mask, img_name
        else:
            return image, img_name, np.array(image_size)
    
    def pil_alpha_to_color_v2(self, image, color=(255, 255, 255)):
        """Alpha composite an RGBA Image with a specified color.
    
        Simpler, faster version than the solutions above.
    
        Source: https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
    
        Keyword Arguments:
        image -- PIL RGBA Image object
        color -- Tuple r, g, b (default 255, 255, 255)
        
        return PIL L
    
        """
        image.load()  # needed for split()
        background = Image.new('RGB', image.size, color)
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return background
    
    def drop_alpha(self, image):
        """drop alpha in PIL image directly
        """
        image = np.array(image)
        image = image[...,:3]
        return F.to_pil_image(image)

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
        test_root = '../data/stage1_test/'
    elif phases == 'train1':
        test_root = '../data/stage1_train/'
    
    transformed_dataset_test = NucleiDataset(root_dir=test_root,
                                             mode = None,
                                             transform=transforms_master.Compose([
                                                     transforms_master.Resize((256,256)),
#                                                     transforms_master.CenterCrop(224),
                                                     transforms_master.ToTensor() 
                                                     ])
                                               )           
    dataloader = data.DataLoader(transformed_dataset_test, batch_size=batch_size,shuffle=False, num_workers=INPUT_WORKERS)

    
    #calculate mean and variance
    mean_meter = AverageMeter()
    for i, (image, mask, img_name) in enumerate(dataloader):  # nchw
        if i%10 ==0:
            print(i)
        mean_meter.update(image.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True), image.size(0))  
    
    mean = mean_meter.avg
    print(mean.squeeze())
    std_meter =  AverageMeter()
    for i, (image, mask, img_name) in enumerate(dataloader):  # nchw
        if i%10 ==0:
            print(i)
        std_meter.update(((image-mean)**2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True), image.size(0))  
    print(std_meter.avg.squeeze().sqrt())
    
    
    
    
    
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