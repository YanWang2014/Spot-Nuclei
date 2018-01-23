import numpy as np
import os
#import cv2
import imageio

train_data_root = '../data/stage1_train/' 


'''test
'''
mask_dir = '../data/stage1_train/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9/masks/'
the_images = []
for image in os.listdir(mask_dir):
    the_image = imageio.imread(mask_dir + image)
    the_images.append(the_image)
    
the_mask = sum(the_images)
imageio.imwrite('../data/stage1_train/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9/images/' + 'mask.png', the_mask)


'''for all
'''
image_dirs = os.listdir(train_data_root)
print(len(image_dirs))
i = 0

for image_dir in image_dirs:
    i += 1
    print(i)
    mask_dir = train_data_root + image_dir + '/masks'
    masks = os.listdir(mask_dir)
    
    the_images = []
    for image in masks:
        the_image_dir = mask_dir + '/' + image
        the_image = imageio.imread(the_image_dir)
        the_images.append(the_image)
        
    the_mask = sum(the_images)
    imageio.imwrite(train_data_root + image_dir + '/images/' + 'mask.png', the_mask)