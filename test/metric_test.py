#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 10:00:07 2018

based on https://www.kaggle.com/wcukierski/example-metric-implementation

based on true masks of training data, test the impact of morphology ops
    for true masks, clean_img worsen the mAP
"""

import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import skimage.segmentation
from skimage import morphology
from skimage.morphology import closing, opening, disk

# Load a single image and its associated masks
id = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9'
file = "../data/stage1_train/{}/images/{}.png".format(id,id)
masks = "../data/stage1_train/{}/masks/*.png".format(id)
true_masks = "../data/stage1_train/{}/images/mask.png".format(id)

image = skimage.io.imread(file)
masks = skimage.io.imread_collection(masks).concatenate()

height, width, _ = image.shape
num_masks = masks.shape[0]

'''instance labels from mask images, can also from the provided csv file
'''
# Make a ground truth label image (pixel value is index of object label)
labels = np.zeros((height, width), np.uint16)
for index in range(0, num_masks):
    labels[masks[index] > 0] = index + 1

# Show label image
fig = plt.figure()
plt.imshow(image)
plt.title("Original image")

fig = plt.figure()
plt.imshow(labels)
plt.title("Ground truth masks")
print(np.unique(labels))

true_masks = skimage.io.imread(true_masks)
true_masks = true_masks[:,:,0]
fig = plt.figure()
plt.imshow(true_masks)
plt.title("True masks")
print(np.unique(true_masks))

'''semantic to instances
'''
#true_masks = true_masks>128
def clean_img(x, th=0.5):
    """http://blog.csdn.net/haoji007/article/details/52063306
    """
    x[x<(th*255)] = 0
    x[x>=(th*255)] = 255
    return opening(closing(x, disk(1)), disk(3))
#true_masks = clean_img(true_masks)

y_pred = morphology.label(true_masks, 
                          neighbors=None, 
                          background=None, 
                          return_num=False, 
                          connectivity=None)
fig = plt.figure()
plt.imshow(y_pred)
plt.title("y_pred")
print(np.unique(y_pred))


'''metric
'''
# Compute number of objects
true_objects = len(np.unique(labels))
pred_objects = len(np.unique(y_pred))
print("Number of true objects:", true_objects)
print("Number of predicted objects:", pred_objects)

# Compute intersection between all objects
intersection = \
    np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

# Compute areas (needed for finding the union between all objects)
area_true = np.histogram(labels, bins = true_objects)[0]
area_pred = np.histogram(y_pred, bins = pred_objects)[0]
area_true = np.expand_dims(area_true, -1)
area_pred = np.expand_dims(area_pred, 0)    

# Compute union
union = area_true + area_pred - intersection

# Exclude background from the analysis
intersection = intersection[1:,1:]
union = union[1:,1:]
union[union == 0] = 1e-9

# Compute the intersection over union
iou = intersection / union


# Precision helper function
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn

# Loop over IoU thresholds
prec = []
print("Thresh\tTP\tFP\tFN\tPrec.")
for t in np.arange(0.5, 1.0, 0.05):
    tp, fp, fn = precision_at(t, iou)
    p = tp / (tp + fp + fn)
    print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
    prec.append(p)
print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))






