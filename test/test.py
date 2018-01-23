#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Unet, LinkNet 车分割冠军等
    https://www.kaggle.com/c/carvana-image-masking-challenge
        http://blog.kaggle.com/2017/12/22/carvana-image-masking-first-place-interview/
            https://github.com/asanakoy/kaggle_carvana_segmentation (PyTorch, U-Net + LinkNet + CV)
        https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution (keras, U-Net + Dilated Convolution)
            https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/40199


http://brainiac2.mit.edu/isbi_challenge/
    http://blog.csdn.net/u012931582/article/details/70215756 (扭曲对数据进行增强)
    http://blog.csdn.net/hduxiejun/article/details/71107285

实例分割  
    https://github.com/broadinstitute/keras-rcnn
    https://www.kaggle.com/c/data-science-bowl-2018/discussion/47690 (实例分割讨论 Heng CherKeng)
    https://www.kaggle.com/c/data-science-bowl-2018/discussion/47686
    
    代码
    https://github.com/msracver/FCIS (mxnet)
    
    https://www.zhihu.com/question/57403701   
    https://github.com/TuSimple/mx-maskrcnn （Resnet-50-FPN）
    https://github.com/matterport/Mask_RCNN (Keras and TensorFlow， ResNet101有coco的权重！！！)
    
    Recurrent Neural Networks for Semantic Instance Segmentation (PyTorch)
        https://github.com/imatge-upc/rsis 
        
    End-to-End Instance Segmentation with Recurrent Attention  (TF)
        https://github.com/renmengye/rec-attend-public
    
    pix2pix
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
        
分割专栏    
    https://www.zhihu.com/question/51704852
    http://blog.csdn.net/column/details/15893.html (实例分割)
    http://blog.csdn.net/zhyj3038/article/details/71195262 (FCN SegNet U-net DeconvNet)
    linknet
        http://blog.csdn.net/cv_family_z/article/details/76061007
        http://blog.csdn.net/zhangjunhit/article/details/75097842

代码架构
    https://github.com/chenyuntc/pytorch-best-practice
    https://github.com/filick/GRU-RCN
    
细胞
    http://appsrv.cse.cuhk.edu.hk/~hchen/research/2015miccai_gland.html (15冠军)
    https://www.cnblogs.com/xiangfeidemengzhu/p/7453207.html

    https://github.com/jr0th/segmentation    
        硕士论文，try to classify each pixel of an image into either background, cell or boundary

 
"""

