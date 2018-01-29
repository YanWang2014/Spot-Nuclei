#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:29:07 2018

@author: wayne
"""

from PIL import Image
import numpy as np
image = Image.open('test.png') #w 706 h 420

print (type(image)) 
print (image.size)  
print (image.mode) 

# resize w*h
image = image.resize((200,100))
print (image.size) # out: (200,100)

