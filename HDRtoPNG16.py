#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import sys
import os
import numpy as np


# In[2]:


def savePic(picture,fileName,extention,outPath): #saves the given array as a pictures to the given output path
    outPath = outPath+fileName+'.'+extention
    try:

        #print(picture.shape)
        print('Writing picture to ====> '+outPath)
        cv2.imwrite(outPath,picture)
        
    except:
        print('Failed while saving picture: '+fileName+' to '+ outPath+' sorry :(')
        print('--------------------')


# In[3]:


#Read Picture and return it

def readThePicture(picturepath):
    #  open ImageObject
    try:
        print('Reading <==== '+picturepath)
        img = cv2.imread(picturepath, cv2.IMREAD_UNCHANGED)# | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    except:
        print('There was an error while reading the picture')
        img = 0
    return img


# In[10]:


inputPathHDR = input('spezify the input path of .hdr pictures [default: ./hdrOut/ ] ') or './hdrInput/' #set the picture save path if it is choosen
if not os.path.exists(inputPathHDR):
        os.mkdir(inputPathHDR)
outPath = input('spezify the output path of .png16bit pictures [default: ./generatedHDR/ ] ') or './convHDRpng/' #set the picture save path if it is choosen
if not os.path.exists(outPath):
        os.mkdir(outPath)


# In[11]:


print('converting .hdr into .png 16bit files')
i = 0

import os, os.path

# simple version for working with CWD
while(len(os.listdir(inputPathHDR)) >= i ):
    try:
    
        if((os.listdir(inputPathHDR)[i].split('.')[1]) != 'hdr'):
                i = i + 1
                print('Fail at picture: '+os.listdir(inputPathHDR)[i])
        if((os.listdir(inputPathHDR)[i].split('.')[1]) == 'hdr'):
            name = os.listdir(inputPathHDR)[i]
            pic = (readThePicture(inputPathHDR+name))#.astype(np.float32)
            #print(i)
            #print(pic.min())
            pic = (np.clip((pic*((2**16)-1)),0,((2**16)-1))).astype(np.uint16)
            saveName = name.split('_')[0]
            savePic(pic,saveName,'png',outPath)
        i = i + 1
        print(i)
    except:
        print('')
        i = i + 1
print('DONE.')

