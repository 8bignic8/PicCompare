#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import imageio
import mat73
import os
import cv2
import random
import time
import csv
import sys
from skimage.measure import compare_ssim


# In[2]:


##global Variable default state
global csv_file #defines csv_file as a global variable
global pathtoHDRgt
global csvPath
global pathtoSDR
global pathtoHDRgen
global result
csv_name = 'pictureData.csv'
csvPath = './csv/'
pathtoHDRgt = './groundTruthHDR/'
pathtoSDR = './LDR/'
pathtoHDRgen = './generatedHDR/'
result = './results/'
global HDRgtPic
global SDRpic
global HDRgenPic


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


# In[4]:


def writeToCV(name, file):
    print('Write data to '+name)
    np.savetxt(name, file, delimiter=",")


# In[5]:


def readCSV(file):
    #https://docs.python.org/3/library/csv.html
    print('Read data to '+str(name))
    result = np.array(list(csv.reader(open(file, "rb"), delimiter=","))).astype("float")
    return result


# In[6]:


def savePic(picture,fileName,extention,outPath): #saves the given array as a pictures to the given output path
    outPath = outPath+fileName+'.'+extention
    try:

        #print(picture.shape)
        print('Writing picture to ====> '+outPath)
        cv2.imwrite(outPath,picture)
        
    except:
        print('Failed while saving picture: '+fileName+' to '+ outPath+' sorry :(')
        print('--------------------')


# In[7]:


def together():
    try:
        global SDRpic,HDRgtPic,HDRgenPic  
        together = np.hstack((SDRpic,HDRgenPic,HDRgtPic))
        print(together.shape)
    except:
        print('Fail Pictures do not have the same size')
        print('Ground Truth: '+str(HDRgtPic.shape))
        print('Low Res pic: ' +str(SDRpic.shape))
        print('Low Res pic: ' +str(SDRpic_new.shape))
        print('Generated Picture: ' +str(HDRgenPic.shape))
    return together 


# In[8]:


def scale(img,factor):
    print('Scaling up factor SDR: '+str(factor))
    scale = (int(img.shape[1])*factor, int(img.shape[0])*factor)       
    img_new = cv2.resize(img, scale, interpolation = cv2.INTER_AREA)
    return img_new


# In[9]:


# Usage:
#
# python3 script.py --input original.png --output modified.png
# Based on: https://github.com/mostafaGwely/Structural-Similarity-Index-SSIM-

# 2. Construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-f", "--first", required=True, help="Directory of the image that will be compared")
#ap.add_argument("-s", "--second", required=True, help="Directory of the image that will be used to compare")
#args = vars(ap.parse_args())

# 3. Load the two input images
#imageA = cv2.imread(args["first"])
#imageB = cv2.imread(args["second"])
def ssim(imageA,imageB):
    # 4. Convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY) #image that will be compared
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY) #image that will be used to compare

    # 5. Compute the Structural Similarity Index (SSIM) between the two
    #    images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8") #diff image needs to be uint16 and 16 bit?

    # 6. You can print only the score if you want
    print("SSIM: {}".format(score))


# In[10]:


def psnrfunc(img_orig, img_out):
    img_out = (img_out / ((2**16)-1)).astype(np.float32)
    img_orig = (img_orig / ((2**16)-1)).astype(np.float32)
    psnr = cv2.PSNR(img_out, img_orig)
    return psnr


# In[11]:


def text(Wtext,img):
    #https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
    
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # fontScale
    fontScale = 2
    # org
    img.shape[0]
    org = (50*fontScale,int(img.shape[0])-50*fontScale)
    print(org)
    

    # Blue color in BGR
    color = (0, 0, ((2**16)-1))

    # Line thickness of 2 px
    thickness = 5

    # Using cv2.putText() method
    image = cv2.putText(img, Wtext, org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
    return image


# In[12]:


def inputData():
    global csv_file #defines csv_file as a global variable
    global pathtoHDRgt
    global csvPath
    global pathtoSDR
    global pathtoHDRgen
    global result
    
    print('This script stiches three pictures together in the Order: ')
    print('Ground Trouth, SDR inputpicture, Generated HDR Picture')
    result = input('Enter path for the results (default: '+result+'): ') or result
    
    if not os.path.exists(result):
        os.mkdir(result)
    csv_name = input('Enter the .csv file name (default: pictureData): ') or ('pictureData') +('.csv')
    csvPath = input('Enter the path to save Path file (deflaut: '+csvPath+'): ') or ('./csv/')
    if not os.path.exists(csvPath):
        os.mkdir(csvPath)
    pathtoHDRgt = input('Enter the path to the Ground Truth Picture (deflaut: '+pathtoHDRgt+') ') or pathtoHDRgt
    if not os.path.exists(pathtoHDRgt):
        os.mkdir(pathtoHDRgt)
    pathtoSDR = input('Enter the path to the SDR inputpicture (deflaut: '+pathtoSDR+') ') or pathtoSDR
    if not os.path.exists(pathtoSDR):
        os.mkdir(pathtoSDR)
    pathtoHDRgen = input('Enter the path to the Generated HDR Picture (deflaut: '+pathtoHDRgen+') ') or pathtoHDRgen
    if not os.path.exists(pathtoHDRgen):
        os.mkdir(pathtoHDRgen)
    
    assert os.path.exists(csvPath), "I did not find the path "+str(csvPath)
    csv_file = open(csv_name, 'w')
    csv_file = csv.writer(csv_file)
    amountOfPictures = int(sum(1 for f in os.listdir(pathtoHDRgt) if f.endswith('.'+(os.listdir(pathtoHDRgt)[1].split('.')[1]))))
    print(amountOfPictures)
    global psnr
    psnr = np.zeros((amountOfPictures+1,3)).astype(np.float32)


# In[13]:


start_time = time.time() #start the timeing of the Prgramm
#finding the rigth picture pairs in paths
#global HDRgtPic

#global HDRgenPic
i = 0
inputData()
try:
    while( (sum(1 for f in os.listdir(pathtoHDRgen)))-1 >= i ):
        if((os.listdir(pathtoHDRgen)[i].split('.')[1]) != 'png'):
            i = i + 1    
        else:
            print(i)
            currentFileName = os.listdir(pathtoHDRgen)[i]
            i = i+1
            picName = currentFileName.split('.')[0]
            HDRgenPic = readThePicture(pathtoHDRgen+currentFileName)
            #search for the right picture with the same name but different ending
            foundFileSDR = False
            posSDR = 0
            while(foundFileSDR != True):
                if(picName == (os.listdir(pathtoSDR)[posSDR].split('.')[0])):
                    currentFileName = os.listdir(pathtoSDR)[posSDR]
                    SDRpic = readThePicture(str(pathtoSDR+currentFileName))
                    SDRpic = SDRpic/((2**8)-1)
                    SDRpic = (SDRpic*((2**16)-1)).astype(np.uint16)
                    foundFileSDR = True
                posSDR = posSDR + 1

            foundFileGT = False
            posGT = 0
            while(foundFileGT != True):
                if(picName == (os.listdir(pathtoHDRgt)[posGT].split('.')[0])):
                    currentFileName = os.listdir(pathtoHDRgt)[posGT]
                    HDRgtPic = readThePicture(str(pathtoHDRgt+currentFileName))
                    foundFileGT = True
                posGT = posGT + 1
            
            if(HDRgtPic.shape != SDRpic.shape != HDRgenPic.shape): ##upsizing if needed
                factor = int(HDRgtPic.shape[1]/SDRpic.shape[1])
                SDRpic = scale(SDRpic,factor)
            
            psnr[i,0] = psnrfunc(HDRgtPic,HDRgtPic)
            print(psnr[i,0])
            psnr[i,1] = psnrfunc(HDRgtPic,HDRgenPic) 
            print(psnr[i,1])
            HDRgenPic = text('PSNR: '+str(psnr[i,1])+'_generated HDR',HDRgenPic)
            HDRgtPic = text('Ground truth high dynamic range (HDR)',HDRgtPic)
            SDRpic = text('standard dynamic range (SDR)',SDRpic)
            picTure = together()
            
            savePic(picTure,(str(i)+'Result_'+picName),'png',result)
            #if(colorSpaceAnal == True ):
            #    savePic(picTure[:,:,0]*255,(str(i)+'Result_ColorSpace'+picName),'png',result)
             #   savePic(picTure[],(str(i)+'Result_ColorSpace'+picName),'png',result)
              #  savePic(picTure[],(str(i)+'Result_ColorSpace'+picName),'png',result)
        
except: 
    print('There was an error while finding the pictures to compare')
    print('Picture name to find: '+picName)
writeToCV(csv_name, psnr)
print('Finished and it took: '+str((time.time() - start_time)/60)+'minutes')


# In[14]:


#savePic(together(),(str(i)+'Result_'+picName),'png',result)

#psnr[0,0] = psnrfunc(HDRgtPic,HDRgenPic)
#writeToCV('./csv/pictureData.csv', psnr)
scale(SDRpic,factor)


# In[15]:


P = readThePicture('/Users/littledragon/Documents/BA 13022020/programme/PicCompare/LDR/000019.png')
P = P/((2**8)-1)
P = (P*((2**16)-1)).astype(np.uint16)
savePic(P,'Result_Test','png','/Users/littledragon/Documents/BA 13022020/programme/PicCompare/LDR')


# In[ ]:




