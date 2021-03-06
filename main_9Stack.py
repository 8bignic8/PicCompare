#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numba import jit
import os
import cv2
import time
import csv
import sys
from skimage.measure import compare_ssim
import math
from xlwt import Workbook
#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


# In[ ]:





# In[ ]:


##global Variable default state
global csv_file #defines csv_file as a global variable
global pathtoHDRgt
global xlsPath
global pathtoSDR
global pathtoHDRgen
global result
csv_name = 'pictureData.csv'
xlsPath = './xls/'
pathtoHDRgt = './groundTruthHDR/'

mashinePath = './jsiGan/'
pathtoSDR = 'SDR/'
pathtoHDRgen = 'HDR/'
tmo_path = ('reinhard/','mantiuk/','drago/','linear/')


result = './results/'
global HDRgtPic
global SDRpic
global HDRgenPic
global data


# In[ ]:


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


# In[ ]:





# In[ ]:


###https://github.com/SVLaursen/Python-RGB-to-HSI/blob/master/converter.py
def HLS(pictureToTest):
    pictureToTest = np.float32(pictureToTest/((2**16)-1))

    pictureToTest = cv2.cvtColor(pictureToTest, cv2.COLOR_BGR2HLS).astype(np.float64)

    return pictureToTest[:,:,0].mean(),pictureToTest[:,:,1].mean(),pictureToTest[:,:,2].mean()


# In[ ]:


def savePic(picture,fileName,extention,outPath): #saves the given array as a pictures to the given output path
    outPath = outPath+fileName+'.'+extention
    try:

        #print(picture.shape)
        print('Writing picture to ====> '+outPath)
        cv2.imwrite(outPath,picture)
        
    except:
        print('Failed while saving picture: '+fileName+' to '+ outPath+' sorry :(')
        print('--------------------')


# In[ ]:


def horStack(startPic,addPic):
    try:
        global SDRpic,HDRgtPic,HDRgenPic  
        #together = np.vstack((SDRpic,HDRgenPic,HDRgtPic))
        
        together = np.hstack((startPic,addPic))
        #print(together.shape)
    except:
        print('Fail Pictures do not have the same size')
        print('Ground Truth: '+str(HDRgtPic.shape))
        print('Low Res pic: ' +str(SDRpic.shape))
        print('Low Res pic: ' +str(SDRpic_new.shape))
        print('Generated Picture: ' +str(HDRgenPic.shape))
    return together 


# In[ ]:


def vertStack(startPic,addPic):
    try:
        global SDRpic,HDRgtPic,HDRgenPic  
        together = np.vstack((startPic,addPic))
        #print(together.shape)
    except:
        print('Fail Pictures do not have the same size')
        print('Ground Truth: '+str(HDRgtPic.shape))
        print('Low Res pic: ' +str(SDRpic.shape))
        print('Low Res pic: ' +str(SDRpic_new.shape))
        print('Generated Picture: ' +str(HDRgenPic.shape))
    return together 


# In[ ]:


def scale(img,factor):
    #print('Scaling up factor SDR: '+str(factor))
    scale = (int(img.shape[1])*factor, int(img.shape[0])*factor)       
    img_new = cv2.resize(img, scale, interpolation = cv2.INTER_AREA)
    return img_new


# In[ ]:


# Usage:
#
# Based on: https://github.com/mostafaGwely/Structural-Similarity-Index-SSIM-
#https://github.com/mostafaGwely/Structural-Similarity-Index-SSIM-/blob/master/ssim.py

# 2. Construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-f", "--first", required=True, help="Directory of the image that will be compared")
#ap.add_argument("-s", "--second", required=True, help="Directory of the image that will be used to compare")
#args = vars(ap.parse_args())

# 3. Load the two input images
#imageA = cv2.imread(args["first"])
#imageB = cv2.imread(args["second"])
def ssim(imageB,imageA):
    # 4. Convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY) #image that will be compared
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY) #image that will be used to compare

    # 5. Compute the Structural Similarity Index (SSIM) between the two
    #    images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    #diff = (diff * 255).astype("uint8") #diff image needs to be uint16 and 16 bit?

    # 6. You can print only the score if you want
    #print("SSIM: {}".format(score))
    return score


# In[ ]:


def psnrfunc(img_orig, img_out):
    img_out = (img_out / ((2**16)-1)).astype(np.float32) #change to 255 for 8 bit pictures!!!
    img_orig = (img_orig / ((2**16)-1)).astype(np.float32)
    psnr = cv2.PSNR(img_out, img_orig)
    return psnr


# In[ ]:


def ms_SSIM(img_orig,img_out): #or MS psnr 
    #https://github.com/4og/mssim
    #MS_SSIM_val = ms_ssim( a, b, data_range=1, size_average=False )
    return 0


# In[ ]:


def text(Wtext,img):
    #https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
    
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # fontScale
    fontScale = 2
    # org
    img.shape[0]
    org = (50*fontScale,int(img.shape[0])-50*fontScale)
    #print(org)
    

    # Blue color in BGR
    color = (0, 0, ((2**16)-1))

    # Line thickness of 2 px
    thickness = 5

    # Using cv2.putText() method
    image = cv2.putText(img, Wtext, org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
    return image


# In[ ]:


def inputData():
    global csv_file #defines csv_file as a global variable
    global pathtoHDRgt
    global xlsPath
    global pathtoSDR
    global pathtoHDRgen
    global result
    global mashinePath
    
    print('This script stiches three pictures together in the Order: ')
    print('Ground Trouth, SDR inputpicture, Generated HDR Picture')
    result = input('Enter path for the results (default: '+result+'): ') or result
    
    if not os.path.exists(result):
        os.mkdir(result)
    xlsPath = input('Enter the path to save .xls file (deflaut: '+xlsPath+'): ') or xlsPath
    
    if not os.path.exists(xlsPath):
        os.mkdir(xlsPath)
    pathtoHDRgt = input('Enter the path to the Ground Truth Picture (deflaut: '+pathtoHDRgt+') ') or pathtoHDRgt
    if not os.path.exists(pathtoHDRgt):
        os.mkdir(pathtoHDRgt)
            
   ####TMO folder structure for tmo    tmo_path = ('/reinhard/','/mantiuk/','/drago/','/linear/')
    ### Maschine/reinhard/HDR/picture.png
    
    mashinePath = input('Enter the path to the generated picture folder (deflaut: '+mashinePath+') ') or mashinePath
    tmo = 0       
    while (tmo < len(tmo_path)):
        pth = mashinePath
        if not os.path.exists(pth):
            os.mkdir(pth)
        pth = mashinePath+tmo_path[tmo]
        if not os.path.exists(pth):
            os.mkdir(pth) 
        pthSDR = pth+'SDR'
        #pathtoSDR = input('Enter the path to the SDR inputpictures (deflaut: '+mashinePath+pathtoSDR+') ') or mashinePath+pathtoSDR
        if not os.path.exists(pthSDR):
            os.mkdir(pthSDR)
            
        pthHDR = pth+'HDRGen'
        #pathtoHDRgen = input('Enter the path to the Generated HDR Pictures (deflaut: '+mashinePath+') ') or mashinePath
        if not os.path.exists(pthHDR):
            os.mkdir(pthHDR)
        tmo = tmo + 1
    
inputData()


# In[ ]:


start_time = time.time() #start the timeing of the Prgramm
#finding the rigth picture pairs in paths
global data
data_temp = ''
i = 0
wb = Workbook()

reinhard = wb.add_sheet('reinhard')
mantiuk = wb.add_sheet('mantiuk')
drago = wb.add_sheet('drago')
linear = wb.add_sheet('linear')
firstRun = True;
try:
    while(len(os.listdir(pathtoHDRgt)) > i ):
        if((os.listdir(pathtoHDRgt)[i].split('.')[1]) != 'png'):
            i = i + 1    
        else:
            currentFileName = os.listdir(pathtoHDRgt)[i]
            HDRgtPic = readThePicture(pathtoHDRgt+currentFileName)
            
            ###Generated SDR section
            picTure = HDRgtPic #adds the Picture to the patch Pic
            tmo = 0
            while (tmo < len(tmo_path)):
                try:
                    
                    print(tmo)
                    pth_local = mashinePath+tmo_path[tmo]+'SDR/'+currentFileName
                    SDRpic = readThePicture(pth_local)
                    SDRpic = SDRpic/((2**8)-1) #dividing by 8 bit to norm to 0,1
                    SDRpic = (SDRpic*((2**16)-1)).astype(np.uint16)
                except:
                    print('fail to find: '+ pth_local)
                    
            ###Generated HDR section
                try:
                    pth_local = mashinePath+tmo_path[tmo]+'HDRGen/'+currentFileName
                    HDRpic = readThePicture(pth_local)
                except:
                    print('fail to find: ' + pth_local)
                    
                if(HDRgtPic.shape != SDRpic[tmo].shape): ##upsizing if needed
                    factor = int(HDRgtPic.shape[1]/SDRpic.shape[1])
                    SDRpic = scale(SDRpic,factor)

                
                ####Data Section
                ##Definition where the data should be stored 
                pos = (0,0, 'Picture number')
                ssim_HDRgt_HDR = (0,1, 'ssim_HDRgt_HDR')
                ssim_HDRgt_SDR = (0,2,'ssim_HDRgt_SDR')
                ssim_SDR_HDR = (0,3,'ssim_SDR_HDR')
                
                psnr_HDRgt_HDR = (0,4,'psnr_HDRgt_HDR')
                psnr_HDRgt_SDR = (0,5,'psnr_HDRgt_SDR')
                psnr_SDR_HDR = (0,6,'psnr_SDR_HDR')
                
                hue_HDR = (0,7,'Hue_HDR')
                sat_HDR = (0,8,'Saturation_HDR')
                li_HDR = (0,9,'Lightness_HDR')
                
                hue_GT = (0,10,'Hue_GT')
                sat_GT = (0,11,'Saturation_GT')
                li_GT = (0,12,'Lightness_GT')
                
                hue_SDR = (0,13,'Hue_SDR')
                sat_SDR = (0,14,'Saturation_SDR')
                li_SDR = (0,15,'Lightness_SDR')
                
                ###SSIM Analyses
                S_GTHDR = ssim(HDRgtPic,HDRpic) #(image that will be used to compare,image that will be compared)
                S_GTSDR = ssim(HDRgtPic,SDRpic)
                S_HDSDR = ssim(HDRpic,SDRpic)
                
                
                ###PSNR Analyses
                P_GTHDR = psnrfunc(HDRgtPic,HDRpic)
                P_GTSDR = psnrfunc(HDRgtPic,SDRpic)
                P_HDSDR = psnrfunc(HDRpic,SDRpic)
                
                ###HSL Analyses
                Hue,Lightness,Saturation = HLS(HDRpic) #Farbton (Hue), Farbs??ttigung (Saturation) und Helligkeit (Lightness)
                SDR_Hue,SDR_Lightness,SDR_Saturation = HLS(SDRpic)
                GT_Hue,GT_Lightness,GT_Saturation = HLS(HDRgtPic)
                
                
                ##Reinhard save datapath
                if(tmo_path[tmo]=='reinhard/'):
                    
                    if(firstRun):
                        #picName
                        reinhard.write(pos[0], pos[1],pos[2])
                        #SSIM
                        reinhard.write(ssim_HDRgt_HDR[0],ssim_HDRgt_HDR[1],ssim_HDRgt_HDR[2])
                        reinhard.write(ssim_HDRgt_SDR[0],ssim_HDRgt_SDR[1],ssim_HDRgt_SDR[2])
                        reinhard.write(ssim_SDR_HDR[0],ssim_SDR_HDR[1],ssim_SDR_HDR[2])
                        #PSNR
                        reinhard.write(psnr_HDRgt_HDR[0],psnr_HDRgt_HDR[1],psnr_HDRgt_HDR[2])
                        reinhard.write(psnr_HDRgt_SDR[0],psnr_HDRgt_SDR[1],psnr_HDRgt_SDR[2])
                        reinhard.write(psnr_SDR_HDR[0],psnr_SDR_HDR[1],psnr_SDR_HDR[2])
                        #H
                        reinhard.write(hue_HDR[0],hue_HDR[1],hue_HDR[2])
                        reinhard.write(hue_GT[0], hue_GT[1], hue_GT[2])
                        reinhard.write(hue_SDR[0], hue_SDR[1], hue_SDR[2])
                        ##S
                        reinhard.write(sat_HDR[0],sat_HDR[1],sat_HDR[2])
                        reinhard.write(sat_GT[0], sat_GT[1], sat_GT[2])
                        reinhard.write(sat_SDR[0], sat_SDR[1], sat_SDR[2])
                        ##L
                        reinhard.write(li_HDR[0],li_HDR[1],li_HDR[2])
                        reinhard.write(li_GT[0], li_GT[1], li_GT[2])
                        reinhard.write(li_SDR[0], li_SDR[1], li_SDR[2])

                        
                    ###number
                    
                    reinhard.write(i+1,0, currentFileName.split('.')[0])

                    ####SSIM
                    
                    reinhard.write(i+1,1, S_GTHDR)
                    reinhard.write(i+1,2, S_GTSDR)
                    reinhard.write(i+1,3, S_HDSDR)
                    
                    ####psnr
                    
                    reinhard.write(i+1,4, P_GTHDR)
                    reinhard.write(i+1,5, P_GTSDR)
                    reinhard.write(i+1,6, P_HDSDR)
                    
                    ####HSL
                    reinhard.write(i+1,7, Hue)
                    reinhard.write(i+1,8, Saturation) 
                    reinhard.write(i+1,9, Lightness)
                    
                    reinhard.write(i+1,10, GT_Hue)
                    reinhard.write(i+1,11, GT_Lightness)
                    reinhard.write(i+1,12, GT_Saturation)
                    
                    reinhard.write(i+1,13, SDR_Hue)
                    reinhard.write(i+1,14, SDR_Lightness)
                    reinhard.write(i+1,15, SDR_Saturation)
                    
                elif(tmo_path[tmo] == 'mantiuk/'):
                    
                    if(firstRun):
                        #Pos
                        mantiuk.write(pos[0], pos[1],pos[2])
                        #SSIM
                        mantiuk.write(ssim_HDRgt_HDR[0],ssim_HDRgt_HDR[1],ssim_HDRgt_HDR[2])
                        mantiuk.write(ssim_HDRgt_SDR[0],ssim_HDRgt_SDR[1],ssim_HDRgt_SDR[2])
                        mantiuk.write(ssim_SDR_HDR[0],ssim_SDR_HDR[1],ssim_SDR_HDR[2])
                        ##PSNR
                        mantiuk.write(psnr_HDRgt_HDR[0],psnr_HDRgt_HDR[1],psnr_HDRgt_HDR[2])
                        mantiuk.write(psnr_HDRgt_SDR[0],psnr_HDRgt_SDR[1],psnr_HDRgt_SDR[2])
                        mantiuk.write(psnr_SDR_HDR[0],psnr_SDR_HDR[1],psnr_SDR_HDR[2])
                        #H
                        mantiuk.write(hue_HDR[0],hue_HDR[1],hue_HDR[2])
                        mantiuk.write(hue_GT[0], hue_GT[1], hue_GT[2])
                        mantiuk.write(hue_SDR[0], hue_SDR[1], hue_SDR[2])
                        ##S
                        mantiuk.write(sat_HDR[0],sat_HDR[1],sat_HDR[2])
                        mantiuk.write(sat_GT[0], sat_GT[1], sat_GT[2])
                        mantiuk.write(sat_SDR[0], sat_SDR[1], sat_SDR[2])
                        ##L
                        mantiuk.write(li_HDR[0],li_HDR[1],li_HDR[2])
                        mantiuk.write(li_GT[0], li_GT[1], li_GT[2])
                        mantiuk.write(li_SDR[0], li_SDR[1], li_SDR[2])

                        
                    ###number
                    
                    mantiuk.write(i+1,0, currentFileName.split('.')[0])
                    
                    ####SSIM
                    
                    mantiuk.write(i+1,1, S_GTHDR)
                    mantiuk.write(i+1,2, S_GTSDR)
                    mantiuk.write(i+1,3, S_HDSDR)
                    
                    ####psnr
                    
                    mantiuk.write(i+1,4, P_GTHDR)
                    mantiuk.write(i+1,5, P_GTSDR)
                    mantiuk.write(i+1,6, P_HDSDR)
                    
                    ####HLS
                    
                    mantiuk.write(i+1,7, Hue)
                    mantiuk.write(i+1,8, Saturation) 
                    mantiuk.write(i+1,9, Lightness)
                    
                    mantiuk.write(i+1,10, GT_Hue)
                    mantiuk.write(i+1,11, GT_Lightness)
                    mantiuk.write(i+1,12, GT_Saturation)
                    
                    mantiuk.write(i+1,13, SDR_Hue)
                    mantiuk.write(i+1,14, SDR_Lightness)
                    mantiuk.write(i+1,15, SDR_Saturation)
                    
                elif(tmo_path[tmo]=='drago/'):
                    if(firstRun):
                        #POS
                        drago.write(pos[0], pos[1],pos[2])
                        ##SSIM
                        drago.write(ssim_HDRgt_HDR[0],ssim_HDRgt_HDR[1],ssim_HDRgt_HDR[2])
                        drago.write(ssim_HDRgt_SDR[0],ssim_HDRgt_SDR[1],ssim_HDRgt_SDR[2])
                        drago.write(ssim_SDR_HDR[0],ssim_SDR_HDR[1],ssim_SDR_HDR[2])
                        ##PSNR
                        drago.write(psnr_HDRgt_HDR[0],psnr_HDRgt_HDR[1],psnr_HDRgt_HDR[2])
                        drago.write(psnr_HDRgt_SDR[0],psnr_HDRgt_SDR[1],psnr_HDRgt_SDR[2])
                        drago.write(psnr_SDR_HDR[0],psnr_SDR_HDR[1],psnr_SDR_HDR[2])
                        #H
                        drago.write(hue_HDR[0],hue_HDR[1],hue_HDR[2])
                        drago.write(hue_GT[0], hue_GT[1], hue_GT[2])
                        drago.write(hue_SDR[0], hue_SDR[1], hue_SDR[2])
                        ##S
                        drago.write(sat_HDR[0],sat_HDR[1],sat_HDR[2])
                        drago.write(sat_GT[0], sat_GT[1], sat_GT[2])
                        drago.write(sat_SDR[0], sat_SDR[1], sat_SDR[2])
                        ##L
                        drago.write(li_HDR[0],li_HDR[1],li_HDR[2])
                        drago.write(li_GT[0], li_GT[1], li_GT[2])
                        drago.write(li_SDR[0], li_SDR[1], li_SDR[2])
                    
                    ###number
                    
                    drago.write(i+1,0, currentFileName.split('.')[0])
                    
                    ####SSIM
                    
                    drago.write(i+1,1, S_GTHDR)
                    drago.write(i+1,2, S_GTSDR)
                    drago.write(i+1,3, S_HDSDR)
                    
                    ####psnr
                    
                    drago.write(i+1,4, P_GTHDR)
                    drago.write(i+1,5, P_GTSDR)
                    drago.write(i+1,6, P_HDSDR)
                    
                    ####HLS
                    
                    drago.write(i+1,7, Hue)
                    drago.write(i+1,8, Saturation) 
                    drago.write(i+1,9, Lightness)
                    
                    drago.write(i+1,10, GT_Hue)
                    drago.write(i+1,11, GT_Lightness)
                    drago.write(i+1,12, GT_Saturation)
                    
                    drago.write(i+1,13, SDR_Hue)
                    drago.write(i+1,14, SDR_Lightness)
                    drago.write(i+1,15, SDR_Saturation)
                    
                elif(tmo_path[tmo]=='linear/'):
                    if(firstRun):
                        #Pos
                        linear.write(pos[0], pos[1],pos[2])
                        ##SSIM
                        linear.write(ssim_HDRgt_HDR[0],ssim_HDRgt_HDR[1],ssim_HDRgt_HDR[2])
                        linear.write(ssim_HDRgt_SDR[0],ssim_HDRgt_SDR[1],ssim_HDRgt_SDR[2])
                        linear.write(ssim_SDR_HDR[0],ssim_SDR_HDR[1],ssim_SDR_HDR[2])
                        ##PSNR
                        linear.write(psnr_HDRgt_HDR[0],psnr_HDRgt_HDR[1],psnr_HDRgt_HDR[2])
                        linear.write(psnr_HDRgt_SDR[0],psnr_HDRgt_SDR[1],psnr_HDRgt_SDR[2])
                        linear.write(psnr_SDR_HDR[0],psnr_SDR_HDR[1],psnr_SDR_HDR[2])
                        
                        #H
                        linear.write(hue_HDR[0],hue_HDR[1],hue_HDR[2])
                        linear.write(hue_GT[0], hue_GT[1], hue_GT[2])
                        linear.write(hue_SDR[0], hue_SDR[1], hue_SDR[2])
                        ##S
                        linear.write(sat_HDR[0],sat_HDR[1],sat_HDR[2])
                        linear.write(sat_GT[0], sat_GT[1], sat_GT[2])
                        linear.write(sat_SDR[0], sat_SDR[1], sat_SDR[2])
                        ##L
                        linear.write(li_HDR[0],li_HDR[1],li_HDR[2])
                        linear.write(li_GT[0], li_GT[1], li_GT[2])
                        linear.write(li_SDR[0], li_SDR[1], li_SDR[2])
                        
                        firstRun = False
                    
                    ###number
                    
                    linear.write(i+1,0, currentFileName.split('.')[0])
                    
                    ####SSIM
                    
                    linear.write(i+1,1, S_GTHDR)
                    linear.write(i+1,2, S_GTSDR)
                    linear.write(i+1,3, S_HDSDR)
                    
                    ####psnr
                    
                    linear.write(i+1,4, P_GTHDR)
                    linear.write(i+1,5, P_GTSDR)
                    linear.write(i+1,6, P_HDSDR)
                    
                    ####HLS
                    
                    linear.write(i+1,7, Hue)
                    linear.write(i+1,8, Saturation) 
                    linear.write(i+1,9, Lightness)
                    
                    linear.write(i+1,10, GT_Hue)
                    linear.write(i+1,11, GT_Lightness)
                    linear.write(i+1,12, GT_Saturation)
                    
                    linear.write(i+1,13, SDR_Hue)
                    linear.write(i+1,14, SDR_Lightness)
                    linear.write(i+1,15, SDR_Saturation)
                        
                wb.save(xlsPath+str(mashinePath.split('/')[1])+'.xls')     
            ###Text in picture Setting 
                if(tmo == 0):
                    picTure = text(str(currentFileName.split('.')[0])+'GT_HDR',picTure)
                HDRpic = text(str(mashinePath.split('/')[1])+'_GenHDR_'+tmo_path[tmo].split('/')[0],HDRpic)    
                SDRpic = text(tmo_path[tmo].split('/')[0]+ '_SDR',SDRpic)
                
            ###Output Picture section
                
                if(tmo == 0): #generating a white Picture to add to the right for the gt 
                    white = (np.zeros((picTure.shape))+1*((2**16)-1)).astype(np.uint16)
                    picTure = horStack(picTure,white)
                
                tmoPic = horStack(SDRpic,HDRpic)
                picTure = vertStack(picTure,tmoPic) #adding pictures to make a 2x4 matrix picture
                tmo = tmo + 1
            
            savePic(picTure,(str(currentFileName.split('.')[0])+'GT_RMDL'),'png',result)
            i = i+1
           
        
except: 
    print('There was an error while finding the pictures to compare') 
print('Finished and it took: '+str((time.time() - start_time)/60)+'minutes')


# In[ ]:


exit()


# In[ ]:





# In[ ]:




