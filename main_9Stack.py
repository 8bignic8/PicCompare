#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import math
from xlwt import Workbook
#from psnrhvsm import psnrhvsm


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


def readCSV(file):
    #https://docs.python.org/3/library/csv.html
    print('Read data to '+str(name))
    result = np.array(list(csv.reader(open(file, "rb"), delimiter=","))).astype("float")
    return result


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
def ssim(imageA,imageB):
    # 4. Convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY) #image that will be compared
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY) #image that will be used to compare

    # 5. Compute the Structural Similarity Index (SSIM) between the two
    #    images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8") #diff image needs to be uint16 and 16 bit?

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


##https://pypi.org/project/psnr-hvsm/
def mpsnr(a,b):
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
                
                mpsnr_HDRgt_HDR = (0,7,'mpsnr_HDRgt_HDR')
                mpsnr_HDRgt_SDR = (0,8,'mpsnr_HDRgt_SDR')
                mpsnr_SDR_HDR = (0,9,'mpsnr_SDR_HDR')
                
                S_GTHDR = ssim(HDRgtPic,HDRpic)
                S_GTSDR = ssim(HDRgtPic,SDRpic)
                S_HDSDR = ssim(HDRpic,SDRpic)
                
                P_GTHDR = psnrfunc(HDRgtPic,HDRpic)
                P_GTSDR = psnrfunc(HDRgtPic,SDRpic)
                P_HDSDR = psnrfunc(HDRpic,SDRpic)

                mP_GTHDR = mpsnr(HDRgtPic,HDRpic)
                mP_GTSDR = mpsnr(HDRgtPic,SDRpic)
                mP_HDSDR = mpsnr(HDRpic,SDRpic)
                
                ##Reinhard save datapath
                if(tmo_path[tmo]=='reinhard/'):
                    
                    if(firstRun):
                        reinhard.write(pos[0], pos[1],pos[2])
                        reinhard.write(ssim_HDRgt_HDR[0],ssim_HDRgt_HDR[1],ssim_HDRgt_HDR[2])
                        reinhard.write(ssim_HDRgt_SDR[0],ssim_HDRgt_SDR[1],ssim_HDRgt_SDR[2])
                        reinhard.write(ssim_SDR_HDR[0],ssim_SDR_HDR[1],ssim_SDR_HDR[2])
                        reinhard.write(psnr_HDRgt_HDR[0],psnr_HDRgt_HDR[1],psnr_HDRgt_HDR[2])
                        reinhard.write(psnr_HDRgt_SDR[0],psnr_HDRgt_SDR[1],psnr_HDRgt_SDR[2])
                        reinhard.write(psnr_SDR_HDR[0],psnr_SDR_HDR[1],psnr_SDR_HDR[2])
                        reinhard.write(mpsnr_HDRgt_HDR[0],mpsnr_HDRgt_HDR[1],mpsnr_HDRgt_HDR[2])
                        reinhard.write(mpsnr_HDRgt_SDR[0],mpsnr_HDRgt_SDR[1],mpsnr_HDRgt_SDR[2])
                        reinhard.write(mpsnr_SDR_HDR[0],mpsnr_SDR_HDR[1],mpsnr_SDR_HDR[2])
                        
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
                    
                    ####mpsnr
                    
                    reinhard.write(i+1,7, mP_GTHDR)
                    reinhard.write(i+1,8, mP_GTSDR) 
                    reinhard.write(i+1,9, mP_HDSDR)

                elif(tmo_path[tmo] == 'mantiuk/'):
                    
                    if(firstRun):
                        mantiuk.write(pos[0], pos[1],pos[2])
                        mantiuk.write(ssim_HDRgt_HDR[0],ssim_HDRgt_HDR[1],ssim_HDRgt_HDR[2])
                        mantiuk.write(ssim_HDRgt_SDR[0],ssim_HDRgt_SDR[1],ssim_HDRgt_SDR[2])
                        mantiuk.write(ssim_SDR_HDR[0],ssim_SDR_HDR[1],ssim_SDR_HDR[2])
                        mantiuk.write(psnr_HDRgt_HDR[0],psnr_HDRgt_HDR[1],psnr_HDRgt_HDR[2])
                        mantiuk.write(psnr_HDRgt_SDR[0],psnr_HDRgt_SDR[1],psnr_HDRgt_SDR[2])
                        mantiuk.write(psnr_SDR_HDR[0],psnr_SDR_HDR[1],psnr_SDR_HDR[2])
                        mantiuk.write(mpsnr_HDRgt_HDR[0],mpsnr_HDRgt_HDR[1],mpsnr_HDRgt_HDR[2])
                        mantiuk.write(mpsnr_HDRgt_SDR[0],mpsnr_HDRgt_SDR[1],mpsnr_HDRgt_SDR[2])
                        mantiuk.write(mpsnr_SDR_HDR[0],mpsnr_SDR_HDR[1],mpsnr_SDR_HDR[2])
                        
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
                    
                    ####mpsnr
                    
                    mantiuk.write(i+1,7, mP_GTHDR)
                    mantiuk.write(i+1,8, mP_GTSDR) 
                    mantiuk.write(i+1,9, mP_HDSDR)
                    
                elif(tmo_path[tmo]=='drago/'):
                    if(firstRun):
                        drago.write(pos[0], pos[1],pos[2])
                        drago.write(ssim_HDRgt_HDR[0],ssim_HDRgt_HDR[1],ssim_HDRgt_HDR[2])
                        drago.write(ssim_HDRgt_SDR[0],ssim_HDRgt_SDR[1],ssim_HDRgt_SDR[2])
                        drago.write(ssim_SDR_HDR[0],ssim_SDR_HDR[1],ssim_SDR_HDR[2])
                        drago.write(psnr_HDRgt_HDR[0],psnr_HDRgt_HDR[1],psnr_HDRgt_HDR[2])
                        drago.write(psnr_HDRgt_SDR[0],psnr_HDRgt_SDR[1],psnr_HDRgt_SDR[2])
                        drago.write(psnr_SDR_HDR[0],psnr_SDR_HDR[1],psnr_SDR_HDR[2])
                        drago.write(mpsnr_HDRgt_HDR[0],mpsnr_HDRgt_HDR[1],mpsnr_HDRgt_HDR[2])
                        drago.write(mpsnr_HDRgt_SDR[0],mpsnr_HDRgt_SDR[1],mpsnr_HDRgt_SDR[2])
                        drago.write(mpsnr_SDR_HDR[0],mpsnr_SDR_HDR[1],mpsnr_SDR_HDR[2])
                    
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
                    
                    ####mpsnr
                    
                    drago.write(i+1,7, mP_GTHDR)
                    drago.write(i+1,8, mP_GTSDR) 
                    drago.write(i+1,9, mP_HDSDR)
                    
                elif(tmo_path[tmo]=='linear/'):
                    if(firstRun):
                        linear.write(pos[0], pos[1],pos[2])
                        linear.write(ssim_HDRgt_HDR[0],ssim_HDRgt_HDR[1],ssim_HDRgt_HDR[2])
                        linear.write(ssim_HDRgt_SDR[0],ssim_HDRgt_SDR[1],ssim_HDRgt_SDR[2])
                        linear.write(ssim_SDR_HDR[0],ssim_SDR_HDR[1],ssim_SDR_HDR[2])
                        linear.write(psnr_HDRgt_HDR[0],psnr_HDRgt_HDR[1],psnr_HDRgt_HDR[2])
                        linear.write(psnr_HDRgt_SDR[0],psnr_HDRgt_SDR[1],psnr_HDRgt_SDR[2])
                        linear.write(psnr_SDR_HDR[0],psnr_SDR_HDR[1],psnr_SDR_HDR[2])
                        linear.write(mpsnr_HDRgt_HDR[0],mpsnr_HDRgt_HDR[1],mpsnr_HDRgt_HDR[2])
                        linear.write(mpsnr_HDRgt_SDR[0],mpsnr_HDRgt_SDR[1],mpsnr_HDRgt_SDR[2])
                        linear.write(mpsnr_SDR_HDR[0],mpsnr_SDR_HDR[1],mpsnr_SDR_HDR[2])
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
                    
                    ####mpsnr
                    
                    linear.write(i+1,7, mP_GTHDR)
                    linear.write(i+1,8, mP_GTSDR) 
                    linear.write(i+1,9, mP_HDSDR)
                    
                    
                wb.save(xlsPath+str(mashinePath.split('/')[1])+'.xls')     
            ###Text in picture Setting 
                if(tmo == 0):
                    picTure = text(str(currentFileName.split('.')[0])+'GT_HDR',picTure)
                HDRpic = text(str(mashinePath.split('/')[1])+'_GenHDR_'+tmo_path[tmo].split('/')[0],HDRpic)    
                SDRpic = text(tmo_path[tmo].split('/')[0]+ '_SDR',SDRpic)
                
            ###Output Picture section
                    
                picTure = horStack(picTure,SDRpic)
                picTure = horStack(picTure,HDRpic)
                tmo = tmo + 1
            
            savePic(picTure,(str(currentFileName.split('.')[0])+'GT_RMDL'),'png',result)
            i = i+1
            
            #toDo
            #if(colorSpaceAnal == True ):
            #    savePic(picTure[:,:,0]*255,(str(i)+'Result_ColorSpace'+picName),'png',result)
             #   savePic(picTure[],(str(i)+'Result_ColorSpace'+picName),'png',result)
              #  savePic(picTure[],(str(i)+'Result_ColorSpace'+picName),'png',result)
        
except: 
    print('There was an error while finding the pictures to compare')
    print('Picture name to find: '+picName) 
print('Finished and it took: '+str((time.time() - start_time)/60)+'minutes')

